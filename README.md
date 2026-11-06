# Intelligent Face Tracker with Auto-Registration & Unique Visitor Counting

Real-time pipeline that detects faces (YOLOv8-face), tracks them, recognizes them with InsightFace/ArcFace,
auto-registers new people, logs **exactly one entry and one exit per visit** (crop image + timestamp + DB row),
and maintains a **unique visitor count** that does not grow when the same person returns.

## Demo video
**Loom / YouTube:** `<ADD LINK HERE>`  (required by the evaluators)

## Architecture
```mermaid
flowchart LR
    A[Video file / RTSP] --> B[FrameSource<br/>threaded + auto-reconnect]
    B --> C{frame % skip+1 == 0?}
    C -- yes --> D[YOLOv8-face detector]
    C -- no --> E[Track prediction only]
    D --> F[ByteTrack-style tracker<br/>IoU + velocity, 2-stage association]
    E --> F
    F --> G{New confirmed track?}
    G -- yes --> H[Quality gate -> crop -> InsightFace ArcFace embedding]
    H --> I{cosine sim >= threshold?}
    I -- match --> J[Re-identify existing face_id]
    I -- no match x N good frames --> K[Register new face_id]
    J --> L[ENTRY event]
    K --> L
    F -- track lost / left frame --> M[EXIT event]
    L --> N[(SQLite: faces, embeddings, events)]
    M --> N
    L --> O[logs/entries/YYYY-MM-DD/*.jpg]
    M --> P[logs/exits/YYYY-MM-DD/*.jpg]
    L --> Q[logs/events.log]
    M --> Q
```

## AI planning document
**Plan.** (1) Frame ingestion robust to RTSP drops, (2) detect every N frames, (3) track in between,
(4) embed only when a track is new (cheap), (5) gallery matching + safe registration, (6) event state machine,
(7) triple-sink logging (log file, image store, DB), (8) unique count = `COUNT(*) FROM faces`.

**Features.**
- YOLOv8-face detection with configurable frame skipping (`detection_skip_frames` in `config.json`)
- ByteTrack-style tracker (high/low confidence association, velocity coasting through skipped frames)
- InsightFace `buffalo_l` ArcFace 512-d embeddings, cosine matching against a gallery (up to 5 embeddings/face)
- Auto-registration with unique ID; needs `min_embeds_to_register` good, mutually-unmatched embeddings to avoid false new IDs
- Quality gate (min face size, blur) so poor frames never create identities
- Exactly-once entry/exit via an ownership map (`present`), with track hand-over when a track fragments
- Re-entry of a known person logs a new entry/exit but does **not** change the unique count
- End-of-stream flush closes every open visit; atomic image writes; SQLite WAL; per-frame exception guard
- Annotated output video and optional live window

**Estimated compute (approximate, measure with the FPS in `events.log` `STATS` lines).**

| Component | CPU-only (modern 4–8 core) | GPU (e.g. RTX 3050/3060 class) |
|---|---|---|
| YOLOv8n-face @640 | ~40–80 ms per detection frame | ~5–10 ms |
| InsightFace det+ArcFace per embed | ~60–150 ms (only on new tracks) | ~10–20 ms |
| Tracker + logging | <2 ms / frame | <2 ms / frame |
| Expected throughput (skip=2) | ~15–25 FPS | real-time 25–30 FPS with headroom |

Skipping frames divides detector cost by (skip+1); embeddings run on a handful of frames per person, not every frame.
Memory: ~1–2 GB (models + runtime). Use `onnxruntime-gpu` and a CUDA build of PyTorch to enable GPU.

## Analytics dashboard (React + TypeScript)
A read-only web dashboard over the same SQLite DB, image store and `events.log`:
- **Overview** - KPI cards (unique visitors, currently inside, entries/exits, average time on camera), activity chart, new-vs-returning split.
- **Visitors** - searchable gallery of registered faces. Open a visitor for **Face & Embedding**: the registered close-up crop, every stored
  embedding with the close-up it came from, a 512-value heatmap + distribution + raw values (copyable), and the closest other identities
  with cosine similarity against the match threshold. A second tab shows that visitor's entry/exit timeline.
- **Event Log** - paginated, filterable entry/exit table with face crops.
- **System Log** - live, filterable view of `events.log` (registration, recognition, embedding, tracking, entry/exit), colour-coded, with clickable face IDs.

Data refreshes every 3 s (pause button in the top bar).

```bash
pip install -r requirements-api.txt
cd web && npm install && npm run build && cd ..      # one-time build
uvicorn api.server:app --port 8000                   # run from the repo root -> open http://localhost:8000
```
Dev mode with hot reload: run `uvicorn` as above, then `cd web && npm run dev` (http://localhost:5173, proxies to :8000).
The dashboard can run while `main.py` is processing (SQLite WAL allows concurrent readers).

No results yet? Preview with clearly fake demo data (written to `demo/`, never touches your real DB):
```bash
python scripts/seed_demo.py
TRACKER_DB=demo/visitors.db TRACKER_LOG_DIR=demo/logs uvicorn api.server:app --port 8000
```
Stack: FastAPI (`api/server.py`) -> React 18 + TypeScript + Vite + Recharts (`web/`). Endpoints: `/api/summary`, `/api/timeline`,
`/api/visitors`, `/api/visitors/{id}`, `/api/visitors/{id}/embeddings`, `/api/events`, `/api/logs`, images under `/media`.

## Developing further (Cursor / AI tools)
Give `docs/KNOWLEDGE.md` to your AI assistant (rules file already at `.cursor/rules/project.mdc`). It lists what is built, invariants, known gaps and the prioritized next steps.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # GPU: replace onnxruntime with onnxruntime-gpu
# put the provided sample video at data/sample_video.mp4  (or set video_source in config.json)
python main.py --config config.json
```
- YOLO face weights auto-download to `models/yolov8n-face.pt` on first run (or place any YOLOv8 face model there).
- InsightFace `buffalo_l` models auto-download to `~/.insightface` on first run.
- On Windows, `insightface` may need *Microsoft C++ Build Tools*; Python 3.9–3.11 is the safest.

**Multiple videos, one combined unique count:**
```bash
python main.py --sources data/video1.mp4 data/video2.mp4 data/video3.mp4
```
Videos run in order as a single session: one DB (reset once at the start), one face gallery, so a person seen in several videos
is counted once. Each video's open visits are closed at its end (exit events), and a person reappearing in the next video logs a
new entry under the same `face_id`. Annotated videos are written per source (`output/annotated_<video>.mp4`).
Use `--no-reset` to keep an existing DB, or list the files under `"video_sources"` in `config.json`.

**RTSP (interview):** `python main.py --source rtsp://user:pass@host:554/stream --display`

**Results / export:** `python scripts/report.py --db data/visitors.db --csv-dir samples`
**Tests (no models needed):** `pytest -q`

## config.json
```json
{
  "video_source": "data/sample_video.mp4",
  "detection_skip_frames": 2,
  "detector":    { "weights": "models/yolov8n-face.pt", "conf_threshold": 0.30, "imgsz": 640, "device": "auto" },
  "recognition": { "model_pack": "buffalo_l", "match_threshold": 0.45, "min_embeds_to_register": 2,
                   "embed_retry_interval": 2, "max_embed_attempts": 10, "max_embeddings_per_face": 5 },
  "quality":     { "min_face_px": 45, "min_blur_var": 35.0 },
  "tracker":     { "high_conf": 0.5, "new_track_conf": 0.5, "iou_threshold": 0.25, "min_hits": 2, "max_lost_frames": 20 },
  "storage":     { "db_path": "data/visitors.db", "log_dir": "logs", "events_log": "logs/events.log", "reset_on_start": true },
  "stream":      { "reconnect_delay_sec": 3, "max_reconnects": 0, "read_timeout_sec": 10 },
  "output":      { "display": false, "annotated_video": "output/annotated.mp4" }
}
```
`detection_skip_frames = N` means: run YOLO on one frame, skip the next N. `max_lost_frames` is auto-raised if too small for N.

## Output layout
```
logs/events.log                         # ENTRY, EXIT, FACE_RECOGNIZED, FACE_REGISTERED, EMBEDDING_GENERATED, TRACK_STARTED/ENDED ...
logs/entries/YYYY-MM-DD/entry_face0001_HHMMSS_ffffff.jpg
logs/exits/YYYY-MM-DD/exit_face0001_HHMMSS_ffffff.jpg
logs/faces/YYYY-MM-DD/face_face0001_HHMMSS_ffffff.jpg   # tight close-up saved with every stored embedding
data/visitors.db                        # tables: faces (+close-up path), embeddings (+crop path), events
```
Unique count: `SELECT COUNT(*) FROM faces;`

## Assumptions
- A "visit" starts when a track gets an identity and ends when the track is lost for `max_lost_frames` frames or leaves the frame; a person who returns later gets a new entry/exit pair under the same `face_id`.
- Faces below `min_face_px` or too blurry are ignored for identity (they are never counted).
- One camera; timestamps are local wall-clock time of processing.
- `reset_on_start: true` gives a clean DB per run for evaluation; set `false` to keep identities across restarts.
- `match_threshold` (0.45) is a starting point for `buffalo_l`; tune on your footage (higher = more unique IDs, lower = more merges).

## Sample output
Generated from the provided video (see `samples/`, `logs/`): `<ADD after running: counts, a few crops, DB export>`

---
This project is a part of a hackathon run by https://katomaran.com
