# VYRON — Intelligent Face Tracker (powered by ARIA)

Real-time pipeline that detects faces (YOLOv8-face + RetinaFace assist), tracks them, recognizes them with InsightFace/ArcFace,
auto-registers new people, logs **exactly one entry and one exit per visit** (crop image + timestamp + DB row),
and maintains a **unique visitor count** that does not grow when the same person returns.

## Demo video
**Loom / YouTube:** `<ADD LINK HERE>`  (required by the evaluators — submission without a video is not reviewed)

## Project structure
```
backend/          Python tracker + FastAPI + scripts + tests (package names src/api unchanged)
frontend/         Vite React dashboard (src/api, pages, components, public/brand)
data/             videos, models, db, runtime, demo (local; mostly gitignored)
deploy/           docker-compose Postgres, MediaMTX, local certs
docs/             KNOWLEDGE, ARCHITECTURE, PROMPTS, assets/
logs/             session events + crops (submission deliverable; stays at root)
samples/          CSV / export samples (stays at root)
config.json       runtime config (stays at root so paths match PROJECT_ROOT)
main.py           thin shim → backend/main.py (same CLI flags)
Makefile          install / run / api / web / test helpers
```
Root keeps only entry/config/docs-adjacent files so `python3 main.py` and relative log paths stay stable. Python imports still use `src.*` / `api.*` with `pythonpath=backend`.

## Architecture

Full system diagram (sources → preprocess → detect/track → identity → Postgres/logs → dashboard + ARIA):

![VYRON architecture](docs/assets/vyron-architecture.png)

More detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`docs/KNOWLEDGE.md`](docs/KNOWLEDGE.md).

```mermaid
flowchart LR
    A[Webcam / file / RTSP] --> B[FrameSource<br/>original frames]
    B --> PP[Optional async preprocess<br/>downscale → dull/dark gate → filters]
    PP --> S{Enhanced ready?}
    B --> S
    S -->|yes| D[YOLOv8-face + RetinaFace assist]
    S -->|no / off| D
    D --> M[Map boxes to original]
    M --> F[ByteTrack-style tracker]
    F --> H[Blur gate → crop from original → InsightFace]
    H --> I{cosine match?}
    I -- yes --> J[FACE_RECOGNIZED]
    I -- no x N --> K[FACE_REGISTERED]
    J --> L[ENTRY / EXIT · 1 per visit]
    K --> L
    L --> N[(PostgreSQL)]
    L --> Q[logs/ + crops]
    N --> UI[React dashboard + ARIA]
    Q --> UI
```

**Invariant:** enhanced frames feed the **detector only**. Crops, embeddings, Live Monitor, away, and disturbance always use the **original** frame so identity and evidence do not drift.

### Presence / visit state machine
```mermaid
stateDiagram-v2
    [*] --> PRESENT: ENTRY
    PRESENT --> HIDDEN: track lost (grace by edge/stationary/occlusion)
    PRESENT --> BODY_ONLY: face gone, person box still overlaps
    HIDDEN --> PRESENT: VISIT_RESUMED
    BODY_ONLY --> PRESENT: face returns
    HIDDEN --> EXITED: grace timeout / stream_gap / stream_end
    BODY_ONLY --> EXITED: body also gone past grace
    EXITED --> [*]
```
Grace is **seconds** (see `presence.*` in config). EXIT timestamps use the last real detection (`last_seen_ts`), not removal time. `time_in_frame = visible_seconds + occluded_seconds`.

## AI planning document
**Plan.** (1) Frame ingestion robust to RTSP drops, (2) detect every N frames, (3) track in between,
(4) embed only when a track is new (cheap), (5) gallery matching + safe registration, (6) presence visit state machine,
(7) triple-sink logging (log file, image store, DB), (8) unique count = `COUNT(*) FROM faces`.

**Features.**
- YOLOv8-face detection (+ optional RetinaFace assist) with configurable frame skipping (`detection_skip_frames`)
- Optional async video preprocess for dull/dark detector assist (OFF by default; never mutates crops/embeddings)
- ByteTrack-style tracker + lost-pool rematch, stationary prior, occlusion freeze
- InsightFace `buffalo_l` ArcFace 512-d embeddings, cosine matching against a gallery (up to 5 embeddings/face)
- Auto-registration with unique ID; needs `min_embeds_to_register` good embeddings to avoid false new IDs
- Quality / blur gate (min face size, Laplacian blur) so poor frames never create identities
- Exactly-once entry/exit per visit via `PresenceManager` (HIDDEN grace, VISIT_RESUMED)
- Re-entry of a known person logs a new entry/exit but does **not** change the unique count
- Away-walker person path (KPI only; excluded from unique faces)
- Disturbance freeze (mass loss / brightness / stream gaps); visits table for dwell metrics
- End-of-stream flush closes every open visit; atomic image writes; local PostgreSQL required
- React dashboard + optional ARIA (Groq) read-only assistant

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
Web UI over the same PostgreSQL DB, image store and `events.log` (tracker Start/Stop + preprocess write to runtime only):
- **Overview** — KPIs (unique, entries/exits, inside), compact live, activity charts, count audit.
- **Live Monitor** — Webcam / local file / RTSP launcher; annotated feed (always **original**). When preprocess is on, a large **Original / Enhanced** strip appears under the feed.
- **Registered Faces** — searchable gallery; drawer with Face & Embedding (512-d heatmap, close-ups, nearest identities) + visit timeline.
- **Event Logs** — paginated entry/exit table with crops.
- **Analytics** — visitor flow / status charts.
- **Configuration** — theme, invariants, count audit, **Video preprocessing** card (hot-reload).
- **System Log** — `events.log` / `away_events.log` viewer.
- **ARIA** (optional) — top-bar pill or Ctrl/Cmd+K. Read-only Groq tools; live KPI snapshot each turn; whitelist tags/nav chips. **Privacy:** no images or 512-d embeddings sent to Groq (`GROQ_MODEL`, default in `.env.example`).

Data refreshes every ~3 s (pause in the top bar).

```bash
make install          # backend pip + frontend npm
make db               # local Postgres (deploy/docker-compose.yml)
cp .env.example .env  # set DATABASE_URL; optional GROQ_API_KEY
make build-web && make api   # API + built UI → http://127.0.0.1:8000
# Dev UI: make api  (terminal 1)  +  make web  (terminal 2) → http://127.0.0.1:5173
```
Without `GROQ_API_KEY` the API and dashboard still start; ARIA shows setup instructions.

No results yet? Preview with fake demo data (`data/demo/`, never touches real logs):
```bash
python3 backend/scripts/seed_demo.py
TRACKER_LOG_DIR=data/demo/logs make api
```
Stack: FastAPI (`backend/api`) → React + Vite (`frontend/`). Key routes: `/api/summary`, `/api/count`, `/api/timeline`,
`/api/visitors`, `/api/events`, `/api/logs`, `/api/preprocess*`, `/api/assistant/*`, `/api/tracker/*`, images under `/media`.

## Developing further (Cursor / AI tools)
Give `docs/KNOWLEDGE.md` to your AI assistant (rules file already at `.cursor/rules/project.mdc`). It lists what is built, invariants, known gaps and the prioritized next steps.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install                                         # or: make install-backend / make install-frontend
make db && cp .env.example .env                      # then edit DATABASE_URL if needed
make run                                             # python3 main.py
# Clean slate: make fresh
```
Defaults in committed `config.json`: `storage.backend=postgres`, `reset_on_start=false`,
`away_walkers.enabled=true`, `appearance.enabled=true`, `preprocess.enabled=false`.
Postgres DSN is **not** in `config.json` — only in gitignored `.env` (`DATABASE_URL`).
Without a working DB the API/CLI raise (no SQLite fallback). Crops + `events.log` stay under `logs/`.
Never commit `.env`, `.venv/`, `data/videos/`, or a real DSN.

### Interview (RTSP) — paste the link in the dashboard
```bash
make build-web && make api
# open http://127.0.0.1:8000 → Live Monitor → RTSP → paste rtsp://… → Start
# or: make api + make web → http://127.0.0.1:5173
```
CLI alternative: `make rtsp` then paste the URL, or  
`python3 backend/scripts/run_rtsp.py "rtsp://USER:PASS@CAMERA_IP:554/STREAM_PATH"`

- YOLO face weights auto-download to `data/models/yolov8n-face.pt` on first run (or place any YOLOv8 face model there).
- InsightFace `buffalo_l` models auto-download to `~/.insightface` on first run.
- On Windows, `insightface` may need *Microsoft C++ Build Tools*; Python 3.9–3.11 is the safest.

**Multiple videos, one combined unique count:**
```bash
python3 main.py --sources data/videos/video1.mp4 data/videos/video2.mp4 data/videos/video3.mp4
```
Videos run in order as a single session: one DB (reset once at the start), one face gallery, so a person seen in several videos
is counted once. Each video's open visits are closed at its end (exit events), and a person reappearing in the next video logs a
new entry under the same `face_id`. Annotated videos are written per source when enabled in `output`.
Use `--no-reset` to keep an existing DB, or list the files under `"video_sources"` in `config.json`.

**RTSP (interview):** `python3 main.py --source rtsp://user:pass@host:554/stream --display`

**Results / export:** `python3 backend/scripts/report.py --csv-dir samples`  
**Tests (no models needed):** `make test` or `pytest -q`

## config.json
Excerpt of the committed defaults (full file at repo root — prefer editing that file):
```json
{
  "video_source": "data/videos/video_sample1.mp4",
  "detection_skip_frames": 3,
  "detector":    { "weights": "data/models/yolov8n-face.pt", "conf_threshold": 0.28, "imgsz": 512, "device": "auto" },
  "recognition": { "model_pack": "buffalo_l", "match_threshold": 0.52, "min_embeds_to_register": 2,
                   "embed_retry_interval": 2, "max_embed_attempts": 12, "max_embeddings_per_face": 5 },
  "quality":     { "min_face_px": 12, "min_blur_var": 8.0 },
  "tracker":     { "high_conf": 0.32, "new_track_conf": 0.28, "iou_threshold": 0.28, "min_hits": 2, "max_lost_frames": 36 },
  "presence": {
    "exit_grace_seconds": 10, "edge_exit_grace_seconds": 2, "stationary_grace_seconds": 45,
    "occlusion_grace_multiplier": 2.0, "edge_margin_px": 24, "stationary_speed_px_s": 15,
    "resume_spatial_iou": 0.2, "reverify_interval_seconds": 5, "occlusion_iou": 0.3,
    "mass_loss_ratio": 0.6, "max_stream_gap_seconds": 20, "exit_zones": []
  },
  "away_walkers": { "enabled": true },
  "appearance":   { "enabled": true },
  "storage": {
    "backend": "postgres",
    "postgres": {},
    "log_dir": "logs", "events_log": "logs/events.log", "reset_on_start": false
  },
  "stream": {
    "reconnect_delay_sec": 3, "max_reconnects": 0, "read_timeout_sec": 10,
    "realtime": true, "catch_up_drop": true, "process_max_width": 960
  },
  "output": { "display": false, "live_preview": "data/runtime/live.jpg" },
  "preprocess": { "enabled": false, "process_at_detector_resolution": true, "mode": "async" }
}
```
`detection_skip_frames = N` means: run YOLO on one frame, skip the next N. `max_lost_frames` is auto-raised if too small for N.
Presence grace must satisfy `stationary_grace >= exit_grace >= edge_exit_grace` (seconds).

**Live overlay:** yellow/`…` = face tracked but not yet ID’d; green `ID N` = registered/recognized. Backs-of-heads and faces too small/blurry for InsightFace stay as `…` by design.

### Optional video preprocessing (OFF by default)
Dashboard **Configuration → Video preprocessing**. Improves dull/dark footage for the **face detector only** (optional denoise for grain; optional blur-gate assist on the 112×112 crop). Embeddings, entry/exit crops, live preview, away detection, and disturbance stats always use the **original** frame. Async worker downscales first; detector takes **enhanced if ready, else original** (no lag).

| Step | What | Why |
|---|---|---|
| Auto gate | Brightness + contrast only (dull / dark) | Skip work on good daylight; does **not** measure noise |
| Denoise | Light Gaussian / bilateral | Optional grain calm (not part of auto-gate) |
| White balance | Gray-world gains (skips IR/gray) | Colour-cast cameras |
| Gamma | LUT curve (`<1` brightens) | Under/over exposure |
| CLAHE | Luminance contrast (LAB) | Hazy / shadowed faces |
| Sharpen | Mild unsharp mask | Soft / compressed video |
| Quality assist | **Not a frame filter** — contrast on 112×112 blur-gate crop from the original | Fewer false “blurry” rejects |

Settings: `config.json` `"preprocess"` overridden by `data/runtime/preprocess.json` (hot-reload ~1s). API: `GET/PUT /api/preprocess`, `POST /api/preprocess/reset`, `GET /api/preprocess/preview`. Bench: `python3 backend/scripts/bench_preprocess.py`.

Measured on this machine (all frame steps, detector resolution ~640 long side): pipeline avg **1.7 ms** / p95 **2.0 ms**. Async mode adds ~0 ms to the main loop when the worker is ready (fallback to original if not). Full-res and interview-machine FPS not claimed here.

**File playback speed:** `stream.realtime=true` paces to the source FPS (never faster than 1×). `stream.catch_up_drop` defaults to `true` so when the CPU falls behind, file playback skips ahead and the live UI stays near realtime (set `false` for strict chronological, slower-than-realtime preview).

**Database:** local Postgres only (`storage.backend=postgres`). Start with `docker compose -f deploy/docker-compose.yml up -d db`, then set `DATABASE_URL=postgresql://facetrack:facetrack@localhost:5432/facetrack` in `.env` (never commit the DSN). If the DSN is missing or unreachable, startup **raises** (no SQLite fallback). Face crops and `events.log` remain on the local filesystem under `logs/`. Overview / `/api/health` shows `backend=postgres`.

## Output layout
```
logs/events.log                         # main session log (see event table below)
logs/away_events.log                    # AWAY_* only (when logging.separate_away_log=true)
logs/archive/events_<timestamp>.log     # previous events.log when reset_on_start archives
logs/entries/YYYY-MM-DD/entry_face0001_HHMMSS_ffffff.jpg
logs/exits/YYYY-MM-DD/exit_face0001_HHMMSS_ffffff.jpg
logs/faces/YYYY-MM-DD/face_face0001_HHMMSS_ffffff.jpg   # tight close-up saved with every stored embedding
PostgreSQL (facetrack)                  # tables: faces, embeddings, events, visits (+ away_*)
```
### Count definitions
- **Unique visitors** = `COUNT(*)` rows in `faces` = distinct `face_id` on `FACE_REGISTERED` lines for one session. Re-recognition (`FACE_RECOGNIZED`) and re-entries never change it. Away walkers are **not** part of it.
- **Entries / exits** = rows in `events` by `event_type` = `ENTRY` / `EXIT` log lines.
- **Returning visitors** = faces with `visit_count > 1`.
- **Open visits** = entries without matching exits (DB: `visits.exit_ts IS NULL`).

### How to get the count
1. **SQL (source of truth in the DB):**
   ```sql
   SELECT COUNT(*) FROM faces;
   ```
2. **CLI (DB ↔ log reconcile):**
   ```bash
   python3 backend/scripts/count.py --source both
   python3 backend/scripts/verify_log.py
   ```
   Exit `0` = consistent, `1` = mismatch, `2` = could not compute.
3. **Final hygiene check** (count + verify + date/session/AWAY/archive/secret scans):
   ```bash
   make check
   # or: python3 backend/scripts/final_check.py
   ```
   Prints `PASS`/`FAIL` per check; exits `1` on any failure. Run after a clean `--fresh` session.
4. **API:** `GET /api/count?session=latest` — DB when present, log fallback, plus `consistent` / `diffs`.
5. **macOS-safe shell (BSD grep has no `-P`) — one session only:**
   ```bash
   awk '/FACE_REGISTERED/ { for (i=1;i<=NF;i++) if ($i ~ /^face_id=/) print $i }' logs/events.log | sort -u | wc -l
   ```
   This is only correct for a **single session’s** log (that is why `events.log` is archived on `reset_on_start` / `--fresh`).

### Log line format
`YYYY-MM-DD HH:MM:SS,mmm | LEVEL   | EVENT                sid=<session_id> key=value ...`

- Values contain no spaces. Paths are relative with forward slashes (never truncated in the file).
- Every line includes `sid=<session_id>` as the first key=value (session id like `20261107T125600_ab12`).
- Timestamps stored in the DB / shown as `ts=` are ISO-8601 with milliseconds (wall clock via `src/clock.py`). Monotonic time is used only for durations/deadlines.

| Event | Typical fields |
|---|---|
| `SYSTEM_START` | `session`, `sources`, `detection_skip_frames`, `match_threshold`, `reset_on_start`, `features`, `log_level` |
| `SYSTEM_STOP` | `session`, `unique_visitors`, `entries`, `exits`, `open_visits` (+ `sid` on every line) |
| `SOURCE_START` | `source`, `live`, `realtime`, `video_fps` |
| `STREAM_END` | `source`, `frames`, `skipped`, `reason=eof\|stopped\|error\|disconnect`, `new_faces`, `unique_so_far` |
| `COUNT_SNAPSHOT` | `unique_total`, `entries`, `exits` (every STATS interval + at STREAM_END / SYSTEM_STOP) |
| `SESSION_RECOVERED` | `visit_id`, `face_id`, `last_ts`, `reason=crash_recovery` |
| `LOG_ARCHIVED` | `path` |
| `TRACK_STARTED` / `TRACK_ENDED` | `track`, (+ `face_id`, `reason` on end). One-to-one; tentative drops → DEBUG `TRACK_DROPPED` |
| `EMBEDDING_GENERATED` | `track`, `det_score`, `attempt` |
| `FACE_REGISTERED` | `face_id`, `unique_total`, `track` (+ `sid`) — new unique identity |
| `FACE_RECOGNIZED` | `face_id`, `track`, `similarity` (does **not** change unique count) |
| `ENTRY` | `ts`, `visit_id`, `face_id`, `track`, `image`, `entry_kind=new\|re_entry` (`ts` = DB `events.timestamp`) |
| `NEW_ENTRY` | Same fields as ENTRY — first visit for this face (audit companion to `ENTRY`) |
| `RE_ENTRY` | Same fields as ENTRY — returning visit after a prior EXIT (audit companion) |
| `EXIT` | `ts`, `visit_id`, `face_id`, `track`, `image`, `duration_sec`, `visible_sec`, `occluded_sec`, `reason`, `last_seen` |
| `VISIT_HIDDEN` | `visit_id`, `face_id`, `track`, `grace_sec`, `deadline_in_sec`, `deadline_ts` (ISO), `reason` |
| `VISIT_RESUMED` | `face_id`, `track`, `gap_sec`, … |
| `TRACK_HANDOVER` / `IDENTITY_MISMATCH` | identity continuity / re-verify |
| `DISTURBANCE_START` / `DISTURBANCE_END` | `ts` (ISO); END also `duration_sec` |
| `STATS` | `frames`, `fps`, `unique`, `away_summary=away_tracks=…` (away detail lives in `away_events.log`) |
| `TIMESTAMP_ANOMALY` | WARNING when a computed stamp skews >60s from wall; clamped to `wall_now()` |

Away events (`AWAY_TRACK_STARTED/ENDED`, `AWAY_ENTRY/EXIT`, `AWAY_RESUMED`, …) go to `logs/away_events.log`. DEBUG-only (`FACE_ROI_REFINE`, `TRACK_LOST`, `EMBED_NO_FACE`, …) require `logging.level=DEBUG` or `debug_events=true`.

### Sessions
- `SYSTEM_START` opens a session; `SYSTEM_STOP` always runs in a `finally` (SIGINT/SIGTERM/exception) exactly once.
- Dashboard **Start** keeps the face gallery and appends to `events.log` by default. Tick **Clear registered faces** only when you want a wipe (archives logs + resets DB for that job).
- CLI: `storage.reset_on_start` (default false) or `python3 main.py --fresh` archives `events.log` / `away_events.log` and resets the **active** visitor DB. `--fresh` never touches other schemas or unrelated user files.
- When not resetting, the log appends and any open visits from a crashed prior run get **one EXIT** stamped at `faces.last_seen` with `reason=crash_recovery` (`SESSION_RECOVERED`). Recovery does **not** invent a matching ENTRY.

## Assumptions
- A "visit" starts when a track gets an identity (ENTRY) and ends only after a **time-based** grace from last detection (or stream end). Brief occlusion / turn-away / flicker resumes the same visit (`VISIT_RESUMED`).
- A person who truly leaves and returns later gets a new entry/exit pair under the same `face_id` (unique count unchanged).
- Faces below `min_face_px` or too blurry are ignored for identity (they are never counted).
- One camera; file and live sources both stamp events with **wall-clock processing time** (`src/clock.wall_now`); `time.monotonic` is only for grace durations.
- `reset_on_start: false` is the committed default (keep faces/logs across restarts). Use the dashboard clear checkbox or `--fresh` to wipe.
- `match_threshold` (committed default **0.52**) is a starting point for `buffalo_l`; tune on your footage (higher = more unique IDs, lower = more merges).
- Quality check: `python3 backend/scripts/presence_report.py` (`DATABASE_URL`; entries should equal exits; short re-entries ≈ 0).

## Sample output
After a clean session (`python3 main.py --fresh` or dashboard **Clear registered faces on Start**), expect:
- `logs/events.log` with `SYSTEM_START` … `FACE_REGISTERED` / `ENTRY` / `EXIT` … `SYSTEM_STOP` / `COUNT_SNAPSHOT`
- Crops under `logs/faces/`, `logs/entries/`, `logs/exits/` (dated folders)
- Postgres rows in `faces`, `embeddings`, `events`, `visits`
- Export helpers: `python3 backend/scripts/report.py --csv-dir samples` · reconcile with `make count`

Paste concrete counts, a few crop paths, and a short DB export here after your submission run: `<ADD LINK / NUMBERS>`

---
This project is a part of a hackathon run by https://katomaran.com

