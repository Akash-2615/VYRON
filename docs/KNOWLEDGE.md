# Project Knowledge File — Intelligent Face Tracker (Katomaran Hackathon)

> Purpose: give this file to Cursor (or any AI coding tool) as context. It states what exists, why it is built the way it is,
> what is verified vs. unverified, and exactly what to build next. Read sections 1-6 before changing code.

## 1. Goal and hard deadline
Hackathon task by Katomaran: an AI visitor counter that processes a video (dev) or a **live RTSP stream (interview)**, detects/tracks/recognizes
faces, auto-registers new faces, logs **exactly one entry and one exit per visit** (crop image + timestamp + type + face id) to a log file,
local image folders and a database, and reports the **unique visitor count** (re-identification must NOT increase it).
- Submission: GitHub repo via Google Form **before 6 PM, Sunday 8 Nov 2026**.
- README must contain: setup, assumptions, sample `config.json`, AI planning doc, architecture diagram, **Loom/YouTube video link
  (no video = not reviewed)**, sample output (logs, images, DB entries), and the exact last line:
  `This project is a part of a hackathon run by https://katomaran.com`.
- Mandatory stack: YOLO (v5/v8) detection, InsightFace/ArcFace embeddings (NOT the `face_recognition` library), OpenCV/DeepSort/ByteTrack tracking,
  Python backend, any DB (SQLite chosen), `config.json` with frame-skip parameter, log file + image store + DB.
- Prompts used with AI tools are evaluated in the interview -> keep `docs/PROMPTS.md` updated and understand every module.

## 2. Status: what is built
All three core modules are implemented. **Everything below was tested only with fake detector/embedder on synthetic video
(`pytest`: 6 passing). It has NOT yet been run with real YOLO + InsightFace, real videos, a GPU, or a live RTSP stream, and the React
dashboard has not been viewed in a browser** (type-check and production build pass; API endpoints verified with curl on demo data).

| Area | State |
|---|---|
| Detection (YOLOv8-face), embeddings (InsightFace `buffalo_l`), tracking (own ByteTrack-style), gallery matching, auto-registration | Built, untested with real models |
| Entry/exit state machine (exactly-once, handover, end-of-stream flush) | Built, unit-tested |
| Logging: `events.log` + crops in `logs/{entries,exits,faces}/YYYY-MM-DD/` + SQLite | Built, tested |
| Unique count (`SELECT COUNT(*) FROM faces`) | Built, tested |
| Multi-video session with one combined unique count (`--sources a.mp4 b.mp4`) | Built, unit-tested |
| RTSP (TCP, threaded latest-frame reader, auto-reconnect) | Built, never run against a real camera |
| FastAPI read-only API + React/TS dashboard (Overview, Visitors w/ Face & Embedding tab, Event Log, System Log) | Built; demo data via `scripts/seed_demo.py` |
| README with architecture diagram, planning doc, compute estimates | Written; has placeholders (video link, sample output) |

## 3. Repository map
```
config.json                 runtime config (see section 5)
main.py                     CLI entry: --config, --source, --sources (multi-video session), --no-reset, --display, --max-frames
src/config.py               DEFAULTS + deep-merge + validation (auto-raises tracker.max_lost_frames to >= 2*(skip+1)+2)
src/stream.py               FrameSource: file = sequential cv2 read; live (rtsp/http/webcam idx) = reader thread, latest frame, reconnect
src/detector.py             YoloFaceDetector.detect(frame) -> (N,5) [x1,y1,x2,y2,conf]; auto-downloads weights
src/recognizer.py           FaceEmbedder.embed(crop) -> (L2-normalised 512-d, det_score) | None  (InsightFace re-detects/aligns inside the crop)
src/gallery.py              In-memory cosine-matching gallery (<= max_embeddings_per_face per identity)
src/tracker.py              ByteTrackLite: IoU + velocity prediction, 2-stage (high/low conf) association, coasting through skipped frames
src/pipeline.py             FaceTrackingPipeline: detect -> track -> identify -> entry/exit events. THE core file.
src/database.py             SQLite (WAL, commit-per-write, auto-migration of image_path columns)
src/event_logger.py         events.log writer (flush per line, rotating) + atomic crop saving
src/visualize.py            Overlay for annotated video / live window
scripts/report.py           Print/export DB (CSV) ; scripts/seed_demo.py  fake demo data into demo/ (never touches real data)
api/server.py               FastAPI read-only API (create_app factory) + serves web/dist and /media
web/                        Vite + React 18 + TypeScript + Recharts + lucide-react dashboard
tests/test_pipeline.py      synthetic end-to-end logic tests (fake detector/embedder: coloured squares)
tests/test_api.py           API tests (skipped if fastapi/httpx missing)
docs/PROMPTS.md             prompt log template (fill this in!)
```

## 4. How the pipeline works (read before editing `src/pipeline.py`)
1. `process_frame(frame, idx)`: run YOLO only when `idx % (detection_skip_frames+1) == 0`; otherwise tracks coast on velocity.
2. `ByteTrackLite.update` returns active + removed tracks. A track is *confirmed* after `min_hits` detection matches.
   Removed when unmatched for `max_lost_frames`, when a tentative track misses one cycle, or when its predicted centre leaves the frame.
3. Identification (only for confirmed tracks, only on frames where the track was freshly matched):
   quality gate (`min_face_px`, Laplacian blur `min_blur_var`) -> crop (25% margin) -> `embedder.embed` -> `gallery.match`.
   - sim >= `match_threshold` -> recognized existing `face_id`.
   - else collect embeddings; after `min_embeds_to_register` unmatched embeddings, average them, re-check the gallery, then **register a new face**.
   - Each stored embedding also saves a tight close-up crop (`logs/faces/...`) and stores its path in `embeddings.image_path`;
     the first one becomes `faces.image_path` (registered close-up).
4. Entry/exit ownership: `present[face_id] = owning track id`.
   - Identity resolved and nobody owns it -> track owns it -> **ENTRY** event.
   - Owner exists but is stale (`time_since_update > 0`) -> **handover** (no entry/exit; handles track fragmentation).
   - Owner is live -> duplicate: new track does not own it, logs `DUPLICATE_TRACK`.
   - Owning track removed -> **EXIT** event. `finalize()` closes all open visits at stream end / between videos (`note=stream_end`).
5. Each event writes, in this order: crop image (atomic tmp+rename) -> DB row (one transaction, also updates `faces.last_seen/visit_count`) -> `events.log` line.

### Invariants — do not break these
- Exactly one `entry` and one `exit` per visit; unique count = rows in `faces`; returning person never creates a new `faces` row.
- Identity is never created from a poor-quality frame or from a single embedding (`min_embeds_to_register`).
- `reset_on_start` happens once per session (never between videos in a multi-video run).
- `events.log` must contain: TRACK_STARTED/ENDED, EMBEDDING_GENERATED, FACE_REGISTERED, FACE_RECOGNIZED, ENTRY, EXIT (graders verify with it).
- Heavy imports (ultralytics, insightface) stay inside `main.py`/detector/recognizer so tests run without models (dependency injection into the pipeline).
- DB writes stay commit-per-write (resilience to interruption).

## 5. Config (`config.json`) — keys and meaning
`video_source` / `video_sources[]`; `detection_skip_frames` (N = skip N frames between YOLO runs);
`detector.{weights,weights_url,conf_threshold,imgsz,device}`; `recognition.{model_pack,det_size,match_threshold(0.45),max_embeddings_per_face,
min_embeds_to_register,embed_retry_interval,max_embed_attempts}`; `quality.{min_face_px,min_blur_var}`;
`tracker.{high_conf,new_track_conf,iou_threshold,min_hits,max_lost_frames}`; `storage.{db_path,log_dir,events_log,reset_on_start}`;
`stream.{reconnect_delay_sec,max_reconnects,read_timeout_sec}`; `output.{display,annotated_video}`.
Tuning guide: more false "new people" -> lower `match_threshold` or relax quality gate; different people merged -> raise `match_threshold`;
false exits during occlusion -> raise `max_lost_frames`; slow -> raise `detection_skip_frames`, lower `imgsz`.

## 6. Database schema (SQLite, `data/visitors.db`)
- `faces(face_id PK, first_seen, last_seen, visit_count, image_path)`
- `embeddings(id PK, face_id FK, vector BLOB float32[512], created_at, image_path)`
- `events(id PK, face_id FK, event_type 'entry'|'exit', timestamp ISO local time, image_path, track_id, frame_idx, note)`

## 7. API (`api/server.py`, run from repo root: `uvicorn api.server:app --port 8000`)
`GET /api/health, /api/summary, /api/timeline, /api/visitors, /api/visitors/{id}, /api/visitors/{id}/embeddings, /api/events?type=&face_id=&limit=&offset=, /api/logs?limit=&event=&level=&q=`;
images under `/media/...` (from `logs/`); serves `web/dist` at `/` if built. Read-only (`PRAGMA query_only`). Image paths in the DB are
relative to the working directory, so the API must start from the repo root. Env overrides: `TRACKER_CONFIG`, `TRACKER_DB`, `TRACKER_LOG_DIR`, `TRACKER_EVENTS_LOG`.

## 8. Frontend (`web/`)
Vite + React 18 + TS + Recharts. `src/api.ts` (types + fetchers), `src/usePolling.ts` (3 s polling, pause), pages `Overview/Visitors/Events/Logs`,
components `Layout, KpiCard, ActivityChart, VisitorDrawer (tabs), EmbeddingPanel (heatmap/histogram/similar identities), ui`. Plain CSS in `styles.css`.
Dev: `uvicorn ...` + `cd web && npm run dev` (proxy to :8000). Build: `npm run build`. Keep it simple: no state libraries, no CSS frameworks.

## 9. Run / test commands
```bash
pip install -r requirements.txt && pip install -r requirements-api.txt
python main.py --config config.json                                  # one video (video_source)
python main.py --sources data/v1.mp4 data/v2.mp4                     # one combined unique count
python main.py --source rtsp://user:pass@host:554/stream --display   # live
python scripts/report.py --db data/visitors.db --csv-dir samples
pytest -q
python scripts/seed_demo.py   # fake demo data -> demo/
TRACKER_DB=demo/visitors.db TRACKER_LOG_DIR=demo/logs uvicorn api.server:app --port 8000
cd web && npm install && npm run build
```

## 10. Known limitations / honest gaps
1. Real-model behaviour unverified; thresholds (0.45 match, quality gates, `max_lost_frames`) are starting guesses.
2. ENTRY is logged when identity is confirmed (~2 good embeddings), so it lags first appearance by ~1 s; EXIT is stamped when the track is dropped
   (after `max_lost_frames`), not at last-seen time. Timestamps are wall-clock processing time, not video time.
3. Faces that never pass the quality gate / never embed are silently not counted (by design, but small background faces are missed).
4. `frame_idx` restarts per video in multi-video sessions; `Track._next_id` is process-global (fine, just not reset).
5. Tracker is IoU+velocity only (no appearance model); fast motion/heavy occlusion can fragment tracks (handover logic mitigates).
6. API recomputes stats from full tables on each request (fine for hackathon scale); no auth, CORS `*`.
7. Dashboard not visually verified in a browser; bundle ~620 kB (no code-splitting).
8. `docs/PROMPTS.md`, README video link and README "Sample output" section are placeholders to fill.

## 11. NEXT STEPS (priority order, each with acceptance criteria)

### P0 — before the Nov 8 6 PM deadline (do these first)
1. **Run on the real sample video(s)** and inspect `output/annotated.mp4`. Acceptance: no obvious duplicate IDs for one person, no merged people;
   record final unique count; `entries == exits`. Tune `config.json` only (do not change logic first).
2. **Exit timestamp = last-seen time.** In `Track`, store `last_seen_ts` (set whenever `matched_now`); in `_on_track_removed` use it for the
   EXIT event timestamp/image (pass an explicit `ts` into `_emit`). Add a test: exit timestamp must be <= removal time and >= last detection time.
3. **Verify RTSP**: test with a phone/IP-cam or `ffmpeg`/MediaMTX loop of the sample video. Acceptance: reconnect after killing the stream; no
   growing latency (reader keeps only the latest frame); log lines `Stream read failed; reconnecting`.
4. **Generate submission artifacts**: `python scripts/report.py --csv-dir samples`; commit `logs/` (events.log + some crops) and `samples/`; fill README "Sample output".
5. **Record the Loom/YouTube demo** (show: architecture, config skip param, run, annotated video, events.log, DB query, dashboard) and paste the link in README.
6. **Fill `docs/PROMPTS.md`** with the real prompts used and what you verified; be ready to explain `pipeline.py`, `tracker.py`, thresholds.
7. Confirm README last line is exactly the katomaran sentence and the form submission is made.

### P1 — quality improvements (after P0)
- Use **video time** for file sources (`frame_idx / fps`) as an optional timestamp mode (`config.timestamps: "wall"|"video"`), keeping wall-clock for RTSP.
- **ID-switch guard:** re-verify an identified track every K detection cycles; if embedding strongly disagrees (< threshold) for several checks, log
  `IDENTITY_MISMATCH` and re-identify (end old visit, start new). Add a synthetic test where two squares cross.
- **Appearance-aided tracking**: add embedding similarity into association, or swap `ByteTrackLite` for ultralytics/`supervision` ByteTrack while keeping the
  `update(dets, frame_size) -> (active, removed)` interface and Track fields used by `pipeline.py`.
- Better embedding quality: pose/aligned-crop filtering using InsightFace landmarks; use best-of-N frame for the registered close-up.
- Config schema validation (fail fast with clear messages) and a `--dry-run` that checks weights/RTSP connectivity.

### P2 — dashboard / platform polish (keep it simple, no over-engineering)
- Live preview: MJPEG endpoint streaming the annotated frame (new route in `api/` + `<img>` on Overview).
- Date/time range filter and CSV export buttons (`/api/export/events.csv`); dwell-time histogram on Overview.
- Visitor drawer: show entry/exit crops side by side; "merge suspicious identities" view driven by the similar-identities list.
- Code-split the bundle (`React.lazy` per page); basic auth token for the API; Dockerfile + docker-compose (tracker + api).
- GitHub Actions: run `pytest` and `npm run build`.

### P3 — ideas
Multi-camera support (camera_id column), PostgreSQL option behind `src/database.py`, alerting webhooks, privacy options (hash embeddings, retention policy).

## 12. Working agreements for AI assistants editing this repo
- Keep modules small and commented; no new heavy dependencies without a reason. Python 3.9-3.11, type hints where cheap.
- Any change to `pipeline.py` or `tracker.py` must keep `pytest` green and add/adjust a synthetic test (fake detector/embedder pattern in `tests/test_pipeline.py`).
- Do not change DB schema without adding a migration in `Database._migrate`.
- Never write demo/fake data into the real `data/` or `logs/` folders.
- Do not mark anything "verified" unless it was actually run; state what was and was not tested.
- Windows-friendly paths (use `pathlib`), no shell-specific assumptions in code.
