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
  Python backend, PostgreSQL (default; SQLite for tests), `config.json` with frame-skip parameter, log file + image store + DB.
- Prompts used with AI tools are evaluated in the interview -> keep `docs/PROMPTS.md` updated and understand every module.

## 2. Status: what is built
All three core modules are implemented. **Everything below was tested only with fake detector/embedder on synthetic video
(`pytest`: 6 passing). It has NOT yet been run with real YOLO + InsightFace, real videos, a GPU, or a live RTSP stream, and the React
dashboard has not been viewed in a browser** (type-check and production build pass; API endpoints verified with curl on demo data).

| Area | State |
|---|---|
| Detection (YOLOv8-face + RetinaFace/SCRFD fallback), embeddings (InsightFace `buffalo_l`), tracking (own ByteTrack-style), gallery matching, auto-registration | Built |
| Entry/exit state machine (exactly-once, handover, end-of-stream flush) | Built, unit-tested |
| Logging: `events.log` + crops in `logs/{entries,exits,faces}/YYYY-MM-DD/` + PostgreSQL | Built, tested |
| Unique count (`SELECT COUNT(*) FROM faces`) | Built, tested |
| Multi-video session with one combined unique count (`--sources a.mp4 b.mp4`) | Built, unit-tested |
| RTSP (TCP, threaded latest-frame reader, auto-reconnect) | Built; live path unit-tested; interview runner `run_rtsp.py` (paste URL only) |
| FastAPI read-only API + React/TS dashboard (Overview, Visitors w/ Face & Embedding tab, Event Log, System Log) | Built; demo data via `scripts/seed_demo.py` |
| README with architecture diagram, planning doc, compute estimates | Written; has placeholders (video link, sample output) |

## 3. Repository map
```
config.json                 runtime config (see section 5)
main.py                     CLI entry: --config, --source, --sources (multi-video session), --no-reset, --display, --max-frames
run_rtsp.py                 Interview helper: paste RTSP URL → auto models + DB + live window + dashboard
src/config.py               DEFAULTS + deep-merge + validation (auto-raises tracker.max_lost_frames to >= 2*(skip+1)+2)
src/stream.py               FrameSource: file = sequential cv2 read; live (rtsp/http/webcam idx) = reader thread, latest frame, reconnect
src/detector.py             YoloFaceDetector + HybridFaceDetector (Retina/SCRFD fallback); detect -> (N,5)
src/retina_face.py          RetinaFace-family secondary detector (SCRFD det_10g)
src/recognizer.py           FaceEmbedder.embed(crop) -> (L2-normalised 512-d, det_score) | None  (InsightFace re-detects/aligns inside the crop)
src/gallery.py              In-memory cosine-matching gallery (<= max_embeddings_per_face per identity)
src/tracker.py              ByteTrackLite: IoU + velocity prediction, 2-stage (high/low conf) association, coasting through skipped frames
src/pipeline.py             FaceTrackingPipeline: detect -> track -> identify -> entry/exit events. THE core file.
src/database.py             PostgreSQL (default) or SQLite; commit-per-write; auto-migration
src/clock.py                wall_now / mono / wall_from_mono / iso_ms (single time source)
src/session.py              session_id, archive-on-reset, crash recovery, SYSTEM_START/STOP
src/counting.py             count_from_db / count_from_log / reconcile / verify_log_invariants
src/event_logger.py         events.log + away_events.log (sid on every line, TRACK pairing) + crops
scripts/count.py            CLI reconcile DB ↔ log (exit 0/1/2)
scripts/verify_log.py       session invariant checker
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
4. Entry/exit ownership + **presence visit state machine** (`src/presence.py`):
   - Identity resolved and nobody owns it -> track owns it -> **ENTRY** + open `visits` row.
   - Owner exists but is stale -> **handover** (no entry/exit).
   - Owner is live -> `DUPLICATE_TRACK`.
   - Owning track lost -> visit moves to **HIDDEN** (not EXIT). Grace is **seconds**, chosen by
     edge exit / mid-frame stationary / default (+ occlusion multiplier). `EXIT` uses `last_seen_ts`.
   - Same face before deadline -> **VISIT_RESUMED** (gap added to `occluded_seconds`).
   - Deadline / stream_end / stream_gap -> one **EXIT**. Lost tracks sit in a rematch pool.
   - Optional BODY_ONLY keep-alive when person-detector still overlaps the last box.
   - Mass loss / brightness jump / stream gaps freeze hidden deadlines (`DISTURBANCE_*`).
5. Each event writes, in this order: crop image (atomic tmp+rename) -> DB row (one transaction, also updates `faces.last_seen/visit_count`) -> `events.log` line.
   Log events include: `VISIT_HIDDEN`, `VISIT_RESUMED`, `IDENTITY_MISMATCH`, `DISTURBANCE_START/END`, `TRACK_LOST` (DEBUG).
   `EXIT_PENDING` was merged into `VISIT_HIDDEN` (do not reintroduce a duplicate pending line).
   Time: `src/clock.py` — `wall_now()` for all stored/logged stamps; `mono()` for durations only;
   convert with `wall_from_mono(instant)`. Never add monotonic values to wall time.

### Count definitions
- **Unique visitors** = rows in `faces` = distinct `face_id` on `FACE_REGISTERED` (one session). `FACE_RECOGNIZED` / re-entries never change it. Away walkers are not included.
- **Entries / exits** = `events` rows by type = `ENTRY`/`EXIT` log lines. **Returning** = `visit_count > 1`. **Open visits** = entries without exits.

### Invariants — do not break these
- Exactly one `entry` and one `exit` per visit; unique count = rows in `faces`; returning person never creates a new `faces` row.
- **DB count and log-derived count must agree** (enforced by `python scripts/count.py` / `tests/test_counting.py` in CI-style checks). `src/counting.py` is the shared implementation; `GET /api/count` exposes it.
- Identity is never created from a poor-quality frame or from a single embedding (`min_embeds_to_register`).
- `reset_on_start` happens once per session (never between videos in a multi-video run).
- **Single clock source:** all `events.timestamp` / `last_seen` / image folder dates / ENTRY·EXIT `ts=` come from wall clock (ISO-ms). Monotonic is durations-only.
- **Every log line has `sid=<session_id>`** (first key=value). When `reset_on_start` / `--fresh`, archive prior `events.log` (+ `away_events.log`) so the log and DB describe the same session; exactly one `SYSTEM_STOP` per start.
- **Crash recovery** (`reset_on_start=false` / `--no-reset`): each leftover open visit gets **one EXIT** at `faces.last_seen` with `reason=crash_recovery` (+ `SESSION_RECOVERED`). Never invent a synthetic ENTRY.
- `events.log` must contain: TRACK_STARTED/ENDED (paired), EMBEDDING_GENERATED, FACE_REGISTERED (`face_id`,`unique_total`,`track`), FACE_RECOGNIZED, ENTRY, EXIT, COUNT_SNAPSHOT; STREAM_END carries `new_faces`/`unique_so_far`. AWAY_* live in `away_events.log`.
- Heavy imports (ultralytics, insightface) stay inside `main.py`/detector/recognizer so tests run without models (dependency injection into the pipeline).
- DB writes stay commit-per-write (resilience to interruption).

## 5. Config (`config.json`) — keys and meaning
`video_source` / `video_sources[]`; `detection_skip_frames` (N = skip N frames between YOLO runs);
`detector.{weights,weights_url,conf_threshold,imgsz,device}`; `recognition.{model_pack,det_size,match_threshold(0.45),max_embeddings_per_face,
min_embeds_to_register,embed_retry_interval,max_embed_attempts}`; `quality.{min_face_px,min_blur_var}`;
`tracker.{high_conf,new_track_conf,iou_threshold,min_hits,max_lost_frames}`; `storage.{backend,postgres,log_dir,events_log,reset_on_start}`
(committed defaults: `backend=postgres`, `reset_on_start=false`, `postgres={}` — **no DSN in config**; DSN only via `DATABASE_URL` / `TRACKER_DSN`; missing/unreachable DB → hard error, no SQLite fallback; face crops + `events.log` stay under `logs/`; dashboard Start keeps faces/logs unless the clear checkbox is on; use `--fresh` for a clean CLI wipe);
`stream.{reconnect_delay_sec,max_reconnects,read_timeout_sec,realtime,catch_up_drop,stream_gap_seconds}`
(`catch_up_drop` default **true**: file playback skips ahead when CPU is behind so the live UI stays near realtime);
`output.{display,annotated_video}`;
`logging.{level,separate_away_log,debug_events}` (AWAY_* → `logs/away_events.log` when separate);
`away_walkers.enabled` default **true** (person YOLO → no-face KPI, excluded from unique faces);
`appearance.enabled` default **false** in committed config.
Live overlay: yellow/`…` = tracked face pending ID; green `ID N` = registered/recognized.
Tuning guide: more false "new people" -> lower `match_threshold` or relax quality gate; different people merged -> raise `match_threshold`;
false exits during occlusion -> raise `max_lost_frames`; slow -> raise `detection_skip_frames`, lower `imgsz`
(or keep `catch_up_drop=false` and accept slower-than-realtime preview).

## 6. Database schema (local Postgres required; SQLite for unit tests only)
Runtime requires **local Postgres**: `docker compose up -d db`, then set `DATABASE_URL=postgresql://facetrack:facetrack@localhost:5432/facetrack` in `.env` only (never commit the DSN). `src/db_bootstrap.open_database` opens Postgres and raises `DatabaseError` if the DSN is missing or connect fails — **no SQLite fallback**. Face crops and `events.log` remain on the local filesystem under `logs/`.
- `faces(face_id PK, first_seen, last_seen, visit_count, image_path)`
- `embeddings(id PK, face_id FK, vector BYTEA/BLOB float32[512], created_at, image_path)`
- `events(..., visit_id nullable)` — still exactly one entry and one exit per visit
- `visits(visit_id, face_id, entry_ts, exit_ts, visible_seconds, occluded_seconds, resume_count, exit_reason)`
  - `time_in_frame = visible + occluded` per visit; person total sums visits
- Open via `src.db_bootstrap.open_database`; tests keep `Database(path, reset=True)` (SQLite).

## 7. API (`api/server.py`, run from repo root: `uvicorn api.server:app --port 8000`)
`GET /api/health, /api/summary, /api/count?session=latest, /api/timeline, /api/visitors, /api/visitors/{id}, /api/visitors/{id}/embeddings, /api/events?type=&face_id=&limit=&offset=, /api/logs?limit=&event=&level=&q=&source=events|away, /api/tracker/status, /api/assistant/status`;
`POST /api/tracker/start` `{url, reset?}`, `POST /api/tracker/stop` — dashboard RTSP job (background YOLO+InsightFace pipeline);
`POST /api/assistant/chat` — optional Groq SSE assistant (tools via `api/services.py` + `api/assistant_tools.py`); disabled/503 without `GROQ_API_KEY`.
images under `/media/...` (from `logs/`); serves `web/dist` at `/` if built. Read-only connections. Image paths in the DB are
relative to the working directory, so the API must start from the repo root. Env overrides: `TRACKER_CONFIG`, `DATABASE_URL` / `TRACKER_DSN` (required Postgres), `TRACKER_LOG_DIR`, `TRACKER_EVENTS_LOG`, `GROQ_API_KEY`, `GROQ_MODEL`, `ASSISTANT_ENABLED`.
DB open path: `src/db_bootstrap.open_database` — Postgres only; missing/unreachable → `DatabaseError` + `/api/health` `ok=false` (secrets redacted). Logs `DB_BACKEND backend=postgres` (never `DB_FALLBACK`).
Shared analytics: `api/services.AnalyticsService` (used by assistant tools; HTTP routes still have local helpers).
API SQL uses `?` placeholders; `Database.readonly()` adapts to `%s` for Postgres. `PRAGMA` / `lastrowid` stay inside `src/database.py` (sqlite test path only). Timestamps are TEXT; `api` `_secs`/`secs` accept ISO `T` or space-separated and `datetime` objects.

## 8. Frontend (`web/`)
Vite + React 18 + TS + Recharts. `src/api.ts` (types + fetchers), `src/usePolling.ts` (3 s polling, pause), pages `Overview/Visitors/Events/Logs`,
components `Layout, KpiCard, ActivityChart, VisitorDrawer (tabs), EmbeddingPanel, assistant/* (docked Groq chat), ui`. Plain CSS in `styles.css`.
Dev: `uvicorn ...` + `cd web && npm run dev` (proxy to :8000). Build: `npm run build`. Keep it simple: no state libraries, no CSS frameworks.

## 9. Run / test commands
```bash
pip install -r requirements.txt && pip install -r requirements-api.txt
python main.py --config config.json                                  # one video (video_source)
python main.py --fresh --config config.json                          # archive logs + reset active DB, then start
python main.py --sources data/v1.mp4 data/v2.mp4                     # one combined unique count
python main.py --source rtsp://user:pass@host:554/stream --display   # live
python scripts/count.py --source both --session latest
python scripts/verify_log.py
python scripts/final_check.py                                        # PASS/FAIL hygiene; exit 1 on FAIL
python scripts/report.py --db data/visitors.db --csv-dir samples
pytest -q
python scripts/seed_demo.py   # fake demo data -> demo/
DATABASE_URL=postgresql://… TRACKER_LOG_DIR=demo/logs uvicorn api.server:app --port 8000
cd web && npm install && npm run build
```

## 10. Known limitations / honest gaps
1. Real-model behaviour unverified; thresholds (0.45 match, quality gates, `max_lost_frames`) are starting guesses.
2. ENTRY is logged when identity is confirmed (~2 good embeddings), so it lags first appearance by ~1 s; EXIT uses last real detection (`last_seen`)
   after the presence grace. Timestamps are wall-clock processing time (not video PTS); mono is durations-only.
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
2. **Presence / visit state machine (done in code; verify on real video).** Time-based grace, last_seen EXIT,
   visits table, disturbance freeze, lost-pool rematch. Acceptance on real clips: entries≈exits, short re-entries ≈0
   (`python scripts/presence_report.py --db data/visitors.db`).
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
Multi-camera support (camera_id column), alerting webhooks, privacy options (hash embeddings, retention policy).

## 12. Working agreements for AI assistants editing this repo
- Keep modules small and commented; no new heavy dependencies without a reason. Python 3.9-3.11, type hints where cheap.
- Any change to `pipeline.py` or `tracker.py` must keep `pytest` green and add/adjust a synthetic test (fake detector/embedder pattern in `tests/test_pipeline.py`).
- Do not change DB schema without adding a migration in `Database._migrate`.
- Never write demo/fake data into the real `data/` or `logs/` folders.
- Do not mark anything "verified" unless it was actually run; state what was and was not tested.
- Windows-friendly paths (use `pathlib`), no shell-specific assumptions in code.
