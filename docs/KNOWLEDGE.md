# VYRON — Project knowledge

Authoritative context for the codebase: what exists, why, invariants, and config knobs. Read sections **1–5** before changing pipeline or counting logic.

## 1. Goal

Katomaran hackathon: AI visitor counter for **video (dev)**, **webcam**, or **live RTSP**.
Detect → track → recognize → auto-register faces; log **exactly one entry and one exit per visit** (crop + timestamp + face id) to log file, image folders, and Postgres; report **unique visitors** that do **not** grow on re-identification.

**Stack:** Ultralytics YOLOv8-face (`yolov8n-face.pt`) + optional RetinaFace assist; COCO YOLOv8n person for away/NO-FACE; InsightFace/ArcFace (`buffalo_l`, 512-d, not `face_recognition`); ByteTrack-style tracker; Python backend; PostgreSQL (SQLite tests only); `config.json`; React dashboard; optional ARIA (Groq).

**Submission checklist (README):** setup, assumptions, sample config, planning notes, architecture diagram, Loom/YouTube link, sample output, last line exactly  
`This project is a part of a hackathon run by https://katomaran.com`.

Planning / features / compute: [`docs/PLANNING.md`](PLANNING.md).  
Design prompts: [`docs/PROMPTS.md`](PROMPTS.md). Layout & diagrams: [`docs/ARCHITECTURE.md`](ARCHITECTURE.md).

## 2. What is built

| Area | State |
|---|---|
| Detect / embed / track / gallery auto-register | Built |
| Presence visit SM (1 entry + 1 exit / visit) | Built + unit-tested |
| `events.log` + crops + Postgres | Built + tested |
| Unique = `COUNT(faces)` + count audit CLI/API | Built + tested |
| Multi-video one gallery (`--sources`) | Built + unit-tested |
| RTSP TCP latest-frame + reconnect; webcam; local file | Built |
| Dashboard: Overview, Live Monitor, Faces, Events, Analytics, Config, System Log | Built |
| Optional preprocess (detector-only, OFF by default) | Built + tested |
| Away walkers (person YOLO → NO-FACE, not unique) | Built |
| Appearance tags (OpenCLIP, optional) | Built |
| ARIA read-only assistant | Built (needs `GROQ_API_KEY` live) |
| Architecture diagram + demo video in README | Shipped (`docs/assets/…`, YouTube link in README §5) |

Honest gaps: thresholds still need footage-specific tuning; identity can mis-match in heavy crowd / blur / profile. Prefer measured FPS from `STATS` over claimed numbers.

## 3. Repository map

```
config.json                 runtime config (repo root)
main.py                     shim → backend/main.py
Makefile / pyproject.toml   DX; pytest pythonpath=backend
backend/                    Python root (src + api)
  main.py                   CLI
  src/core/                 config, clock, paths
  src/session_svc/          session bootstrap / crash recovery
  src/event_logging/        EventLogger + crops
  src/storage/              Database + schema + open_database
  src/detection|recognition|tracking|video_io|presence|pipeline|reporting|viz/
  api/                      FastAPI + ARIA + services
  scripts/                  count, final_check, run_rtsp, check_layers, …
  tests/
frontend/                   Vite + React dashboard
data/                       videos, models, db, runtime, demo (mostly gitignored)
deploy/                     docker-compose Postgres, MediaMTX
docs/                       PLANNING, KNOWLEDGE, ARCHITECTURE, PROMPTS, assets/
logs/  samples/             deliverables at repo root
```

Layers: `python backend/scripts/check_layers.py`. Paths via `src.core.paths.PROJECT_ROOT`.

## 4. Pipeline (do not break)

1. **Detect** every `detection_skip_frames+1` frames (else tracks coast). Face YOLO (± Retina if enabled).
2. Optional **preprocess** → detector input = enhanced if ready else original; boxes **mapped back** to original.
3. **ByteTrack-style** confirm after `min_hits`; drop after `max_lost_frames` / leave-frame. High then low conf association.
4. **Identify (2-step):** size + blur gate → crop from **original** → InsightFace embed → gallery.
   - Hard / soft / ambiguous claim via `match_threshold`, `soft_match_margin`, `min_match_gap`
   - else collect embeds → after `min_embeds_to_register` → mean embed → **anti_split** check → else `FACE_REGISTERED`
5. **Away (optional):** person YOLO → separate tracker → NO-FACE overlay / away DB — **not** in unique faces.
6. **Presence:** identity + no owner → **ENTRY**; lost → **HIDDEN** (grace seconds); return → **VISIT_RESUMED**; deadline/stream end → **EXIT** at `last_seen_ts`.
7. **Write order:** crop (atomic) → DB → `events.log`. Clock: `wall_now()` for stamps; `mono()` for durations only.

### Counts

| Metric | Definition |
|---|---|
| Unique | `COUNT(*)` from `faces` |
| Entries / exits | `events` rows `ENTRY` / `EXIT` |
| Returning | `visit_count > 1` |
| Inside / open | entries without exits (`visits.exit_ts IS NULL`) |

Away walkers are **not** unique faces.

### Invariants

- One entry + one exit per visit; returning person never new `faces` row.
- DB count and log-derived count must agree (`counting.py`, `scripts/count.py`, `GET /api/count`).
- No identity from bad frames or a single embed when 2-step is ON (`min_embeds_to_register ≥ 2`).
- Every log line has `sid=<session_id>`; archive logs on wipe (`--fresh` / clear checkbox).
- Crash recovery: one EXIT at `last_seen`, never invent ENTRY.
- Preprocess OFF by default; when on, **never** feeds embeddings/crops/live/away/disturbance.
- `quality_assist` = blur-gate 112×112 gray only (not a frame filter).
- Live overlay: outer green body box on identified faces is `body_box()` expand — **not** person-YOLO.
- Heavy ML lazy; pipeline injectable for tests.
- Runtime Postgres only (`DATABASE_URL`); no SQLite fallback; no DSN in `config.json`.

## 5. Config (`config.json`)

Committed file at repo root is the **recommended** profile. DSN **only** via `.env` → `DATABASE_URL`.

### 5.1 Current committed highlights

| Key | Value now | Role |
|---|---|---|
| `detection_skip_frames` | `0` | Detect every frame |
| `detector.conf_threshold` | `0.12` | Face YOLO confidence |
| `detector.imgsz` | `768` | Detector input size |
| `retina_face.enabled` | `false` | Retina assist off |
| `recognition.match_threshold` | `0.52` | Hard ArcFace match |
| `recognition.soft_match_margin` | `0.06` | Soft rematch band |
| `recognition.min_match_gap` | `0.04` | Margin vs 2nd-best gallery face |
| `recognition.anti_split_threshold` | `0.48` | Block twin register when close to existing |
| `recognition.min_embeds_to_register` | `2` | Need ≥2 embeds before new unique |
| `stream.process_max_width` | `1280` | Downscale before YOLO if wider |
| `stream.realtime` / `catch_up_drop` | `true` | Pace file playback; drop when lagging |
| `away_walkers.enabled` | `true` | Person YOLO away path |
| `appearance.enabled` | `true` | OpenCLIP clothing tags |
| `preprocess.enabled` | `false` | Detector enhance off |

### 5.2 Two-step embed matching — ON / OFF

There is **no single boolean**. “2-step” = detect boxes first, then **embed + multi-embed gallery claim** before minting a new `face_id`. Controlled under `recognition` (+ quality gate).

#### Turn **ON** (recommended — fewer twin IDs; slower first ENTRY)

```json
"recognition": {
  "match_threshold": 0.52,
  "soft_match_margin": 0.06,
  "min_match_gap": 0.04,
  "anti_split_threshold": 0.48,
  "min_embeds_to_register": 2,
  "register_min_det_score": 0.38,
  "max_embed_attempts": 16,
  "embed_retry_interval": 1
},
"quality": { "min_face_px": 10, "min_blur_var": 6.0 }
```

What this does:

1. Each good crop → embed → try hard/soft/ambiguous **claim** of an existing face.  
2. If not claimed, buffer embeds until `min_embeds_to_register`.  
3. Mean embed: claim again; if still near an existing face (`sim ≥ anti_split_threshold`) → **anti-split rematch**, no new unique.  
4. Only then `FACE_REGISTERED`.

#### Turn **OFF** / weaken (faster register; more false new uniques)

```json
"recognition": {
  "match_threshold": 0.52,
  "soft_match_margin": 0.0,
  "min_match_gap": 0.0,
  "anti_split_threshold": 1.01,
  "min_embeds_to_register": 1,
  "register_min_det_score": 0.25,
  "max_embed_attempts": 8,
  "embed_retry_interval": 1
}
```

| Knob | Effect when “off” |
|---|---|
| `min_embeds_to_register: 1` | Register after **one** embed (no multi-embed confirm) |
| `anti_split_threshold: 1.01` | Anti-split never fires (cosine ≤ 1) |
| `soft_match_margin: 0` | No soft rematch band |
| `min_match_gap: 0` | No 2nd-best gap requirement |

Use OFF only for debugging speed — **not** for demo/submission counting.

### 5.3 Preset A — smooth video run (CPU / laggy machines)

Goal: steadier FPS, less hitching, still usable IDs.

```json
"detection_skip_frames": 2,
"detector": { "conf_threshold": 0.20, "imgsz": 512, "device": "auto" },
"recognition": {
  "match_threshold": 0.52,
  "soft_match_margin": 0.06,
  "min_match_gap": 0.04,
  "anti_split_threshold": 0.48,
  "min_embeds_to_register": 2,
  "max_identify_per_frame": 6,
  "embed_retry_interval": 2,
  "max_embed_attempts": 10
},
"away_walkers": { "enabled": true, "detect_every_n": 2, "max_head_roi_per_cycle": 3 },
"appearance": { "enabled": false },
"stream": {
  "realtime": true,
  "catch_up_drop": true,
  "process_max_width": 960
},
"output": {
  "live_jpeg_quality": 60,
  "live_max_width": 720,
  "live_every_n_frames": 3
},
"preprocess": { "enabled": false }
```

Also helpful: raise `tracker.max_lost_frames` if skip is high (config loader already bumps it with skip).

### 5.4 Preset B — high-quality detector (accuracy first · current repo default)

Goal: best recall + safer identity; heavier CPU/GPU.

```json
"detection_skip_frames": 0,
"detector": { "conf_threshold": 0.12, "imgsz": 768, "device": "auto" },
"retina_face": { "enabled": false },
"recognition": {
  "match_threshold": 0.52,
  "soft_match_margin": 0.06,
  "min_match_gap": 0.04,
  "anti_split_threshold": 0.48,
  "min_embeds_to_register": 2,
  "max_identify_per_frame": 12,
  "embed_retry_interval": 1,
  "max_embed_attempts": 16
},
"quality": { "min_face_px": 10, "min_blur_var": 6.0 },
"away_walkers": { "enabled": true, "detect_every_n": 1 },
"appearance": { "enabled": true, "describe_on_register": true },
"stream": {
  "realtime": true,
  "catch_up_drop": true,
  "process_max_width": 1280
},
"output": {
  "live_jpeg_quality": 72,
  "live_max_width": 960,
  "live_every_n_frames": 2
},
"preprocess": { "enabled": false }
```

Optional extra accuracy (slower): `"retina_face": { "enabled": true }` and/or `"process_max_width": 0` (no width cap).

### 5.5 Quick tuning cheatsheet

| Symptom | Change |
|---|---|
| Too many new IDs / twins | Keep 2-step ON; raise `anti_split_threshold` toward `match_threshold`; raise `min_embeds_to_register` to 3 |
| Wrong merges (two people → one ID) | Raise `match_threshold` / `min_match_gap`; lower or disable soft margin |
| Lag / stutter | Use Preset A (skip, smaller `imgsz`, lower `process_max_width`, fewer live JPEGs) |
| Missed small faces | Preset B; lower `conf_threshold`; raise `imgsz`; `process_max_width` 1280+ |
| False exits | Raise `presence.exit_grace_seconds` / `tracker.max_lost_frames` |
| Live preview heavy | Lower `live_jpeg_quality`, `live_max_width`, raise `live_every_n_frames` |

**Live overlay:** `…` = tracked pending ID; green `ID N` = known; magenta `NO-FACE` = person path.

## 6. Database

`make db` then `DATABASE_URL` in `.env`.

- `faces`, `embeddings` (512-d), `events`, `visits` (+ away_*)
- Tests may use SQLite via `Database(path, reset=True)`; production path is `open_database` → Postgres only.

## 7. API

`uvicorn api.server:app --app-dir backend --port 8000`

- Read: `/api/health`, `/summary`, `/count`, `/timeline`, `/visitors`, `/events`, `/logs`, `/tracker/status`, `/videos`, `/assistant/status`, `/preprocess`, `/preprocess/preview`, live JPEG
- Write: `/api/tracker/start|stop`, `/api/preprocess` PUT + reset (runtime JSON only), `/api/assistant/chat` (SSE)
- Serves `/media` from `logs/` and `frontend/dist` when built
- ARIA: read-only tools; no images/embeddings to Groq

## 8. Frontend

Vite + React + TS. Pages: Overview, Live Monitor, Registered Faces, Event Logs, Analytics, Configuration, System Log. ARIA pill / Ctrl+K. Dev: `make api` + `make web` (:5173). StreamLauncher: RTSP / local / webcam; shared gallery unless Clear on Start.

## 9. Commands

```bash
make install && make db && cp .env.example .env
make run          # or: python3 main.py
make fresh        # wipe session
make api          # :8000
make web          # :5173
make test         # pytest -q
make count        # DB ↔ log
make check        # final_check.py
make rtsp         # paste URL runner
```

## 10. Limitations (honest)

1. Thresholds need footage-specific tuning; wrong IDs still possible in crowd/blur/profile.
2. ENTRY lags first sight until enough good embeds (by design when 2-step ON).
3. Faces that never pass quality/embed are not counted (by design).
4. Tracker is IoU+velocity (appearance assist optional / separate).
5. API is hackathon-scale (full-table stats, open CORS).
6. Green outer box on identified visitors is geometric expand of the face box, not person-YOLO.

## 11. Before submit (P0)

1. Clean run on sample / webcam / RTSP — unique stable, entries ≈ exits (`make count`).
2. Keep sample screenshots / DOCX under `docs/assets/output/` · `samples/output_screenshots/`.
3. README §5 YouTube link + §6 sample output filled.
4. Rehearse pipeline, presence, preprocess split, counting, ARIA from [`PROMPTS.md`](PROMPTS.md).
5. Confirm README last line + form submission.

## 12. Edit rules

- Keep invariants; add/adjust synthetic tests for pipeline/tracker changes.
- Schema changes → migration in `Database._migrate` / `schema.py`.
- Never dump demo data into real `data/` or `logs/`.
- Do not claim verified unless you ran it.
- Prefer `pathlib`; Python 3.9–3.11.
