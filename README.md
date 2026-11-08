# VYRON — Intelligent Face Tracker (powered by ARIA)

Real-time visitor counter: **YOLOv8-face** detection, **ByteTrack-style** tracking, **InsightFace/ArcFace** recognition, auto-registration, **exactly one entry and one exit per visit**, and a **unique visitor count** that does not grow when the same person returns. Outputs go to PostgreSQL, `events.log`, and face crop images. Includes a React dashboard and optional **ARIA** (Groq) assistant.

## Project structure

```
VYRON/
├── backend/                 # Python root (tracker + FastAPI + tests)
│   ├── main.py              # CLI entry (same flags as root main.py)
│   ├── api/                 # FastAPI app, ARIA assistant, services
│   ├── src/                 # core, detection, recognition, tracking, presence, pipeline, storage, …
│   ├── scripts/             # count, final_check, run_rtsp, check_layers, seed_demo, …
│   └── tests/               # pytest (fake detector/embedder; no GPU required)
├── frontend/                # Vite + React + TypeScript dashboard
│   ├── src/pages/           # Overview, Live, Visitors, Events, Analytics, Config, Logs
│   ├── src/components/      # LiveMonitor, StreamLauncher, KPIs, …
│   └── src/features/        # assistant (ARIA), configuration (preprocess)
├── data/                    # videos, models, db, runtime, demo (mostly gitignored)
├── deploy/                  # docker-compose Postgres, MediaMTX, local certs
├── docs/
│   ├── PLANNING.md          # planning, features, CPU/GPU estimates
│   ├── ARCHITECTURE.md      # layers + preprocess flow
│   ├── KNOWLEDGE.md         # invariants + module map
│   ├── PROMPTS.md           # design prompts used
│   └── assets/              # architecture diagram PNG
├── logs/                    # events.log, away_events.log, entry/exit/face crops
├── samples/                 # CSV / export samples
├── config.json              # runtime config (DSN is NOT here — use .env)
├── .env.example             # DATABASE_URL + optional GROQ_* 
├── main.py                  # thin shim → backend/main.py
├── Makefile                 # install / db / run / api / web / test
└── README.md
```

Python imports use `src.*` / `api.*` with `pythonpath=backend`. Paths resolve from repo root via `PROJECT_ROOT`.

---

## 1. Setup instructions

> **Use the committed [`config.json`](config.json) as-is.**  
> It is the **recommended production profile** for VYRON — tuned on our hardware and sample/crowd footage for **cleaner detections**, **correct re-identification**, and **far fewer false “new visitor” splits** (same face wrongly registered again).  
> Changing match / anti-split / tracker values without measuring usually makes accuracy **worse**. Prefer this file first; only retune after a clean `make count` baseline.

### Requirements
- Python **3.9–3.11**, Node.js 18+ (dashboard)
- Docker (local Postgres)
- macOS / Linux / Windows (Windows may need *Microsoft C++ Build Tools* for InsightFace)
- Camera permission if using **Webcam** (grant to Terminal / Python)

### Install (one-time)
```bash
# 1) Clone
git clone https://github.com/Akash-2615/VYRON.git
cd VYRON

# 2) Python env + deps
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
make install                       # backend pip + frontend npm

# 3) Postgres + env  (required — no SQLite fallback at runtime)
make db                            # docker compose -f deploy/docker-compose.yml up -d db
cp .env.example .env
# Edit .env:
#   DATABASE_URL=postgresql://facetrack:facetrack@localhost:5432/facetrack
# Optional ARIA:
#   GROQ_API_KEY=...

# 4) Keep root config.json unchanged (recommended profile — see §3)
```

### Run — CLI
```bash
make run                           # uses config.json video_source
make fresh                         # wipe gallery + archive logs, then run
```

### Run — Dashboard (recommended for demos)
```bash
# Production-style (API serves built UI):
make build-web && make api
# → http://127.0.0.1:8000

# Or hot-reload UI:
#   Terminal 1: make api
#   Terminal 2: make web   → http://127.0.0.1:5173

# Live Monitor → Local video / Webcam / RTSP → Start
# Leave “Clear registered faces” unchecked to keep one shared gallery across sources.
```

### Useful commands
```bash
make test                          # pytest -q (no GPU/models required)
make count                         # reconcile DB ↔ events.log  (expect Consistent)
make check                         # final hygiene checks
make rtsp                          # paste RTSP URL runner
python3 main.py --sources data/videos/v1.mp4 data/videos/v2.mp4
python3 main.py --source rtsp://user:pass@host:554/stream --display
```

### First-run models
- YOLO face weights → `data/models/yolov8n-face.pt` (auto-download)
- InsightFace `buffalo_l` → `~/.insightface` (auto-download)

### Notes
- Postgres DSN lives **only** in `.env` (`DATABASE_URL`) — never in `config.json`
- Missing/unreachable DB → startup fails
- Crops + `events.log` under `logs/`
- Never commit `.env`, `.venv/`, or a real DSN

---

## 2. Assumptions made

1. A **visit** starts when a track gets an identity (**ENTRY**) and ends only after a **time-based** grace from the last real detection (or stream end). Brief occlusion / turn-away resumes the same visit (`VISIT_RESUMED`), not a new unique person.
2. A person who truly leaves and returns later gets a **new entry/exit pair** under the **same `face_id`**; the **unique count is unchanged**.
3. **Unique visitors** = number of rows in the `faces` table. Re-recognition never creates a new face row.
4. Faces below `min_face_px` or too blurry for the quality gate are **not** registered or counted.
5. **One camera / one stream** at a time in the dashboard tracker job.
6. Event timestamps use **wall-clock processing time**; monotonic time is only for grace durations.
7. Optional **preprocess** may enhance frames for the **face detector only**. Crops, embeddings, Live Monitor, away detection, and disturbance stats always use the **original** frame (identity must not drift).
8. Runtime storage is **PostgreSQL** via `DATABASE_URL`. SQLite is for unit tests only.
9. The committed **`config.json` is our proven accuracy profile** (see §3) — not a random default. Use it before retuning.
10. Dashboard **Start** keeps the gallery by default; tick **Clear registered faces on Start** (or use `--fresh`) when you want a wipe.

---

## 3. Recommended `config.json` (system-tuned · please use this)

**Full file:** [`config.json`](config.json) at the repo root — **do not replace with older excerpts**.

### Why these settings (highly recommended)

| Goal | How this config helps |
|---|---|
| **Cleaner detection** | Face YOLO `conf_threshold` **0.12**, `imgsz` **768**, `detection_skip_frames` **0**, `process_max_width` **1280** — strong recall without downscaling away small faces. |
| **Correct identification** | ArcFace `match_threshold` **0.52** + `min_match_gap` / `soft_match_margin` — same person rematches; strangers stay apart. |
| **No same face as a new unique** | `anti_split_threshold` **0.48**, `min_embeds_to_register` **2**, quality gate, shared gallery across file/webcam/RTSP — stops twin IDs / false `FACE_REGISTERED`. |
| **Stable visits** | Presence grace + ByteTrack `low_conf` recovery — fewer dropouts that look like a “new” person. |
| **Honest away people** | `away_walkers` on — no-face walkers do **not** inflate `faces` unique count. |

On our runs (sample + crowd clips + Live Monitor), moving to this profile produced **more accurate unique counts**, **entries ≈ exits**, and **fewer false new registrations** for people already in the gallery. Demo video: https://youtu.be/1fPQ02_u_wc · sample screenshots in §6.

> **Please use this `config.json` on your system.**  
> Copy the repo file as-is. If you must change one knob later: too many new IDs → lower match slightly or keep anti-split; wrong merges → raise `match_threshold` / `min_match_gap`. Always re-check with `make count`.

### Key values (committed now)

```json
{
  "video_source": "data/videos/video_sample1.mp4",
  "detection_skip_frames": 0,

  "detector": {
    "weights": "data/models/yolov8n-face.pt",
    "conf_threshold": 0.12,
    "imgsz": 768,
    "device": "auto"
  },

  "recognition": {
    "model_pack": "buffalo_l",
    "match_threshold": 0.52,
    "soft_match_margin": 0.06,
    "min_match_gap": 0.04,
    "anti_split_threshold": 0.48,
    "register_min_det_score": 0.38,
    "min_embeds_to_register": 2,
    "max_embeddings_per_face": 5,
    "embed_retry_interval": 1,
    "max_embed_attempts": 16,
    "max_identify_per_frame": 12
  },

  "quality": {
    "min_face_px": 10,
    "min_blur_var": 6.0
  },

  "tracker": {
    "high_conf": 0.30,
    "new_track_conf": 0.30,
    "low_conf": 0.12,
    "iou_threshold": 0.22,
    "low_iou_threshold": 0.18,
    "min_hits": 1,
    "max_lost_frames": 50,
    "reentry_grace_frames": 90,
    "reentry_grace_sec": 30.0
  },

  "presence": {
    "exit_grace_seconds": 10,
    "edge_exit_grace_seconds": 2,
    "stationary_grace_seconds": 45,
    "occlusion_grace_multiplier": 2.0,
    "max_stream_gap_seconds": 20
  },

  "away_walkers": { "enabled": true },
  "appearance": { "enabled": true, "describe_on_register": true },

  "stream": {
    "realtime": true,
    "catch_up_drop": true,
    "process_max_width": 1280
  },

  "output": {
    "live_preview": "data/runtime/live.jpg",
    "live_max_width": 960,
    "live_every_n_frames": 2
  },

  "storage": {
    "backend": "postgres",
    "events_log": "logs/events.log",
    "reset_on_start": false
  },

  "preprocess": { "enabled": false }
}
```

- `detection_skip_frames = 0` → detect every frame (best accuracy; raise only if CPU-bound).
- Postgres DSN is **not** in this file — set `DATABASE_URL` in `.env` only.

---

## 4. AI Planning document and Architecture diagram

### AI Planning document
Full planning write-up (planning steps, feature list, CPU/GPU compute estimates):

**→ [`docs/PLANNING.md`](docs/PLANNING.md)**

| Section in PLANNING.md | Contents |
|---|---|
| Planning the app | Problem, design principles, pipeline stages, work packages, success criteria |
| Features | Sources, detect/track, recognize, presence/counting, preprocess, storage, dashboard, ARIA |
| Compute estimates | Per-component CPU vs GPU cost, throughput targets, memory, cost-control levers |

**Short plan summary**

1. Robust frame ingestion (file / webcam / RTSP)  
2. Detect every N frames; track in between  
3. Embed only when identifying a track (cheap)  
4. Gallery match + safe auto-register  
5. Presence state machine → **1 ENTRY + 1 EXIT per visit**  
6. Triple sink: crop images + Postgres + `events.log`  
7. Unique count = `COUNT(*) FROM faces`  
8. Dashboard + optional detector-only preprocess + optional ARIA  

**Compute (summary)** — measure with `STATS` in `events.log`:

| Component | CPU (4–8 core) | GPU (e.g. RTX 3050/3060) |
|---|---|---|
| YOLOv8n-face | ~40–80 ms / detection frame | ~5–10 ms |
| InsightFace embed | ~60–150 ms (new tracks) | ~10–20 ms |
| Tracker + logging | &lt; 2 ms / frame | &lt; 2 ms |
| Throughput (skip≈2) | ~15–25 FPS | ~25–30 FPS realtime |
| RAM | ~1–2 GB typical | — |

More detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`docs/KNOWLEDGE.md`](docs/KNOWLEDGE.md) · [`docs/PROMPTS.md`](docs/PROMPTS.md)

### Architecture diagram

![VYRON architecture](docs/assets/vyron-architecture.png)

```mermaid
flowchart LR
    A[Webcam / file / RTSP] --> B[FrameSource · original]
    B --> PP[Optional async preprocess]
    PP --> S{Enhanced ready?}
    B --> S
    S -->|yes| D[YOLOv8-face + RetinaFace assist]
    S -->|no / off| D
    D --> M[Map boxes to original]
    M --> F[ByteTrack-style tracker]
    F --> H[Blur gate → crop original → InsightFace]
    H --> I{match?}
    I -->|yes| J[FACE_RECOGNIZED]
    I -->|no × N| K[FACE_REGISTERED]
    J --> L[ENTRY / EXIT · 1 per visit]
    K --> L
    L --> N[(PostgreSQL)]
    L --> Q[logs/ + crops]
    N --> UI[Dashboard + ARIA]
    Q --> UI
```

**Invariant:** enhanced frames feed the **detector only**. Crops, embeddings, and Live Monitor always use the **original** frame.

---

## 5. Loom / YouTube demo video

**Video link:** https://youtu.be/1fPQ02_u_wc

> **Required by evaluators.** If you do not submit an explanatory video, your submission will not be reviewed.

Suggested demo script: architecture overview → `config.json` frame-skip → Start on video/webcam/RTSP → Live Monitor → registered face + entry/exit crops → `events.log` / DB unique count → optional preprocess Original/Enhanced → optional ARIA question.

---

## 6. Sample output (after a clean run)

**Screenshot pack (DOCX):** [`docs/assets/output/VYRON_Sample_Output_Screenshots.docx`](docs/assets/output/VYRON_Sample_Output_Screenshots.docx)  
Same images: [`docs/assets/output/`](docs/assets/output/) · copy under [`samples/output_screenshots/`](samples/output_screenshots/)

### Captured run (local video · Postgres · 8 Nov 2026)

| Metric | Value (Configuration audit) | Later Live Monitor |
|---|---|---|
| Unique visitors (`faces`) | **54** | **58–59** (session continued) |
| Entries / Exits | **114 / 114** | **149 / 149** |
| Count audit | **Consistent** (DB gallery = log gallery) | — |
| Peak occupancy | 20 | — |
| Source | Local MP4 | `video_sample1.mp4` / `record_…mp4` |

Also on disk after a run:

- `logs/events.log` — `SYSTEM_START` … `FACE_REGISTERED` / `ENTRY` / `EXIT` … `COUNT_SNAPSHOT` / `SYSTEM_STOP`
- Crops — `logs/faces/`, `logs/entries/`, `logs/exits/` (dated folders)
- Postgres — `faces`, `embeddings`, `events`, `visits`
- Reconcile — `make count`

### What each screenshot shows

| # | File | Explanation |
|---|---|---|
| 1 | [`01_overview_idle.png`](docs/assets/output/01_overview_idle.png) | Overview before Start — KPIs at zero, empty Live Monitor. |
| 2 | [`02_overview_live_kpis.png`](docs/assets/output/02_overview_live_kpis.png) | Overview mid-run — unique 56, entries/exits, live IDs, charts, Live Events, AI pipeline. |
| 3 | [`03_live_monitor_ids_noface.png`](docs/assets/output/03_live_monitor_ids_noface.png) | Live Monitor — face ID overlays + **NO-FACE** away tracks (not in unique count). |
| 4 | [`04_live_monitor_crowd.png`](docs/assets/output/04_live_monitor_crowd.png) | Dense crowd on `video_sample1.mp4` — many simultaneous tracks + Live Events. |
| 5 | [`05_registered_faces_gallery.png`](docs/assets/output/05_registered_faces_gallery.png) | Registered Faces gallery — unique IDs, visits, dwell, appearance tags. |
| 6 | [`06_visitor_appearance.png`](docs/assets/output/06_visitor_appearance.png) | Visitor drawer — visit history + OpenCLIP appearance crop/tags. |
| 7 | [`07_visitor_embedding.png`](docs/assets/output/07_visitor_embedding.png) | Face & Embedding — ArcFace 512-d samples, heatmap, match threshold. |
| 8 | [`08_visitor_closest_identities.png`](docs/assets/output/08_visitor_closest_identities.png) | Closest other identities — cosine scores below threshold (no false merge). |
| 9 | [`09_aria_assistant.png`](docs/assets/output/09_aria_assistant.png) | ARIA answers “Who stayed longest?” against live gallery stats. |
| 10 | [`10_event_logs.png`](docs/assets/output/10_event_logs.png) | Event Logs — ENTRY/EXIT table with crop evidence, track, frame. |
| 11 | [`11_analytics_noface.png`](docs/assets/output/11_analytics_noface.png) | Analytics — no-face / away people excluded from unique count. |
| 12 | [`12_configuration_audit.png`](docs/assets/output/12_configuration_audit.png) | Configuration — invariants + **Consistent** DB↔log count audit. |
| 13 | [`13_system_log.png`](docs/assets/output/13_system_log.png) | System Log — `events.log` tail (`FACE_RECOGNIZED`, `EXIT`, crop paths). |

---

This project is a part of a hackathon run by https://katomaran.com
