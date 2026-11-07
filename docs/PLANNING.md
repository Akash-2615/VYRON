# VYRON — Planning document

Planning, feature inventory, and compute estimates for the intelligent face tracker (Katomaran hackathon).

Related docs: [ARCHITECTURE.md](ARCHITECTURE.md) · [KNOWLEDGE.md](KNOWLEDGE.md) · [PROMPTS.md](PROMPTS.md) · diagram [`assets/vyron-architecture.png`](assets/vyron-architecture.png)

---

## Contents

1. [Planning the app](#1-planning-the-app)
2. [Features the app performs](#2-features-the-app-performs)
3. [Compute load estimates (CPU and GPU)](#3-compute-load-estimates-cpu-and-gpu)
4. [Assumptions and non-goals](#4-assumptions-and-non-goals)
5. [How this maps to the repo](#5-how-this-maps-to-the-repo)

---

## 1. Planning the app

### 1.1 Problem

Build a visitor counter that:

- Ingests **local video**, **webcam**, or **live RTSP**
- Detects and tracks faces
- Recognizes returning people and **auto-registers** new ones
- Logs **exactly one ENTRY and one EXIT per visit** (crop + timestamp + face id)
- Reports a **unique visitor count** that does **not** increase on re-identification
- Persists to **PostgreSQL**, **`events.log`**, and **image crops**
- Exposes a **dashboard** for live ops and audit

### 1.2 Design principles (decided up front)

| Principle | Decision |
|---|---|
| Count truth | Unique = `COUNT(*)` from `faces`; entries/exits from `events` |
| Visit rule | Exactly one ENTRY + one EXIT per visit (time-based presence, not frame hacks) |
| Identity honesty | Crops + embeddings always from the **original** frame |
| Detector assist | Optional preprocess may enhance frames for detection only |
| No-lag live | Never block the live loop waiting for preprocess; use enhanced **if ready**, else original |
| Ops safety | Postgres required via `DATABASE_URL`; no DSN in `config.json`; secrets never logged |
| Testability | Inject fake detector/embedder; lazy ML imports |
| Observability | `sid` on every log line; count audit DB ↔ log; dashboard KPIs |

### 1.3 Planned pipeline (stages)

```
Source → original frame
    ├─(async)→ downscale → dull/dark gate → filters → enhanced?
    └─→ selector (enhanced if ready else original)
         → YOLOv8-face (+ RetinaFace assist)
         → map boxes to original
         → ByteTrack-style tracker
         → blur/size gate → crop (original) → InsightFace embed
         → gallery match / register
         → presence (ENTRY / HIDDEN / RESUMED / EXIT)
         → Postgres + events.log + crops
         → dashboard / ARIA (read-only)
```

### 1.4 Planned work packages

1. **Ingestion** — file / webcam / RTSP with reconnect and latest-frame reader  
2. **Detect + track** — skip frames; coast between detections  
3. **Recognize + register** — quality gate; multi-embed register; gallery cosine match  
4. **Presence** — grace by edge / stationary / occlusion; EXIT at `last_seen`  
5. **Triple sink** — crop → DB → log  
6. **Dashboard** — live control, gallery, logs, analytics, config  
7. **Optional preprocess** — detector-only assist  
8. **Optional ARIA** — read-only natural-language ops assistant  

### 1.5 Success criteria

- Same person across re-entries → **same `face_id`**, unique unchanged  
- Clean session → **entries ≈ exits**, count audit consistent  
- RTSP drop → reconnect without unbounded latency  
- Preprocess on → Live Monitor still shows **original**; Enhanced strip shows assist  
- Tests green without downloading models (`pytest`)

---

## 2. Features the app performs

### 2.1 Video sources

| Feature | Description |
|---|---|
| Local video | Single file or playlist (`video_sources` / `--sources`) |
| Webcam | Device index via OpenCV |
| RTSP / RTSPS | TCP reader, auto-reconnect, latest-frame only |
| Frame skip | `detection_skip_frames` — run detector every N+1 frames |
| Realtime pacing | File playback can pace to source FPS; optional catch-up drop |

### 2.2 Detection and tracking

| Feature | Description |
|---|---|
| YOLOv8-face | Primary face detector (`config.detector`) |
| RetinaFace assist | Fallback / low-conf verification when enabled |
| ByteTrack-style tracker | IoU + velocity; confirm after `min_hits`; coast when skipping |
| Lost-pool rematch | Re-associate briefly lost tracks |
| Away walkers | Person YOLO path for no-face KPI; **excluded** from unique faces |

### 2.3 Recognition and registration

| Feature | Description |
|---|---|
| InsightFace `buffalo_l` | ArcFace 512-d embeddings |
| Gallery match | Cosine similarity vs `match_threshold` |
| Auto-register | After `min_embeds_to_register` good unmatched embeds |
| Quality / blur gate | Min face size + Laplacian variance; rejects bad frames |
| Quality-gate assist | Optional contrast on 112×112 gray **for blur check only** |
| Multi-embedding gallery | Up to `max_embeddings_per_face` per identity |
| Appearance (optional) | Clothing/appearance summary when enabled |

### 2.4 Visit counting and presence

| Feature | Description |
|---|---|
| ENTRY | When a track first owns an identity |
| EXIT | After time-based grace from last detection (`last_seen_ts`) |
| HIDDEN / VISIT_RESUMED | Occlusion / turn-away without closing the visit |
| Exactly-once rule | One entry + one exit per visit |
| Unique count | Rows in `faces` only |
| Re-entry | New visit, same `face_id`, unique unchanged |
| Disturbance freeze | Mass loss / brightness jump / stream gap pauses grace |
| Crash recovery | Open visits closed with `reason=crash_recovery` on restart |
| End-of-stream flush | All open visits closed cleanly |

### 2.5 Optional video preprocessing (OFF by default)

| Feature | Description |
|---|---|
| Async worker | Does not block live tracking |
| Downscale first | Process at detector resolution when enabled |
| Auto-gate | Dull / dark only (brightness + contrast; **not** noise) |
| Filters | Denoise → white balance → gamma → CLAHE → sharpen |
| Selector | Enhanced if ready, else original |
| Hot-reload | `data/runtime/preprocess.json` overrides `config.json` |
| Preview | **Original \| Enhanced** on Config and Live Monitor |

### 2.6 Persistence and audit

| Feature | Description |
|---|---|
| PostgreSQL | `faces`, `embeddings`, `events`, `visits` |
| `events.log` | Session audit; every line has `sid=` |
| `away_events.log` | Separate away stream when configured |
| Crop store | `logs/{entries,exits,faces}/YYYY-MM-DD/` |
| Count reconcile | CLI `count.py` / `verify_log.py` / `final_check.py` + `/api/count` |
| Clear on Start | Optional wipe of gallery + archive logs |

### 2.7 Dashboard (React)

| Feature | Description |
|---|---|
| Overview | KPIs, compact live, charts, count audit |
| Live Monitor | Start/Stop RTSP / local / webcam; annotated original feed |
| Registered Faces | Gallery + Face & Embedding drawer (heatmap, nearest IDs) |
| Event Logs | Paginated ENTRY/EXIT with crops |
| Analytics | Flow / status charts |
| Configuration | Theme, invariants, preprocess card |
| System Log | `events.log` / `away_events.log` viewer |
| ARIA | Optional Groq assistant (read-only; Ctrl/Cmd+K) |

### 2.8 ARIA assistant (optional)

| Feature | Description |
|---|---|
| Natural-language Q&A | Unique counts, dwell, warnings, preprocess explanation |
| Read-only tools | Never changes DB, tracker, or settings |
| Privacy | No face images or 512-d embeddings sent to Groq |
| Live snapshot | KPIs injected each turn |

---

## 3. Compute load estimates (CPU and GPU)

Estimates are **approximate** for planning and interview discussion. Measure real FPS from `events.log` `STATS` lines on your machine. Values assume ~720p–1080p input, YOLOv8n-face class weights, InsightFace `buffalo_l`.

### 3.1 Per-component cost

| Component | When it runs | CPU (modern 4–8 core) | GPU (e.g. RTX 3050/3060 class) |
|---|---|---|---|
| YOLOv8n-face @ ~512–640 | Every `skip+1` frames | ~40–80 ms / detection frame | ~5–10 ms |
| RetinaFace assist (if on) | Empty / low-conf YOLO | ~30–80 ms (occasional) | ~5–15 ms |
| InsightFace embed (ArcFace) | New / updating tracks only | ~60–150 ms / embed | ~10–20 ms |
| Tracker + presence + logging | Every frame | &lt; 2 ms | &lt; 2 ms |
| Preprocess (all frame filters, at detector res) | Async worker when on + gate active | ~1.5–5 ms typical (measured ~1.7 ms avg on one laptop) | N/A (OpenCV CPU path) |
| Person YOLO (away walkers) | Every N detection cycles | ~20–50 ms when run | ~4–8 ms |
| Dashboard API / UI | Poll ~3 s | Negligible vs models | Negligible |
| ARIA (Groq) | On chat only | Network-bound; **no local GPU** | Cloud LLM |

### 3.2 End-to-end throughput (planning targets)

| Mode | CPU-only | GPU-assisted |
|---|---|---|
| `detection_skip_frames = 2` (detect 1 of 3) | ~15–25 FPS typical | ~25–30 FPS realtime with headroom |
| `detection_skip_frames = 3` (committed default) | Often comfortable on laptop CPU | Headroom for RTSP + UI |
| Embed load | Spikes when many **new** faces appear; steady state is cheap | Same pattern, lower spike |

**Why skip helps:** detector cost is divided by `(skip + 1)`. Embeddings are **not** every frame — only when identifying / refreshing a track.

### 3.3 Memory

| Item | Approx. |
|---|---|
| YOLO face weights + runtime | hundreds of MB |
| InsightFace `buffalo_l` | ~hundreds of MB under `~/.insightface` |
| Working set (models + frames + DB client) | ~**1–2 GB** RAM typical |
| GPU VRAM (if used) | ~1–2 GB class for nano face + embed (device-dependent) |

### 3.4 How to enable GPU

- PyTorch **CUDA** build for YOLO (`detector.device`: `cuda` or `auto`)
- `onnxruntime-gpu` (or CUDA ExecutionProvider) for InsightFace ONNX where applicable
- Confirm with `STATS` FPS in `events.log` — do not claim GPU speedup without measuring

### 3.5 Cost-control levers (config)

| Lever | Effect |
|---|---|
| Raise `detection_skip_frames` | Less YOLO |
| Lower `detector.imgsz` | Cheaper detect |
| `stream.process_max_width` | Downscale huge CCTV before the pipeline |
| `preprocess.process_at_detector_resolution` | Keep assist cheap |
| Disable away / appearance / preprocess | Drop optional paths |
| `recognition.max_identify_per_frame` | Cap embed work per frame |

---

## 4. Assumptions and non-goals

**Assumptions**

- One camera / one stream at a time in the dashboard job  
- Wall-clock timestamps for events (video PTS optional later)  
- Faces too small or too blurry are not counted  
- Interview may use a live RTSP URL pasted in the UI  

**Non-goals (for this hackathon build)**

- Multi-camera fusion  
- Sending face images or embeddings to cloud LLMs  
- Replacing Postgres with a cloud DB requirement  
- Enhancing saved crops for “prettier” evidence  

---

## 5. How this maps to the repo

| Plan item | Where it lives |
|---|---|
| Pipeline orchestration | `backend/src/pipeline/` |
| Presence / visits | `backend/src/presence/` |
| Detect / preprocess | `backend/src/detection/` |
| Embed / gallery | `backend/src/recognition/` |
| Tracker | `backend/src/tracking/` |
| DB + logs | `backend/src/storage/`, `backend/src/event_logging/` |
| API + ARIA | `backend/api/` |
| Dashboard | `frontend/` |
| Runtime config | `config.json`, `.env` (`DATABASE_URL`) |
| Architecture diagram | `docs/assets/vyron-architecture.png` |

---

*This project is a part of a hackathon run by https://katomaran.com*
