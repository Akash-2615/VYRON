# VYRON — Project knowledge

Authoritative context for the codebase: what exists, why, invariants, and what remains for submission. Read sections **1–4** before changing pipeline or counting logic.

## 1. Goal

Katomaran hackathon: AI visitor counter for **video (dev)** or **live RTSP (interview)**.
Detect → track → recognize → auto-register faces; log **exactly one entry and one exit per visit** (crop + timestamp + face id) to log file, image folders, and Postgres; report **unique visitors** that do **not** grow on re-identification.

**Stack:** YOLOv8-face (+ RetinaFace assist), InsightFace/ArcFace (`buffalo_l`, not `face_recognition`), ByteTrack-style tracker, Python backend, PostgreSQL (SQLite tests only), `config.json` with frame-skip, React dashboard, optional ARIA (Groq).

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
| ARIA read-only assistant | Built (needs `GROQ_API_KEY` live) |
| Architecture diagram in README | Shipped (`docs/assets/vyron-architecture.png`) |

Honest gaps: tune thresholds on your footage; fill README **demo video URL** and concrete **sample output** numbers after a clean final run. Prefer stating what you actually measured over claiming unverified FPS.

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

1. **Detect** every `detection_skip_frames+1` frames (else tracks coast).
2. Optional **preprocess** → detector input = enhanced if ready else original; boxes **mapped back** to original.
3. **ByteTrack-style** confirm after `min_hits`; drop after `max_lost_frames` / leave-frame.
4. **Identify** (confirmed + freshly matched): size + blur gate → crop from **original** → embed → gallery.
   - ≥ `match_threshold` → `FACE_RECOGNIZED`
   - else collect embeds → after `min_embeds_to_register` → `FACE_REGISTERED`
5. **Presence:** identity + no owner → **ENTRY**; lost → **HIDDEN** (grace seconds); return → **VISIT_RESUMED**; deadline/stream end → **EXIT** at `last_seen_ts`. Handover / BODY_ONLY / disturbance freeze as documented in code.
6. **Write order:** crop (atomic) → DB → `events.log`. Clock: `wall_now()` for stamps; `mono()` for durations only.

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
- No identity from bad frames or a single embed.
- Every log line has `sid=<session_id>`; archive logs on wipe (`--fresh` / clear checkbox).
- Crash recovery: one EXIT at `last_seen`, never invent ENTRY.
- Preprocess OFF by default; when on, **never** feeds embeddings/crops/live/away/disturbance.
- `quality_assist` = blur-gate 112×112 gray only (not a frame filter).
- Heavy ML lazy; pipeline injectable for tests.
- Runtime Postgres only (`DATABASE_URL`); no SQLite fallback; no DSN in `config.json`.

## 5. Config (`config.json`)

Important keys (committed values may differ slightly — open the file):

- `detection_skip_frames`, `detector.*`, `recognition.match_threshold` (committed **~0.52**), `quality.*`
- `tracker.*`, `presence.*` (grace seconds; `stationary >= exit >= edge`)
- `storage.backend=postgres`, `reset_on_start=false`, `postgres={}`
- `stream.realtime`, `catch_up_drop`, `process_max_width`
- `away_walkers.enabled`, `appearance.enabled`, `preprocess.*` (all assist off by default)
- DSN **only** via `.env` → `DATABASE_URL`

**Tuning:** too many new IDs → lower match threshold or relax quality; merges → raise threshold; false exits → raise `max_lost_frames` / grace; slow → raise skip / lower `imgsz`.

**Live overlay:** `…` = tracked pending ID; green `ID N` = known.

## 6. Database

`make db` then `DATABASE_URL` in `.env`.

- `faces`, `embeddings` (512-d), `events`, `visits` (+ away_*)
- Tests may use SQLite via `Database(path, reset=True)`; production path is `open_database` → Postgres only.

## 7. API

`uvicorn api.server:app --app-dir backend --port 8000`

- Read: `/api/health`, `/summary`, `/count`, `/timeline`, `/visitors`, `/events`, `/logs`, `/tracker/status`, `/assistant/status`, `/preprocess`, `/preprocess/preview`, live JPEG
- Write: `/api/tracker/start|stop`, `/api/preprocess` PUT + reset (runtime JSON only), `/api/assistant/chat` (SSE)
- Serves `/media` from `logs/` and `frontend/dist` when built
- ARIA: read-only tools; no images/embeddings to Groq

## 8. Frontend

Vite + React + TS. Pages: Overview, Live Monitor, Registered Faces, Event Logs, Analytics, Configuration, System Log. ARIA pill / Ctrl+K. Dev: `make api` + `make web` (:5173).

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

1. Thresholds need footage-specific tuning.
2. ENTRY lags first sight until enough good embeds; EXIT uses `last_seen` after grace.
3. Faces that never pass quality/embed are not counted (by design).
4. Tracker is IoU+velocity (appearance assist optional / separate).
5. API is hackathon-scale (full-table stats, open CORS).
6. README still needs the **video link** and filled **sample output** after your submission run.

## 11. Before submit (P0)

1. Clean run on sample / webcam / RTSP — unique stable, entries ≈ exits.
2. Export samples (`report.py`), keep representative `logs/` crops.
3. Paste Loom/YouTube into README; fill sample output.
4. Rehearse explaining pipeline, presence, preprocess split, counting, ARIA from [`PROMPTS.md`](PROMPTS.md).
5. Confirm README last line + form submission.

## 12. Edit rules

- Keep invariants; add/adjust synthetic tests for pipeline/tracker changes.
- Schema changes → migration in `Database._migrate` / `schema.py`.
- Never dump demo data into real `data/` or `logs/`.
- Do not claim verified unless you ran it.
- Prefer `pathlib`; Python 3.9–3.11.
