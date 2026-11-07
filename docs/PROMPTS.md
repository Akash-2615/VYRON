# AI prompts used while building VYRON

Planning and implementation prompts that shaped the system. Each row is a real design ask: invariants first, then modules, then verification.

| # | Tool | Prompt (intent) | What it produced | Verified |
|---|------|-----------------|------------------|----------|
| 1 | Cursor | Design a visitor counter for YOLOv8-face + InsightFace + ByteTrack-style tracking with **exactly one ENTRY and one EXIT per visit**, unique count = `COUNT(faces)`, file + RTSP sources, Postgres + `events.log` + crops, and a `config.json` frame-skip. Prefer injectable detector/embedder so tests run without models. | Package layout, `FaceTrackingPipeline`, gallery register/match, CLI `main.py`, initial README/architecture | Unit tests with fake detector/embedder; layer rules in `docs/ARCHITECTURE.md` |
| 2 | Cursor | Replace frame-count exit heuristics with a **time-based presence state machine**: PRESENT → HIDDEN (grace by edge / stationary / occlusion) → EXIT at `last_seen_ts`; VISIT_RESUMED; lost-pool rematch; disturbance freeze; `visits` table; API + dashboard + tests S1–S9. | `src/presence/`, pipeline wiring, DB visits, `tests/test_presence.py`, `presence_report.py` | `pytest` presence suite green |
| 3 | Cursor | Make timestamps and sessions interview-safe: single clock (`wall_now` / `mono`), ISO-ms stamps, `sid=` on every log line, archive logs on reset, exactly one `SYSTEM_STOP`, separate `away_events.log`, paired TRACK_STARTED/ENDED. | `src/core/clock.py`, session bootstrap, event logger, README event table | `tests/test_clock.py`, `tests/test_session_logging.py` |
| 4 | Cursor | Counts must be retrievable and reconcilable: shared `counting.py`, `COUNT_SNAPSHOT`, CLI `count.py` / `verify_log.py` / `final_check.py`, `GET /api/count`, Overview verify badge + Config audit card. DB and log must agree for a clean session. | `src/counting.py`, API/services, Overview/Config UI, counting tests | `pytest` counting + `make count` path documented |
| 5 | Cursor | Runtime DB is **Postgres only** via `DATABASE_URL` (no DSN in `config.json`, no SQLite fallback). Missing/bad DSN → hard error; logs never print passwords; crops/`events.log` stay on disk under `logs/`. | `open_database`, health `backend=postgres`, `.env.example` | Health/smoke + db tests |
| 6 | Cursor | Optional **async preprocess for the face detector only**: downscale → dull/dark auto-gate (brightness+contrast, not noise) → denoise/WB/gamma/CLAHE/sharpen. Selector: enhanced if ready else original (no live wait). Map boxes back to original → tracker → crops/embeddings from **original**. `quality_assist` is blur-gate crop only, not a frame filter. Hot-reload `data/runtime/preprocess.json`; Config card + Original/Enhanced preview. | `src/detection/preprocess.py`, pipeline/`main.py` detect_frame path, Config/Live UI, preprocess tests | `pytest` preprocess (20); UI Original/Enhanced labels |
| 7 | Cursor | Ship **ARIA**: read-only Groq assistant over FastAPI tools + SSE; live KPI snapshot; whitelist tags/nav; never mutate tracker/DB/settings; never send images or 512-d embeddings to the model. Top-bar pill + Ctrl/Cmd+K. | `api/assistant*`, `frontend/src/features/assistant/*`, `.env.example` | Assistant tests + build; needs `GROQ_API_KEY` for live chat |
| 8 | Cursor | Away walkers (person YOLO) must not inflate unique faces; wrong no-face→face linking: promote/claim only with strong head-band overlap; FACE_REGISTERED always opens ENTRY; harden Unique/Entries/log-diff KPIs. | Pipeline claim/promote, counting/session fixes, tests | Away + counting tests |
| 9 | Cursor | Dashboard Start: keep gallery by default; **Clear registered faces** must wipe then start. Hung webcam workers must not leave Start stuck (`thread_alive` vs `running`). | `tracker_job` reclaim, StreamLauncher stop-retry | Manual Start/Stop after reclaim |
| 10 | Cursor | Produce a single architecture-diagram prompt covering sources, preprocess split, detector fallback, tracker, crops/embeddings on original, Postgres/logs outputs, ARIA, and invariants — then place the image in README. | `docs/assets/vyron-architecture.png`, README Architecture section | Image linked in README |

## Prompt themes (why these asks)

1. **Invariants before features** — one entry/exit per visit; unique = faces rows; original frame for identity evidence.
2. **Fail closed on ops** — Postgres required; secrets out of config/logs; session `sid` + archive on wipe.
3. **Detector assist ≠ identity path** — preprocess may brighten detection; embeddings and crops stay honest.
4. **Observable system** — count audit, COUNT_SNAPSHOT, ARIA read-only tools, Original/Enhanced preview.
5. **Test without GPUs** — inject fake detector/embedder; keep heavy ML imports lazy.

## How to talk through this in review

- Walk prompt **1 → 2 → 4**: architecture → presence → countable truth.
- Walk prompt **6**: why enhanced never feeds InsightFace or saved crops.
- Walk prompt **7**: what ARIA can and cannot do (read-only, no embeddings uploaded).
- Point at `docs/ARCHITECTURE.md` + the README diagram for the end-to-end picture.
