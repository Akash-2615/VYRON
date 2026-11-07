# Backend

Python tracker, FastAPI API, CLIs, and tests. Keep package names `src` and `api`
(import paths unchanged). Run from the **repo root** with `backend/` on
`PYTHONPATH` (see root `pyproject.toml` / `python main.py` shim).

```
backend/
  main.py              CLI entry (real implementation)
  src/                 domain packages (layered — see docs/ARCHITECTURE.md)
    core/              config, clock, paths
    storage/           Database + bootstrap
    event_logging/     EventLogger + crops
    session_svc/       session start/stop / crash recovery
    detection/         YOLO face, RetinaFace, person YOLO
    recognition/       embedder, gallery, appearance
    tracking/          ByteTrackLite
    video_io/          FrameSource + live JPEG publisher
    presence/          visit state machine
    pipeline/          FaceTrackingPipeline + tracker_job
    reporting/         counting / reconcile
    viz/               overlays
  api/                 FastAPI app + assistant + analytics services
  scripts/             CLIs (count, verify, golden, RTSP helper, …)
  tests/               pytest suite
  requirements.txt
  requirements-api.txt
  .golden/             snapshot baselines (gitignored)
```

Start API: `uvicorn api.server:app --app-dir backend --port 8000`
