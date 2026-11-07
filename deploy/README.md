# Deploy / local infra

- `docker-compose.yml` — local Postgres (`facetrack` DB) for the tracker API.
- `mediamtx.yml` — MediaMTX config for RTSP loopback tests (gitignored if present locally).
- `*.crt` / `*.key` — TLS material for MediaMTX if needed (gitignored; never commit).

Start DB: `docker compose -f deploy/docker-compose.yml up -d db`

RTSP helper: `python backend/scripts/run_rtsp.py "rtsp://…"` (or `make rtsp`).
