# Data

Local runtime assets (mostly gitignored).

- `videos/` — input clips (was `Video Datasets/`). Place sample MP4s here.
- `models/` — YOLO `.pt` weights (`yolov8n-face.pt`, `yolov8n.pt`, …). Download on first run or copy manually.
- `db/` — optional SQLite files for offline/tests only (`visitors.db`). Production uses Postgres via `DATABASE_URL`.
- `runtime/` — live preview JPEG, annotated videos, live stats (was `output/`).
- `demo/` — fake seed data for UI preview (`python backend/scripts/seed_demo.py`); never mix into real `logs/`.

Do not commit large videos, weights, or DB files. See root `.gitignore`.
