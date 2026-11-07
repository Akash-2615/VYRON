# API

FastAPI analytics + dashboard backend. Package name stays `api` (import as `api.server`).

```
api/
  server.py           create_app, HTTP routes, static/media mounts
  services.py         AnalyticsService (shared by routes + assistant)
  assistant.py        Groq SSE chat router
  assistant_tools.py  read-only tool handlers
  assistant_prompt.py system prompt + labels
```

Start from repo root:

```bash
uvicorn api.server:app --app-dir backend --port 8000
```

Serves `frontend/dist` when built; media from `logs/`; live preview from `data/runtime/live.jpg`.
