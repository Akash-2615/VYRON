"""HTTP route map for the dashboard API.

Handlers stay in ``api.server.create_app`` (they close over DB / log paths).
This package is the documented index of those routes — importing it has no
side effects and adds no request overhead.
"""

# path -> short description (keep in sync when adding endpoints in server.py)
ROUTE_MAP = {
    "GET /api/health": "DB connectivity",
    "GET /api/videos": "Local video catalogue for the Start picker",
    "GET /api/tracker/status": "Background tracker job state",
    "POST /api/tracker/start": "Start file / RTSP / webcam job",
    "POST /api/tracker/stop": "Stop tracker job",
    "GET /api/live/status": "Live preview metadata",
    "GET /api/live.jpg": "Latest annotated JPEG",
    "GET /api/live": "Live preview alias",
    "GET /api/count": "Unique / entry / exit audit",
    "GET /api/summary": "Dashboard KPI snapshot",
    "GET /api/away-visitors": "No-face walker list",
    "GET /api/away-visitors/{away_id}": "No-face walker detail",
    "GET /api/timeline": "Recent activity feed",
    "GET /api/visitors": "Registered face gallery",
    "GET /api/visitors/{face_id}": "Visitor detail",
    "GET /api/events": "Entry/exit event log",
    "GET /api/visitors/{face_id}/embeddings": "Stored ArcFace vectors",
    "GET /api/logs": "Tail of events.log",
    "GET /api/preprocess": "Preprocess settings + metrics + step catalog",
    "PUT /api/preprocess": "Write runtime preprocess override (settings file only)",
    "POST /api/preprocess/reset": "Remove preprocess override",
    "GET /api/preprocess/preview": "Before/after preprocess JPEG",
}

__all__ = ["ROUTE_MAP"]
