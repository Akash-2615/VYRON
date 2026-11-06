import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from src.database import Database  # noqa: E402
from api.server import create_app  # noqa: E402


def test_api_summary_and_visitors(tmp_path):
    db = Database(str(tmp_path / "t.db"), reset=True)
    f = db.add_face("2026-11-06T10:00:00.000")
    db.log_event(f, "entry", "2026-11-06T10:00:05.000", None, 1, 0)
    db.log_event(f, "exit", "2026-11-06T10:00:35.000", None, 1, 1)
    db.log_event(f, "entry", "2026-11-06T10:02:00.000", None, 2, 2)
    db.close()
    c = TestClient(create_app(str(tmp_path / "t.db"), str(tmp_path / "logs"), web_dist=None,
                              live_preview=str(tmp_path / "missing.jpg")))
    s = c.get("/api/summary").json()
    assert s["unique_visitors"] == 1 and s["total_entries"] == 2 and s["total_exits"] == 1
    assert s["currently_inside"] == 1 and s["returning_visitors"] == 1 and s["avg_dwell_seconds"] == 30.0
    assert c.get("/api/visitors").json()[0]["status"] == "inside"
    assert c.get("/api/visitors/1").json()["events"][0]["event_type"] == "entry"
    assert c.get("/api/visitors/99").status_code == 404
    assert c.get("/api/events?type=exit").json()["total"] == 1
    assert len(c.get("/api/timeline").json()["points"]) >= 2
    live = c.get("/api/live/status").json()
    assert live["active"] is False
    assert c.get("/api/live.jpg").status_code == 404


def test_api_embeddings_and_logs(tmp_path):
    import numpy as np
    db = Database(str(tmp_path / "t.db"), reset=True)
    rng = np.random.default_rng(0)
    for _ in range(2):
        fid = db.add_face("2026-11-06T10:00:00.000")
        v = rng.normal(size=512).astype(np.float32)
        v /= np.linalg.norm(v)
        db.add_embedding(fid, v, "2026-11-06T10:00:00.000", None)
    db.close()
    log = tmp_path / "events.log"
    log.write_text(
        "2026-11-06 10:00:01,123 | INFO    | ENTRY                face_id=1 track=3 note=new\n"
        "2026-11-06 10:00:02,456 | WARNING | DUPLICATE_TRACK      face_id=1 track=4\n"
        "continuation line that is not a log record\n")
    c = TestClient(create_app(str(tmp_path / "t.db"), str(tmp_path / "logs"), web_dist=None,
                              events_log=str(log)))
    e = c.get("/api/visitors/1/embeddings").json()
    assert e["dim"] == 512 and e["count"] == 1 and len(e["samples"][0]["vector"]) == 512
    assert e["similar"][0]["face_id"] == 2 and abs(e["samples"][0]["norm"] - 1) < 1e-3
    assert c.get("/api/visitors/9/embeddings").status_code == 404
    out = c.get("/api/logs").json()
    assert out["total"] == 2 and out["items"][0]["event"] == "DUPLICATE_TRACK"   # newest first
    assert out["items"][1]["fields"]["track"] == "3"
    assert c.get("/api/logs?level=WARNING").json()["total"] == 1
    assert c.get("/api/logs?event=ENTRY").json()["total"] == 1
    assert c.get("/api/logs?q=duplicate").json()["total"] == 1
