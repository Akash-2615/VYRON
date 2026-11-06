"""Away-walker DB + filtering helpers (no YOLO/CLIP required)."""
from src.database import Database


def test_away_visitor_stats(tmp_path):
    db = Database(str(tmp_path / "a.db"), reset=True)
    aid = db.add_away_visitor("2026-11-06T10:00:00.000")
    db.set_away_appearance(aid, "black jacket, jeans", '{"summary":"black jacket, jeans"}', None)
    db.log_away_event(aid, "entry", "2026-11-06T10:00:01.000", None, 1, 0, "away")
    db.log_away_event(aid, "exit", "2026-11-06T10:00:20.000", None, 1, 10, "lost")
    s = db.away_stats()
    assert s["away_unique"] == 1
    assert s["away_entries"] == 1 and s["away_exits"] == 1
    assert s["away_currently"] == 0
    assert s["away_with_clothes"] == 1
    db.close()
