"""Session handling, log format, away routing, TRACK_STARTED/ENDED pairing."""
from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.database import Database
from src.event_logger import EventLogger
from src.pipeline import FaceTrackingPipeline
from src.session import start_session, stop_session
from tests.test_pipeline import FakeDetector, FakeEmbedder, make_frame


def _cfg(tmp_path: Path) -> dict:
    cfg = load_config("config.json")
    cfg["quality"].update(min_face_px=10, min_blur_var=0)
    cfg["recognition"]["min_embeds_to_register"] = 2
    cfg["away_walkers"] = {"enabled": False}
    cfg["appearance"] = {"enabled": False}
    cfg["timestamps"] = "wall"
    cfg["detection_skip_frames"] = 0
    cfg["tracker"]["max_lost_frames"] = 3
    cfg["logging"] = {"level": "INFO", "separate_away_log": True, "debug_events": False}
    cfg["storage"] = {
        "backend": "sqlite",
        "db_path": str(tmp_path / "t.db"),
        "log_dir": str(tmp_path / "logs"),
        "events_log": str(tmp_path / "logs" / "events.log"),
        "reset_on_start": True,
    }
    cfg["presence"] = {
        "exit_grace_seconds": 0.15,
        "edge_exit_grace_seconds": 0.1,
        "stationary_grace_seconds": 0.15,
        "occlusion_grace_multiplier": 1.0,
        "edge_margin_px": 24,
        "stationary_speed_px_s": 15,
        "resume_spatial_iou": 0.2,
        "resume_threshold_margin": 0.05,
        "reverify_interval_seconds": 100,
        "occlusion_iou": 0.3,
        "mass_loss_ratio": 0.6,
        "max_stream_gap_seconds": 20,
        "exit_zones": [],
    }
    return cfg


def test_entry_exit_log_format_matches_db(tmp_path):
    cfg = _cfg(tmp_path)
    db = Database(str(tmp_path / "t.db"), reset=True)
    ev = EventLogger(str(tmp_path / "logs"), str(tmp_path / "logs" / "events.log"),
                     session_id="testsid_ab12")
    pipe = FaceTrackingPipeline(cfg, FakeDetector(), FakeEmbedder(), db, ev)
    pipe.set_timing(25.0, live=True)
    for i in range(12):
        pipe.process_frame(make_frame(40, None), i, ts=float(i) * 0.04)
    for i in range(12, 40):
        pipe.process_frame(make_frame(None, None), i, ts=float(i) * 0.04)
    pipe.finalize()

    log = (tmp_path / "logs" / "events.log").read_text()
    assert "EXIT_PENDING" not in log
    entry_lines = [ln for ln in log.splitlines() if " | ENTRY" in ln or "| ENTRY " in ln
                   or ln.rstrip().endswith("ENTRY") or " ENTRY " in ln]
    # Formatter: "ENTRY                sid=..."
    entry_lines = [ln for ln in log.splitlines() if "ENTRY" in ln and "sid=" in ln
                   and "FACE_" not in ln and "AWAY_" not in ln]
    exit_lines = [ln for ln in log.splitlines() if " EXIT " in ln or "| EXIT" in ln]
    exit_lines = [ln for ln in log.splitlines()
                  if "EXIT" in ln and "sid=" in ln and "AWAY_" not in ln
                  and "EXIT_PENDING" not in ln and "TRACK_ENDED" not in ln]

    assert entry_lines, log
    assert exit_lines, log
    assert "sid=testsid_ab12" in entry_lines[0]
    assert "ts=" in entry_lines[0] and "visit_id=" in entry_lines[0]
    assert "face_id=" in entry_lines[0] and "track=" in entry_lines[0] and "image=" in entry_lines[0]
    assert "duration_sec=" in exit_lines[0] and "visible_sec=" in exit_lines[0]
    assert "occluded_sec=" in exit_lines[0] and "reason=" in exit_lines[0]
    assert "last_seen=" in exit_lines[0]

    rows = db.query(
        "SELECT event_type, timestamp, visit_id FROM events WHERE event_type IN ('entry','exit') ORDER BY id")
    assert rows
    # Pull ts= from first ENTRY line and match DB.
    fields = dict(p.split("=", 1) for p in entry_lines[0].split() if "=" in p)
    entry_db = next(r for r in rows if (r[0] if not isinstance(r, dict) else r["event_type"]) == "entry")
    db_ts = entry_db[1] if not isinstance(entry_db, dict) else entry_db["timestamp"]
    db_vid = entry_db[2] if not isinstance(entry_db, dict) else entry_db["visit_id"]
    assert fields["ts"] == db_ts
    assert fields["visit_id"] == str(db_vid)
    db.close()


def test_session_archives_on_reset(tmp_path):
    cfg = _cfg(tmp_path)
    log_path = Path(cfg["storage"]["events_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("2026-11-07 12:00:00,000 | INFO    | OLD_LINE            sid=old foo=1\n")
    db = Database(str(tmp_path / "t.db"), reset=True)
    events = start_session(cfg, db, reset=True, sources=["fake.mp4"])
    assert not log_path.read_text().startswith("2026-11-07 12:00:00")
    archive = list((tmp_path / "logs" / "archive").glob("events_*.log"))
    assert archive and "OLD_LINE" in archive[0].read_text()
    text = log_path.read_text()
    assert "SYSTEM_START" in text
    assert f"sid={events.session_id}" in text
    assert "COUNT_SNAPSHOT" in text and "phase=system_start" in text
    db.close()


def test_exactly_one_system_stop_on_exception(tmp_path):
    cfg = _cfg(tmp_path)
    db = Database(str(tmp_path / "t.db"), reset=True)
    events = start_session(cfg, db, reset=True, sources=["a.mp4"])
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        stop_session(db, events, error="boom")
    stop_session(db, events, error="again")  # idempotent
    text = (tmp_path / "logs" / "events.log").read_text()
    assert text.count("SYSTEM_STOP") == 1
    assert "unique_visitors=" in text
    assert "open_visits=" in text
    db.close()


def test_system_stop_on_sigterm_flag(tmp_path):
    """Simulate SIGTERM by stop flag + finally stop_session (same path as main)."""
    cfg = _cfg(tmp_path)
    db = Database(str(tmp_path / "t.db"), reset=True)
    events = start_session(cfg, db, reset=True, sources=["cam0"])
    stop = {"flag": True}
    if stop["flag"]:
        stop_session(db, events, reason="stopped")
    text = (tmp_path / "logs" / "events.log").read_text()
    assert text.count("SYSTEM_STOP") == 1
    assert "session=" in text
    db.close()


def test_track_started_ended_pair(tmp_path):
    cfg = _cfg(tmp_path)
    db = Database(str(tmp_path / "t.db"), reset=True)
    ev = EventLogger(str(tmp_path / "logs"), str(tmp_path / "logs" / "events.log"),
                     debug_events=False)
    # Tentative drop without TRACK_STARTED → no TRACK_ENDED in events.log
    ev.log("TRACK_ENDED", track=99, reason="lost")
    text = (tmp_path / "logs" / "events.log").read_text()
    assert "TRACK_ENDED" not in text
    assert "TRACK_DROPPED" not in text  # debug_events off

    ev.log("TRACK_STARTED", track=7, frame=1)
    ev.log("TRACK_ENDED", track=7, reason="lost")
    text = (tmp_path / "logs" / "events.log").read_text()
    assert text.count("TRACK_STARTED") == 1
    assert text.count("TRACK_ENDED") == 1
    db.close()


def test_away_events_not_in_events_log(tmp_path):
    cfg = _cfg(tmp_path)
    log_dir = tmp_path / "logs"
    ev = EventLogger(str(log_dir), str(log_dir / "events.log"), separate_away_log=True)
    ev.log("ENTRY", face_id=1, track=1, ts="2026-11-07T12:00:00.000", visit_id=1, image="x.jpg")
    ev.log("AWAY_ENTRY", away_id=1, track=2)
    ev.log("AWAY_TRACK_STARTED", track=2)
    main = (log_dir / "events.log").read_text()
    away = (log_dir / "away_events.log").read_text()
    assert "AWAY_ENTRY" not in main
    assert "AWAY_TRACK_STARTED" not in main
    assert "ENTRY" in main and "sid=" in main
    assert "AWAY_ENTRY" in away and "sid=" in away


def test_crash_recovery_closes_open_visits(tmp_path):
    cfg = _cfg(tmp_path)
    db = Database(str(tmp_path / "t.db"), reset=True)
    fid = db.add_face("2026-11-07T12:00:00.000")
    vid = db.start_visit(fid, "2026-11-07T12:00:01.000")
    db.log_event(fid, "entry", "2026-11-07T12:00:01.000", None, 1, 0, visit_id=vid)
    # Advance last_seen without a new entry (e.g. embedding refresh).
    with db._lock:
        db._execute("UPDATE faces SET last_seen=? WHERE face_id=?",
                    ("2026-11-07T12:00:30.000", fid))
        db._conn.commit()
    assert db.open_visits()
    before = db.event_counts()
    start_session(cfg, db, reset=False, sources=["recover.mp4"])
    assert not db.open_visits()
    after = db.event_counts()
    assert after.get("entry", 0) == before.get("entry", 0), "recovery must not invent ENTRY"
    assert after.get("exit", 0) == before.get("exit", 0) + 1
    exits = db.fetch_dicts(
        "SELECT timestamp, note, visit_id FROM events WHERE event_type='exit' ORDER BY id")
    assert exits[-1]["note"] == "crash_recovery"
    assert exits[-1]["timestamp"] == "2026-11-07T12:00:30.000"
    assert exits[-1]["visit_id"] == vid
    text = (tmp_path / "logs" / "events.log").read_text()
    assert "SESSION_RECOVERED" in text
    assert "crash_recovery" in text
    assert " EXIT " in text or "| EXIT" in text
    # Recovery must not emit a log ENTRY for the interrupted visit.
    entry_lines = [
        ln for ln in text.splitlines()
        if "ENTRY" in ln and "sid=" in ln and "FACE_" not in ln and "AWAY_" not in ln
    ]
    assert entry_lines == []
    assert after.get("entry", 0) == 1
    db.close()


def test_crash_recovery_does_not_inflate_entry_exit_counts(tmp_path):
    """Open visit + recovery → +1 exit only; entries unchanged."""
    cfg = _cfg(tmp_path)
    db = Database(str(tmp_path / "t.db"), reset=True)
    f1 = db.add_face("2026-11-07T12:00:00.000")
    f2 = db.add_face("2026-11-07T12:00:05.000")
    v1 = db.start_visit(f1, "2026-11-07T12:00:01.000")
    v2 = db.start_visit(f2, "2026-11-07T12:00:06.000")
    db.log_event(f1, "entry", "2026-11-07T12:00:01.000", None, 1, 0, visit_id=v1)
    db.log_event(f2, "entry", "2026-11-07T12:00:06.000", None, 2, 1, visit_id=v2)
    with db._lock:
        db._execute("UPDATE faces SET last_seen=? WHERE face_id=?",
                    ("2026-11-07T12:00:40.000", f1))
        db._execute("UPDATE faces SET last_seen=? WHERE face_id=?",
                    ("2026-11-07T12:00:41.000", f2))
        db._conn.commit()
    assert db.event_counts() == {"entry": 2}
    start_session(cfg, db, reset=False, sources=["a.mp4"])
    counts = db.event_counts()
    assert counts.get("entry") == 2
    assert counts.get("exit") == 2
    assert db.visit_stats()["open_visits"] == 0
    notes = [r["note"] for r in db.fetch_dicts(
        "SELECT note FROM events WHERE event_type='exit'")]
    assert notes == ["crash_recovery", "crash_recovery"]
    db.close()


def test_api_logs_parses_sid_and_away_source(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from api.server import create_app

    db = Database(str(tmp_path / "t.db"), reset=True)
    db.close()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "events.log").write_text(
        "2026-11-07 12:56:38,123 | INFO    | ENTRY                "
        "sid=20261107T125638_ab12 ts=2026-11-07T12:56:38.123 visit_id=1 "
        "face_id=1 track=3 image=logs/entries/2026-11-07/entry_face0001_125638_123000.jpg\n"
        "2026-11-07 12:56:45,000 | INFO    | EXIT                 "
        "sid=20261107T125638_ab12 ts=2026-11-07T12:56:45.000 visit_id=1 "
        "face_id=1 track=3 image=logs/exits/x.jpg duration_sec=6.877 "
        "visible_sec=5.000 occluded_sec=1.877 reason=hidden_timeout "
        "last_seen=2026-11-07T12:56:45.000\n"
    )
    (log_dir / "away_events.log").write_text(
        "2026-11-07 12:56:40,000 | INFO    | AWAY_ENTRY           "
        "sid=20261107T125638_ab12 away_id=1 track=9\n"
    )
    c = TestClient(create_app(str(tmp_path / "t.db"), str(log_dir), web_dist=None,
                              events_log=str(log_dir / "events.log")))
    out = c.get("/api/logs").json()
    assert out["source"] == "events"
    assert out["total"] == 2
    assert out["items"][0]["event"] == "EXIT"
    assert out["items"][1]["fields"]["sid"] == "20261107T125638_ab12"
    assert out["items"][1]["fields"]["ts"] == "2026-11-07T12:56:38.123"
    assert "AWAY_ENTRY" not in out["counts"]
    away = c.get("/api/logs?source=away").json()
    assert away["source"] == "away" and away["total"] == 1
    assert away["items"][0]["event"] == "AWAY_ENTRY"
