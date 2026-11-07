"""DB ↔ log count reconciliation."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.counting import count_from_db, count_from_log, reconcile, verify_log_invariants
from src.database import Database
from tests.test_pipeline import run as pipeline_run


def _line(when: str, event: str, **fields) -> str:
    extras = " ".join(f"{k}={v}" for k, v in fields.items())
    return f"{when} | INFO    | {event:<20} {extras}\n"


def test_pipeline_db_and_log_agree(tmp_path):
    db = pipeline_run(tmp_path, skip=0)
    log_path = tmp_path / "logs" / "events.log"
    # Ensure FACE_REGISTERED lines exist (pipeline writes them).
    text = log_path.read_text()
    assert "FACE_REGISTERED" in text
    db_c = count_from_db(tmp_path / "t.db")
    # Attach SYSTEM_START/STOP + sid so session parsing works; pipeline EventLogger
    # already stamped sid= on every line.
    log_c = count_from_log(log_path, session="latest")
    assert db_c.available and log_c.available
    assert db_c.unique_visitors == 2
    assert db_c.entries == 3 and db_c.exits == 3
    assert log_c.unique_visitors == 2
    assert log_c.entries == 3 and log_c.exits == 3
    result = reconcile(db_c, log_c)
    assert result["consistent"] is True
    db.close()


def test_rerun_session_without_new_registrations_is_consistent(tmp_path):
    """Re-playing video: 0 new FACE_REGISTERED but COUNT_SNAPSHOT matches DB gallery."""
    from src.counting import Counts

    db_c = Counts(unique_visitors=30, entries=62, exits=62, available=True)
    log_c = Counts(
        unique_visitors=0, entries=29, exits=29, available=True,
        last_unique_total=30,
        last_count_snapshot={"unique_total": 30, "entries": 62, "exits": 62},
        system_stop_unique=30,
    )
    result = reconcile(db_c, log_c)
    assert result["consistent"] is True
    assert any("re-run" in n for n in result["notes"])


def test_idle_latest_session_inherits_prior_snapshot(tmp_path):
    """Dashboard restart with empty latest session must not report Log differs by N."""
    log = tmp_path / "events.log"
    log.write_text(
        _line("2026-11-07 23:42:31,000", "SYSTEM_START", sid="sessA", session="sessA")
        + _line("2026-11-07 23:42:40,000", "FACE_REGISTERED", sid="sessA", face_id=1,
                unique_total=1, track=1)
        + _line("2026-11-07 23:42:41,000", "FACE_REGISTERED", sid="sessA", face_id=2,
                unique_total=2, track=2)
        + _line("2026-11-07 23:42:42,000", "FACE_REGISTERED", sid="sessA", face_id=3,
                unique_total=3, track=3)
        + _line("2026-11-07 23:42:40,500", "ENTRY", sid="sessA", face_id=1, visit_id=1, track=1)
        + _line("2026-11-07 23:42:41,500", "ENTRY", sid="sessA", face_id=2, visit_id=2, track=2)
        + _line("2026-11-07 23:42:42,500", "ENTRY", sid="sessA", face_id=3, visit_id=3, track=3)
        + _line("2026-11-07 23:42:46,000", "EXIT", sid="sessA", face_id=1, visit_id=1, track=1)
        + _line("2026-11-07 23:42:46,100", "EXIT", sid="sessA", face_id=2, visit_id=2, track=2)
        + _line("2026-11-07 23:42:46,200", "EXIT", sid="sessA", face_id=3, visit_id=3, track=3)
        + _line("2026-11-07 23:42:46,800", "SYSTEM_STOP", sid="sessA", session="sessA",
                unique_visitors=3, entries=3, exits=3, open_visits=0)
        + _line("2026-11-07 23:42:46,801", "COUNT_SNAPSHOT", sid="sessA",
                unique_total=3, entries=3, exits=3, open_visits=0, phase="system_stop")
        # Idle restart — no FACE_REGISTERED / ENTRY yet.
        + _line("2026-11-07 23:43:01,000", "SYSTEM_START", sid="sessB", session="sessB")
    )
    latest = count_from_log(log, session="latest")
    assert latest.session_id == "sessB"
    assert latest.unique_visitors == 0  # no new registrations this session
    assert latest.last_unique_total == 3
    assert latest.last_count_snapshot and latest.last_count_snapshot["unique_total"] == 3
    assert any("idle" in n for n in latest.notes)

    from src.counting import Counts
    db_c = Counts(unique_visitors=3, entries=3, exits=3, available=True)
    result = reconcile(db_c, latest)
    assert result["consistent"] is True


def test_deleted_face_registered_is_mismatch(tmp_path):

    db = pipeline_run(tmp_path, skip=0)
    log_path = tmp_path / "logs" / "events.log"
    lines = log_path.read_text().splitlines(keepends=True)
    # Drop one FACE_REGISTERED (face_id=2) so log unique becomes 1.
    kept = []
    dropped = False
    for ln in lines:
        if (not dropped) and "FACE_REGISTERED" in ln and "face_id=2" in ln:
            dropped = True
            continue
        kept.append(ln)
    assert dropped
    log_path.write_text("".join(kept))
    db_c = count_from_db(tmp_path / "t.db")
    log_c = count_from_log(log_path)
    result = reconcile(db_c, log_c)
    assert result["consistent"] is False
    assert any("unique_visitors" in d for d in result["diffs"])

    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "count.py"), "--source", "both",
         "--db", str(tmp_path / "t.db"), "--log", str(log_path)],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True, text=True,
    )
    assert proc.returncode == 1
    db.close()


def test_two_sessions_latest_all_and_rotated(tmp_path):
    log = tmp_path / "events.log"
    rotated = tmp_path / "events.log.1"
    archive = tmp_path / "archive"
    archive.mkdir()
    # Older rotated session A
    rotated.write_text(
        _line("2026-11-07 10:00:00,000", "SYSTEM_START", sid="sessA", session="sessA")
        + _line("2026-11-07 10:00:01,000", "FACE_REGISTERED", sid="sessA", face_id=1,
                unique_total=1, track=1)
        + _line("2026-11-07 10:00:02,000", "ENTRY", sid="sessA", face_id=1, visit_id=1, track=1,
                ts="2026-11-07T10:00:02.000", image="x.jpg")
        + _line("2026-11-07 10:00:10,000", "EXIT", sid="sessA", face_id=1, visit_id=1, track=1,
                ts="2026-11-07T10:00:10.000", image="y.jpg", duration_sec=8,
                visible_sec=8, occluded_sec=0, reason="hidden_timeout",
                last_seen="2026-11-07T10:00:10.000")
        + _line("2026-11-07 10:01:00,000", "SYSTEM_STOP", sid="sessA", session="sessA",
                unique_visitors=1, entries=1, exits=1, open_visits=0)
    )
    # Current log: session B with 2 faces
    log.write_text(
        _line("2026-11-07 12:00:00,000", "SYSTEM_START", sid="sessB", session="sessB")
        + _line("2026-11-07 12:00:01,000", "FACE_REGISTERED", sid="sessB", face_id=10,
                unique_total=1, track=2)
        + _line("2026-11-07 12:00:02,000", "FACE_REGISTERED", sid="sessB", face_id=11,
                unique_total=2, track=3)
        + _line("2026-11-07 12:00:03,000", "ENTRY", sid="sessB", face_id=10, visit_id=1, track=2,
                ts="2026-11-07T12:00:03.000", image="a.jpg")
        + _line("2026-11-07 12:00:04,000", "ENTRY", sid="sessB", face_id=11, visit_id=2, track=3,
                ts="2026-11-07T12:00:04.000", image="b.jpg")
        + _line("2026-11-07 12:00:20,000", "EXIT", sid="sessB", face_id=10, visit_id=1, track=2,
                ts="2026-11-07T12:00:20.000", image="c.jpg", duration_sec=17,
                visible_sec=17, occluded_sec=0, reason="hidden_timeout",
                last_seen="2026-11-07T12:00:20.000")
        + _line("2026-11-07 12:00:21,000", "EXIT", sid="sessB", face_id=11, visit_id=2, track=3,
                ts="2026-11-07T12:00:21.000", image="d.jpg", duration_sec=17,
                visible_sec=17, occluded_sec=0, reason="hidden_timeout",
                last_seen="2026-11-07T12:00:21.000")
        + _line("2026-11-07 12:05:00,000", "SYSTEM_STOP", sid="sessB", session="sessB",
                unique_visitors=2, entries=2, exits=2, open_visits=0)
        + "this is garbage not a log line\n"
    )
    latest = count_from_log(log, session="latest")
    assert latest.session_id == "sessB"
    assert latest.unique_visitors == 2
    assert latest.entries == 2 and latest.exits == 2
    all_c = count_from_log(log, session="all", include_rotated=True)
    assert all_c.unique_visitors == 3  # 1 from A + 2 from B
    assert all_c.entries == 3
    only_a = count_from_log(log, session="sessA", include_rotated=True)
    assert only_a.unique_visitors == 1


def test_db_only_and_log_only(tmp_path):
    db = Database(str(tmp_path / "only.db"), reset=True)
    fid = db.add_face("2026-11-07T12:00:00.000")
    db.log_event(fid, "entry", "2026-11-07T12:00:01.000", None, 1, 0)
    db.log_event(fid, "exit", "2026-11-07T12:00:10.000", None, 1, 1)
    db_c = count_from_db(tmp_path / "only.db")
    assert db_c.available and db_c.unique_visitors == 1 and db_c.entries == 1
    missing_log = count_from_log(tmp_path / "nope.log")
    assert missing_log.available is False
    r = reconcile(db_c, missing_log)
    assert r["consistent"] is None
    assert "log_unavailable" in r["notes"] or "log_unavailable" in missing_log.warnings

    log = tmp_path / "solo.log"
    log.write_text(
        _line("2026-11-07 12:00:00,000", "SYSTEM_START", sid="solo", session="solo")
        + _line("2026-11-07 12:00:01,000", "FACE_REGISTERED", sid="solo", face_id=5,
                unique_total=1, track=1)
        + "@@@ not parseable @@@\n"
        + _line("2026-11-07 12:00:02,000", "ENTRY", sid="solo", face_id=5, visit_id=1, track=1,
                ts="2026-11-07T12:00:02.000", image="x.jpg")
        + _line("2026-11-07 12:00:09,000", "EXIT", sid="solo", face_id=5, visit_id=1, track=1,
                ts="2026-11-07T12:00:09.000", image="y.jpg", duration_sec=7,
                visible_sec=7, occluded_sec=0, reason="hidden_timeout",
                last_seen="2026-11-07T12:00:09.000")
        + _line("2026-11-07 12:01:00,000", "SYSTEM_STOP", sid="solo", session="solo",
                unique_visitors=1, entries=1, exits=1, open_visits=0)
    )
    log_c = count_from_log(log)
    assert log_c.available and log_c.unique_visitors == 1
    r2 = reconcile(count_from_db(tmp_path / "missing.db"), log_c)
    assert r2["consistent"] is None
    db.close()


def test_verify_log_missing_exit_and_duplicate_register(tmp_path):
    log = tmp_path / "bad.log"
    log.write_text(
        _line("2026-11-07 12:00:00,000", "SYSTEM_START", sid="bad", session="bad")
        + _line("2026-11-07 12:00:01,000", "FACE_REGISTERED", sid="bad", face_id=1,
                unique_total=1, track=1)
        + _line("2026-11-07 12:00:01,500", "FACE_REGISTERED", sid="bad", face_id=1,
                unique_total=1, track=2)
        + _line("2026-11-07 12:00:02,000", "ENTRY", sid="bad", face_id=1, visit_id=9, track=1,
                ts="2026-11-07T12:00:02.000", image="x.jpg")
        + _line("2026-11-07 12:01:00,000", "SYSTEM_STOP", sid="bad", session="bad",
                unique_visitors=1, entries=1, exits=0, open_visits=1)
    )
    violations = verify_log_invariants(str(log), session="latest")
    assert any("missing EXIT" in v for v in violations)
    assert any("duplicate_FACE_REGISTERED" in v for v in violations)

    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "verify_log.py"), "--log", str(log), "--session", "latest"],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True, text=True,
    )
    assert proc.returncode == 1


def test_api_count_consistent_and_mismatch(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from api.server import create_app

    db = Database(str(tmp_path / "t.db"), reset=True)
    fid = db.add_face("2026-11-07T12:00:00.000")
    vid = db.start_visit(fid, "2026-11-07T12:00:01.000")
    db.log_event(fid, "entry", "2026-11-07T12:00:01.000", None, 1, 0, visit_id=vid)
    db.close_visit(vid, "2026-11-07T12:00:10.000", 9.0, 0.0, 0, "hidden_timeout")
    db.log_event(fid, "exit", "2026-11-07T12:00:10.000", None, 1, 1, visit_id=vid)
    db.close()

    logs = tmp_path / "logs"
    logs.mkdir()
    good = logs / "events.log"
    good.write_text(
        _line("2026-11-07 12:00:00,000", "SYSTEM_START", sid="api1", session="api1")
        + _line("2026-11-07 12:00:00,500", "FACE_REGISTERED", sid="api1", face_id=1,
                unique_total=1, track=1)
        + _line("2026-11-07 12:00:01,000", "ENTRY", sid="api1", face_id=1, visit_id=1, track=1,
                ts="2026-11-07T12:00:01.000", image="x.jpg")
        + _line("2026-11-07 12:00:10,000", "EXIT", sid="api1", face_id=1, visit_id=1, track=1,
                ts="2026-11-07T12:00:10.000", image="y.jpg", duration_sec=9,
                visible_sec=9, occluded_sec=0, reason="hidden_timeout",
                last_seen="2026-11-07T12:00:10.000")
        + _line("2026-11-07 12:01:00,000", "SYSTEM_STOP", sid="api1", session="api1",
                unique_visitors=1, entries=1, exits=1, open_visits=0)
    )
    c = TestClient(create_app(str(tmp_path / "t.db"), str(logs), web_dist=None,
                              events_log=str(good)))
    out = c.get("/api/count?session=latest").json()
    assert out["source"] == "db"
    assert out["unique_visitors"] == 1
    assert out["consistent"] is True
    assert out["derived_from_log"]["unique_visitors"] == 1

    # Mismatch: wipe FACE_REGISTERED from log
    good.write_text(
        _line("2026-11-07 12:00:00,000", "SYSTEM_START", sid="api2", session="api2")
        + _line("2026-11-07 12:00:01,000", "ENTRY", sid="api2", face_id=1, visit_id=1, track=1,
                ts="2026-11-07T12:00:01.000", image="x.jpg")
        + _line("2026-11-07 12:00:10,000", "EXIT", sid="api2", face_id=1, visit_id=1, track=1,
                ts="2026-11-07T12:00:10.000", image="y.jpg", duration_sec=9,
                visible_sec=9, occluded_sec=0, reason="hidden_timeout",
                last_seen="2026-11-07T12:00:10.000")
        + _line("2026-11-07 12:01:00,000", "SYSTEM_STOP", sid="api2", session="api2",
                unique_visitors=0, entries=1, exits=1, open_visits=0)
    )
    # Bust 2s cache by using a different session query then latest after rewrite —
    # cache key is session string; force via session=api2
    bad = c.get("/api/count?session=api2").json()
    assert bad["consistent"] is False
    assert any("unique_visitors" in d for d in bad["diffs"])
