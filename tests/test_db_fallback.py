"""Postgres-only bootstrap: hard fail without DSN / on connect error; secret redaction."""
from __future__ import annotations

import pytest

from src.db_bootstrap import (
    DatabaseError, apply_env_overrides, log_db_open, normalize_database_url,
    open_database, redact_secrets,
)
from src.event_logger import EventLogger


def test_redact_secrets_strips_password():
    dsn = "postgresql://facetrack:s3cretPASS@localhost:5432/facetrack"
    out = redact_secrets(dsn)
    assert "s3cretPASS" not in out
    assert "***" in out
    assert "facetrack" in out  # user kept
    err = "connection failed postgresql://u:hunter2@host/db password=hunter2"
    assert "hunter2" not in redact_secrets(err)


def test_missing_database_url_raises():
    storage = {"backend": "postgres", "postgres": {}, "log_dir": "logs"}
    with pytest.raises(DatabaseError, match="DATABASE_URL"):
        open_database(storage, reset=False, env={})


def test_bad_database_url_raises_no_sqlite(tmp_path):
    storage = {
        "backend": "postgres",
        "postgres": {},
        "log_dir": str(tmp_path / "logs"),
        "events_log": str(tmp_path / "logs" / "events.log"),
    }
    env = {
        "DATABASE_URL": "postgresql://facetrack:SuperSecret99@127.0.0.1:1/nope",
    }
    with pytest.raises(DatabaseError, match="Cannot connect") as ei:
        open_database(storage, reset=True, env=env)
    assert "SuperSecret99" not in str(ei.value)
    assert not (tmp_path / "fallback.db").exists()
    assert not list(tmp_path.glob("*.db"))


def test_storage_dsn_without_env_still_opens_path(tmp_path):
    """apply_env_overrides may use storage.postgres.dsn when env is empty (tests)."""
    st = apply_env_overrides(
        {"backend": "sqlite", "postgres": {"dsn": "postgresql://u:p@h/db"}},
        {},
    )
    assert st["backend"] == "postgres"
    assert st["dsn"].startswith("postgresql://")


def test_apply_env_overrides_database_url():
    st = apply_env_overrides(
        {"backend": "sqlite", "db_path": "x.db", "postgres": {}},
        {"DATABASE_URL": "postgresql://u:p@h/db"},
    )
    assert st["backend"] == "postgres"
    assert st["postgres"]["dsn"].startswith("postgresql://")


def test_normalize_postgres_scheme():
    raw = "postgres://facetrack:secret@localhost:5432/facetrack"
    out = normalize_database_url(raw)
    assert out.startswith("postgresql://")
    assert "localhost" in out
    assert "sslmode=" not in out


def test_log_db_open_postgres_only(tmp_path):
    ev = EventLogger(str(tmp_path / "logs"), str(tmp_path / "logs" / "events.log"),
                     session_id="pgonly_aa01")
    log_db_open(ev, {"backend": "postgres", "fallback": False})
    text = (tmp_path / "logs" / "events.log").read_text()
    assert "DB_BACKEND" in text and "backend=postgres" in text
    assert "DB_FALLBACK" not in text


def test_api_health_postgres_no_fallback(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from api.server import create_app
    from src.database import Database

    # Unit tests inject SQLite; runtime open_database never does.
    db = Database(path=str(tmp_path / "unit.db"), backend="sqlite", reset=True)
    app = create_app(db=db, log_dir=str(tmp_path / "logs"), web_dist=None)
    app.state.db_bootstrap = {"backend": "postgres", "fallback": False, "path": None}
    c = TestClient(app)
    h = c.get("/api/health").json()
    assert h["ok"] is True
    assert h["fallback"] is False
    db.close()


def test_secs_accepts_space_separated_and_datetime():
    pytest.importorskip("fastapi")
    from datetime import datetime
    from api.server import _secs
    from api.services import secs
    assert abs(_secs("2026-11-07T12:00:00") - _secs("2026-11-07 12:00:00")) < 1e-6
    assert abs(secs(datetime(2026, 11, 7, 12, 0, 0)) - secs("2026-11-07T12:00:00")) < 1e-6
