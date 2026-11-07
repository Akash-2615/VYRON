"""Open the visitor DB: local Postgres only.

Env:
  DATABASE_URL / TRACKER_DSN  — required Postgres DSN (password never logged)
  DATABASE_PASSWORD           — optional raw password (URL-encoded into the URI)

No SQLite fallback. If the DSN is missing or the connection fails, raises
``DatabaseError`` so the API / tracker refuse to start. Face crops and
``events.log`` stay on the local filesystem (``storage.log_dir``).
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional, Tuple

from .database import Database

_DSN_PW = re.compile(r"(://[^:/?#\s]+:)([^@/\s]+)(@)")
_PW_QUERY = re.compile(r"([?&](?:password|pass|pwd)=)([^&\s]+)", re.I)


class DatabaseError(RuntimeError):
    """Postgres is required but unavailable or misconfigured."""


def load_env_file() -> None:
    """Load gitignored ``.env`` once (API + CLI + tracker job)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:  # noqa: BLE001
        pass


def redact_secrets(text: str) -> str:
    """Strip passwords from DSNs / connection errors before logging."""
    if not text:
        return text
    s = _DSN_PW.sub(r"\1***\3", str(text))
    s = _PW_QUERY.sub(r"\1***", s)
    s = re.sub(r"(password\s*[=:]\s*)(\S+)", r"\1***", s, flags=re.I)
    return s.replace(" ", "_")


def normalize_database_url(url: str, *, password: str = "") -> str:
    """Normalize a Postgres DSN for psycopg.

    If ``password`` is set (raw, from ``DATABASE_PASSWORD``), it is URL-encoded and
    injected into the URI userinfo so passwords containing ``@`` / ``#`` still work.
    """
    from urllib.parse import quote, urlparse, urlunparse

    u = (url or "").strip().strip('"').strip("'")
    if not u:
        return u
    if u.startswith("postgres://"):
        u = "postgresql://" + u[len("postgres://"):]
    pw = (password or "").strip()
    if pw:
        parts = urlparse(u)
        user = parts.username or "postgres"
        host = parts.hostname or ""
        port = f":{parts.port}" if parts.port else ""
        netloc = f"{quote(user, safe='')}:{quote(pw, safe='')}@{host}{port}"
        u = urlunparse((parts.scheme, netloc, parts.path or "/postgres",
                        parts.params, parts.query, parts.fragment))
    return u


def apply_env_overrides(storage: dict, env: Optional[dict] = None) -> dict:
    """Return a copy of storage with DATABASE_URL applied (always postgres)."""
    if env is None:
        load_env_file()
        env = os.environ
    st = dict(storage)
    url = normalize_database_url(
        env.get("DATABASE_URL") or env.get("TRACKER_DSN") or "",
        password=env.get("DATABASE_PASSWORD") or "",
    )
    if not url:
        # Allow an explicit DSN already on storage (tests / tooling).
        existing = (st.get("dsn") or (st.get("postgres") or {}).get("dsn") or "").strip()
        url = normalize_database_url(existing)
    if not url:
        raise DatabaseError(
            "DATABASE_URL is required (local Postgres). "
            "Set it in .env (e.g. postgresql://facetrack:facetrack@localhost:5432/facetrack). "
            "Face crops and events.log still use the local logs/ folder."
        )
    st["backend"] = "postgres"
    st["postgres"] = {**(st.get("postgres") or {}), "dsn": url}
    st["dsn"] = url
    return st


def open_database(
    storage: dict,
    *,
    reset: bool = False,
    env: Optional[dict] = None,
) -> Tuple[Database, Dict[str, Any]]:
    """Open Postgres. Raises ``DatabaseError`` if the DSN is missing or connect fails.

    Returns ``(db, info)`` with backend=postgres, fallback=False.
    """
    if env is None:
        load_env_file()
        env = os.environ
    try:
        st = apply_env_overrides(storage, env)
    except DatabaseError:
        raise
    try:
        db = Database.from_storage(st, reset=reset)
        db.unique_count()
    except DatabaseError:
        raise
    except Exception as exc:  # noqa: BLE001
        reason = redact_secrets(f"{type(exc).__name__}:{exc}")
        raise DatabaseError(
            f"Cannot connect to Postgres: {reason}. "
            "Fix DATABASE_URL (local Postgres) and retry — the app will not start without the DB."
        ) from exc
    return db, {
        "backend": "postgres",
        "path": None,
        "fallback": False,
        "reason": None,
        "from_backend": None,
    }


def log_db_open(events: Any, info: Dict[str, Any]) -> None:
    """Emit DB_BACKEND. Never includes passwords; never emits DB_FALLBACK."""
    events.log("DB_BACKEND", backend=info.get("backend") or "postgres")
