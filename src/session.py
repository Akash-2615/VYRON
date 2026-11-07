"""Session bootstrap: id, log archive on reset, crash-recovery of open visits."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from . import clock as _clock
from .database import Database
from .event_logger import EventLogger, new_session_id


def make_event_logger(cfg: dict, *, session_id: Optional[str] = None) -> EventLogger:
    st = cfg["storage"]
    log_cfg = cfg.get("logging") or {}
    return EventLogger(
        st["log_dir"],
        st["events_log"],
        session_id=session_id or new_session_id(),
        level=str(log_cfg.get("level", "INFO")),
        separate_away_log=bool(log_cfg.get("separate_away_log", True)),
        debug_events=bool(log_cfg.get("debug_events", False)),
    )


def _face_last_seen(db: Database, face_id: int) -> Optional[str]:
    rows = db.query("SELECT last_seen FROM faces WHERE face_id=?", (int(face_id),))
    if not rows:
        return None
    return rows[0][0]


def recover_crashed_session(db: Database, events: EventLogger) -> int:
    """Close open visits left by a previous crash; return how many were closed.

    Writes exactly one EXIT per open visit, stamped at that face's ``last_seen``
    (fallback: visit entry_ts, then wall_now). Does **not** create a synthetic ENTRY.
    """
    open_visits = db.open_visits()
    if not open_visits:
        return 0
    recovered = 0
    for v in open_visits:
        face_id = int(v["face_id"])
        visit_id = int(v["visit_id"])
        # Prefer the face's last real observation — never invent a new sighting time.
        last_seen = (
            _face_last_seen(db, face_id)
            or v.get("entry_ts")
            or _clock.iso_ms()
        )
        db.close_visit(
            visit_id, last_seen,
            float(v.get("visible_seconds") or 0),
            float(v.get("occluded_seconds") or 0),
            int(v.get("resume_count") or 0),
            "crash_recovery",
        )
        # One EXIT row only (no matching ENTRY).
        db.log_event(
            face_id, "exit", last_seen, None, None, None,
            "crash_recovery", visit_id=visit_id)
        events.log(
            "EXIT",
            ts=last_seen, visit_id=visit_id, face_id=face_id,
            reason="crash_recovery", last_seen=last_seen,
        )
        events.log(
            "SESSION_RECOVERED",
            visit_id=visit_id, face_id=face_id,
            last_ts=last_seen, reason="crash_recovery",
        )
        recovered += 1
    return recovered


def archive_session_logs(events: EventLogger) -> List[Path]:
    """Archive events.log and away_events.log into logs/archive/. Return paths moved."""
    moved: List[Path] = []
    for fn in (events.archive_previous, events.archive_away_log):
        dest = fn()
        if dest is not None:
            moved.append(dest)
    return moved


def prepare_fresh(cfg: dict) -> Dict[str, Any]:
    """Archive active logs and wipe the configured visitor DB for a clean run.

    Never touches other schemas, .env, videos, or user files outside storage paths.
    """
    st = dict(cfg["storage"])
    report: Dict[str, Any] = {"archived": [], "db_reset": None, "db_path": None}

    # Logger used only for archive helpers (no SYSTEM_START yet).
    events = make_event_logger(cfg)
    for dest in archive_session_logs(events):
        report["archived"].append(str(dest))

    db_path = st.get("db_path") or "data/visitors.db"
    report["db_path"] = db_path
    # Reset only the active visitors DB for this project (SQLite file or this Postgres DB).
    from .db_bootstrap import open_database
    db, info = open_database(st, reset=True)
    report["db_reset"] = {
        "backend": db.backend,
        "path": info.get("path") or db_path,
        "unique_after": db.unique_count(),
    }
    db.close()
    return report


def start_session(
    cfg: dict,
    db: Database,
    *,
    reset: bool,
    sources: list,
    extra_fields: Optional[dict] = None,
) -> EventLogger:
    """Create EventLogger, optionally archive prior log, recover crashes, log SYSTEM_START."""
    events = make_event_logger(cfg)
    if reset:
        for dest in archive_session_logs(events):
            events.log("LOG_ARCHIVED", path=str(dest))

    # When DB was NOT reset, close leftover open visits from a crashed prior run.
    if not reset:
        recover_crashed_session(db, events)

    features = []
    if cfg.get("away_walkers", {}).get("enabled"):
        features.append("away_walkers")
    if cfg.get("appearance", {}).get("enabled"):
        features.append("appearance")
    if (cfg.get("retina_face") or {}).get("enabled", True):
        features.append("retina_face")
    fields: dict[str, Any] = {
        "session": events.session_id,
        "sources": len(sources),
        "source_list": ",".join(str(s).replace(" ", "_") for s in sources[:8]),
        "detection_skip_frames": cfg.get("detection_skip_frames"),
        "match_threshold": (cfg.get("recognition") or {}).get("match_threshold"),
        "reset_on_start": reset,
        "features": ",".join(features) if features else "none",
        "log_level": (cfg.get("logging") or {}).get("level", "INFO"),
    }
    if extra_fields:
        fields.update(extra_fields)
    events.log("SYSTEM_START", **fields)
    return events


def stop_session(db: Database, events: EventLogger, **extra) -> None:
    """Emit exactly one SYSTEM_STOP per EventLogger (idempotent)."""
    if getattr(events, "_system_stopped", False):
        return
    counts = db.event_counts()
    visit_stats = db.visit_stats()
    away = {}
    try:
        away = db.away_stats()
    except Exception:  # noqa: BLE001
        pass
    # Reserved KPI fields win over **extra (avoids duplicate kwargs from callers).
    reserved = {
        "session": events.session_id,
        "unique_visitors": db.unique_count(),
        "entries": counts.get("entry", 0),
        "exits": counts.get("exit", 0),
        "open_visits": visit_stats.get("open_visits", 0),
        "away_unique": away.get("away_unique", 0),
        "away_summary": events.away_summary().replace(" ", ","),
    }
    fields = {**extra, **reserved}
    events.log("SYSTEM_STOP", **fields)
    events.log(
        "COUNT_SNAPSHOT",
        unique_total=db.unique_count(),
        entries=counts.get("entry", 0),
        exits=counts.get("exit", 0),
        open_visits=visit_stats.get("open_visits", 0),
        phase="system_stop",
    )
    events._system_stopped = True
