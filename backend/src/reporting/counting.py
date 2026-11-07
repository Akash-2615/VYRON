"""Derive unique visitor counts from the DB and/or events.log and reconcile them.

Unique visitors = rows in `faces` = distinct face_id on FACE_REGISTERED lines
(for one session). FACE_RECOGNIZED / re-entries never change it. Away walkers
are not part of it.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

LOG_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),\d+ \| (\w+)\s*\| (\S+)\s*(.*)$"
)
FIELD_RE = re.compile(r"(\w+)=(\S+)")


@dataclass
class Counts:
    unique_visitors: int = 0
    entries: int = 0
    exits: int = 0
    returning_visitors: int = 0
    open_visits: int = 0
    session_id: Optional[str] = None
    available: bool = True
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    # Cross-check hints from the log itself (not used for the primary unique).
    last_unique_total: Optional[int] = None
    last_count_snapshot: Optional[Dict[str, int]] = None
    system_stop_unique: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


def parse_log_line(line: str) -> Optional[dict]:
    """Tolerant parser matching the API/dashboard (ignore garbage)."""
    m = LOG_RE.match(line.rstrip("\n"))
    if not m:
        return None
    day, clock, level, event, rest = m.groups()
    fields = dict(FIELD_RE.findall(rest))
    return {
        "ts": f"{day}T{clock}",
        "day": day,
        "clock": clock,
        "level": level,
        "event": event,
        "message": rest,
        "fields": fields,
        "sid": fields.get("sid") or fields.get("session"),
    }


def _iter_log_paths(log_path: Union[str, Path], include_rotated: bool) -> List[Path]:
    primary = Path(log_path)
    paths: List[Path] = []
    if include_rotated:
        archive = primary.parent / "archive"
        if archive.is_dir():
            paths.extend(sorted(archive.glob("events_*.log")))
        # RotatingFileHandler backups: events.log.1 (older) … higher N older still.
        rotated = sorted(
            primary.parent.glob(primary.name + ".*"),
            key=lambda p: _rotate_index(p, primary),
            reverse=True,  # oldest numbered first
        )
        paths.extend(p for p in rotated if p.is_file())
    if primary.exists():
        paths.append(primary)
    return paths


def _rotate_index(path: Path, primary: Path) -> int:
    suffix = path.name[len(primary.name):]
    if suffix.startswith("."):
        try:
            return int(suffix[1:])
        except ValueError:
            return 0
    return 0


def _read_lines(paths: Sequence[Path]) -> Iterable[Tuple[Path, str]]:
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            yield path, line


def _session_of(rec: dict, current: Optional[str]) -> Optional[str]:
    sid = rec.get("sid")
    if rec["event"] == "SYSTEM_START":
        return rec["fields"].get("session") or sid or current
    return sid or current


@dataclass
class _SessionAcc:
    session_id: str
    face_ids: List[int] = field(default_factory=list)  # registration order
    face_set: set = field(default_factory=set)
    entries: int = 0
    exits: int = 0
    entry_faces: Dict[int, int] = field(default_factory=dict)  # face_id -> entry count
    entry_visits: set = field(default_factory=set)
    exit_visits: set = field(default_factory=set)
    track_started: set = field(default_factory=set)
    track_ended: set = field(default_factory=set)
    has_start: bool = False
    has_stop: bool = False
    start_ts: Optional[str] = None
    stop_ts: Optional[str] = None
    last_unique_total: Optional[int] = None
    last_snapshot: Optional[Dict[str, int]] = None
    system_stop_unique: Optional[int] = None
    system_stop_entries: Optional[int] = None
    system_stop_exits: Optional[int] = None
    warnings: List[str] = field(default_factory=list)


def _int_field(fields: dict, key: str) -> Optional[int]:
    v = fields.get(key)
    if v is None:
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _accumulate(sessions: Dict[str, _SessionAcc], order: List[str], rec: dict) -> None:
    fields = rec["fields"]
    event = rec["event"]
    sid = rec.get("sid") or fields.get("session")
    if event == "SYSTEM_START":
        sid = fields.get("session") or sid or f"anon_{len(order)}"
        if sid not in sessions:
            sessions[sid] = _SessionAcc(session_id=sid)
            order.append(sid)
        acc = sessions[sid]
        acc.has_start = True
        acc.start_ts = rec["ts"]
        return
    if not sid:
        # Orphan line without sid — attach to latest session if any.
        if not order:
            sid = "_nosession"
            sessions[sid] = _SessionAcc(session_id=sid)
            order.append(sid)
        else:
            sid = order[-1]
    if sid not in sessions:
        sessions[sid] = _SessionAcc(session_id=sid)
        order.append(sid)
    acc = sessions[sid]

    if event == "SYSTEM_STOP":
        acc.has_stop = True
        acc.stop_ts = rec["ts"]
        acc.system_stop_unique = _int_field(fields, "unique_visitors")
        acc.system_stop_entries = _int_field(fields, "entries")
        acc.system_stop_exits = _int_field(fields, "exits")
    elif event == "FACE_REGISTERED":
        fid = _int_field(fields, "face_id")
        if fid is None:
            acc.warnings.append("FACE_REGISTERED_missing_face_id")
        elif fid in acc.face_set:
            acc.warnings.append(f"duplicate_FACE_REGISTERED_face_id={fid}")
        else:
            acc.face_set.add(fid)
            acc.face_ids.append(fid)
        ut = _int_field(fields, "unique_total")
        if ut is not None:
            acc.last_unique_total = ut
    elif event == "ENTRY":
        acc.entries += 1
        fid = _int_field(fields, "face_id")
        if fid is not None:
            acc.entry_faces[fid] = acc.entry_faces.get(fid, 0) + 1
        vid = _int_field(fields, "visit_id")
        if vid is not None:
            acc.entry_visits.add(vid)
    elif event == "EXIT":
        acc.exits += 1
        vid = _int_field(fields, "visit_id")
        if vid is not None:
            if vid not in acc.entry_visits:
                acc.warnings.append(f"EXIT_without_ENTRY_visit_id={vid}")
            acc.exit_visits.add(vid)
    elif event == "COUNT_SNAPSHOT":
        snap = {
            "unique_total": _int_field(fields, "unique_total") or 0,
            "entries": _int_field(fields, "entries") or 0,
            "exits": _int_field(fields, "exits") or 0,
        }
        acc.last_snapshot = snap
        if snap["unique_total"]:
            acc.last_unique_total = snap["unique_total"]
    elif event == "TRACK_STARTED":
        tid = _int_field(fields, "track")
        if tid is not None:
            acc.track_started.add(tid)
    elif event == "TRACK_ENDED":
        tid = _int_field(fields, "track")
        if tid is not None:
            acc.track_ended.add(tid)


def _counts_from_acc(acc: _SessionAcc) -> Counts:
    warnings = list(acc.warnings)
    if not acc.has_start and acc.session_id != "_nosession":
        warnings.append("missing_SYSTEM_START")
    if acc.has_start and not acc.has_stop:
        warnings.append("missing_SYSTEM_STOP")
    for vid in acc.exit_visits:
        if vid not in acc.entry_visits:
            # already warned on EXIT
            pass
    returning = sum(1 for n in acc.entry_faces.values() if n > 1)
    return Counts(
        unique_visitors=len(acc.face_set),
        entries=acc.entries,
        exits=acc.exits,
        returning_visitors=returning,
        open_visits=max(0, acc.entries - acc.exits),
        session_id=acc.session_id if acc.session_id != "_nosession" else None,
        available=True,
        warnings=warnings,
        last_unique_total=acc.last_unique_total,
        last_count_snapshot=acc.last_snapshot,
        system_stop_unique=acc.system_stop_unique,
    )


def count_from_log(
    log_path: Union[str, Path],
    session: Optional[str] = None,
    include_rotated: bool = True,
) -> Counts:
    """Parse events.log (+ rotations/archive). session=None/'latest' | 'all' | id."""
    paths = _iter_log_paths(log_path, include_rotated=include_rotated)
    if not paths:
        return Counts(available=False, warnings=["log_unavailable"],
                      notes=[f"no log at {log_path}"])

    sessions: Dict[str, _SessionAcc] = {}
    order: List[str] = []
    for _path, line in _read_lines(paths):
        rec = parse_log_line(line)
        if rec is None:
            continue
        _accumulate(sessions, order, rec)

    if not sessions:
        return Counts(available=True, warnings=["empty_log"],
                      notes=["no parsable session lines"])

    want = (session or "latest").strip().lower() if session else "latest"
    if want == "all":
        # Sum per-session metrics (as specified for multi-session logs).
        total = Counts(available=True, session_id="all")
        for sid in order:
            c = _counts_from_acc(sessions[sid])
            total.unique_visitors += c.unique_visitors
            total.entries += c.entries
            total.exits += c.exits
            total.returning_visitors += c.returning_visitors
            total.open_visits += c.open_visits
            total.warnings.extend(c.warnings)
        if order:
            last = _counts_from_acc(sessions[order[-1]])
            total.last_unique_total = last.last_unique_total
            total.last_count_snapshot = last.last_count_snapshot
            total.system_stop_unique = last.system_stop_unique
        return total

    if want in ("latest", "none", ""):
        sid = order[-1]
    else:
        # Preserve original case: match exact, then casefold.
        sid = session  # type: ignore[assignment]
        if sid not in sessions:
            folded = {k.casefold(): k for k in sessions}
            sid = folded.get(str(session).casefold())
        if sid is None or sid not in sessions:
            return Counts(available=False, warnings=[f"session_not_found:{session}"],
                          notes=[f"known={list(sessions)}"])
    counts = _counts_from_acc(sessions[sid])
    # Idle dashboard restart: latest session has SYSTEM_START only. Inherit gallery
    # totals from the previous session's COUNT_SNAPSHOT / SYSTEM_STOP so DB↔log
    # audit stays consistent until the next STREAM_END snapshot.
    if (
        want in ("latest", "none", "")
        and len(order) > 1
        and counts.unique_visitors == 0
        and counts.entries == 0
        and counts.exits == 0
        and counts.last_unique_total is None
        and not counts.last_count_snapshot
    ):
        prev = _counts_from_acc(sessions[order[-2]])
        snap = prev.last_count_snapshot
        gallery = prev.last_unique_total
        if gallery is None and snap:
            gallery = snap.get("unique_total")
        if gallery is None:
            gallery = prev.system_stop_unique
        if gallery is not None or snap:
            if snap:
                counts.last_count_snapshot = dict(snap)
            else:
                counts.last_count_snapshot = {
                    "unique_total": int(gallery or 0),
                    "entries": int(prev.system_stop_entries
                                   if prev.system_stop_entries is not None
                                   else prev.entries),
                    "exits": int(prev.system_stop_exits
                                 if prev.system_stop_exits is not None
                                 else prev.exits),
                }
            counts.last_unique_total = (
                int(gallery) if gallery is not None
                else counts.last_count_snapshot.get("unique_total")
            )
            counts.system_stop_unique = prev.system_stop_unique
            counts.notes.append("latest_session_idle_using_prior_snapshot")
    return counts


def count_from_db(db_path: Union[str, Path]) -> Counts:
    """Read counts from a SQLite visitors DB. Postgres callers should use Database APIs."""
    path = Path(db_path)
    if not path.exists():
        return Counts(available=False, warnings=["db_unavailable"],
                      notes=[f"missing {path}"])
    try:
        con = sqlite3.connect(str(path))
        con.row_factory = sqlite3.Row
        try:
            unique = int(con.execute("SELECT COUNT(*) FROM faces").fetchone()[0])
            rows = con.execute(
                "SELECT event_type, COUNT(*) AS c FROM events GROUP BY event_type"
            ).fetchall()
            ec = {r["event_type"]: int(r["c"]) for r in rows}
            entries = int(ec.get("entry", 0))
            exits = int(ec.get("exit", 0))
            returning = int(con.execute(
                "SELECT COUNT(*) FROM faces WHERE visit_count > 1").fetchone()[0])
            try:
                open_v = int(con.execute(
                    "SELECT COUNT(*) FROM visits WHERE exit_ts IS NULL").fetchone()[0])
            except sqlite3.Error:
                open_v = max(0, entries - exits)
            return Counts(
                unique_visitors=unique,
                entries=entries,
                exits=exits,
                returning_visitors=returning,
                open_visits=open_v,
                available=True,
            )
        finally:
            con.close()
    except sqlite3.Error as exc:
        return Counts(available=False, warnings=["db_error"], notes=[str(exc)])


def count_from_database(db: Any) -> Counts:
    """Same as count_from_db but using an open Database instance (sqlite or postgres)."""
    try:
        if hasattr(db, "exists") and not db.exists():
            return Counts(available=False, warnings=["db_unavailable"])
        unique = int(db.unique_count())
        ec = db.event_counts()
        entries = int(ec.get("entry", 0))
        exits = int(ec.get("exit", 0))
        rows = db.query("SELECT COUNT(*) FROM faces WHERE visit_count > 1")
        returning = int(rows[0][0]) if rows else 0
        try:
            vs = db.visit_stats()
            open_v = int(vs.get("open_visits", max(0, entries - exits)))
        except Exception:  # noqa: BLE001
            open_v = max(0, entries - exits)
        return Counts(
            unique_visitors=unique,
            entries=entries,
            exits=exits,
            returning_visitors=returning,
            open_visits=open_v,
            available=True,
        )
    except Exception as exc:  # noqa: BLE001
        return Counts(available=False, warnings=["db_error"], notes=[str(exc)])


def reconcile(
    db_counts: Optional[Counts],
    log_counts: Optional[Counts],
) -> Dict[str, Any]:
    """Compare DB vs log; also cross-check log self-reported unique_total / snapshots."""
    notes: List[str] = []
    diffs: List[str] = []
    db_ok = db_counts is not None and db_counts.available
    log_ok = log_counts is not None and log_counts.available

    if not db_ok and not log_ok:
        return {"consistent": None, "diffs": ["both_unavailable"],
                "notes": ["neither DB nor log available"]}
    if not db_ok:
        notes.append("db_unavailable")
        _log_self_check(log_counts, diffs, notes)
        # Single-source: not an error, but cannot claim cross-check consistency.
        return {"consistent": None, "diffs": diffs, "notes": notes,
                "db": None, "log": log_counts.to_dict() if log_counts else None}
    if not log_ok:
        notes.append("log_unavailable")
        return {"consistent": None, "diffs": [], "notes": notes,
                "db": db_counts.to_dict() if db_counts else None, "log": None}

    assert db_counts is not None and log_counts is not None

    # Gallery size from the log (re-runs often have 0 new FACE_REGISTERED; snapshots still
    # report the live unique_total). Prefer that when comparing to DB `faces` rows.
    snap = log_counts.last_count_snapshot or {}
    log_gallery = (
        log_counts.last_unique_total
        if log_counts.last_unique_total is not None
        else log_counts.system_stop_unique
        if log_counts.system_stop_unique is not None
        else snap.get("unique_total")
        if snap.get("unique_total") is not None
        else log_counts.unique_visitors
    )
    # Cumulative entry/exit totals: COUNT_SNAPSHOT / SYSTEM_STOP match DB when the DB
    # spans sessions; per-session ENTRY line counts alone do not.
    log_entries = snap.get("entries")
    if log_entries is None:
        log_entries = log_counts.entries
    log_exits = snap.get("exits")
    if log_exits is None:
        log_exits = log_counts.exits

    if db_counts.unique_visitors != log_gallery:
        diffs.append(
            f"unique_visitors: db={db_counts.unique_visitors} log={log_gallery}"
        )
    if db_counts.entries != log_entries:
        diffs.append(f"entries: db={db_counts.entries} log={log_entries}")
    if db_counts.exits != log_exits:
        diffs.append(f"exits: db={db_counts.exits} log={log_exits}")
    if db_counts.open_visits != log_counts.open_visits:
        # open_visits is session-scoped on the log; only flag when gallery also disagrees
        # or the session claims open visits while DB has none (or vice versa) mid-run.
        if log_gallery == db_counts.unique_visitors:
            notes.append(
                f"open_visits: db={db_counts.open_visits} log={log_counts.open_visits}"
            )
        else:
            diffs.append(
                f"open_visits: db={db_counts.open_visits} log={log_counts.open_visits}"
            )

    _log_self_check(log_counts, diffs, notes)

    # FACE_REGISTERED-in-session vs gallery unique_total: expected gap on re-runs
    # (no new registrations). Only fail when a snapshot/total disagrees with itself.
    registered = log_counts.unique_visitors
    if log_counts.last_unique_total is not None and log_counts.last_unique_total != registered:
        if registered < log_counts.last_unique_total:
            notes.append(
                f"session_new_FACE_REGISTERED={registered} "
                f"gallery_unique_total={log_counts.last_unique_total} "
                f"(ok if re-run with existing faces)"
            )
        else:
            diffs.append(
                f"log_unique_total_vs_FACE_REGISTERED: "
                f"derived={registered} last_unique_total={log_counts.last_unique_total}"
            )
    if (
        log_counts.system_stop_unique is not None
        and log_gallery is not None
        and log_counts.system_stop_unique != log_gallery
    ):
        diffs.append(
            f"SYSTEM_STOP_unique_visitors: "
            f"gallery={log_gallery} stop={log_counts.system_stop_unique}"
        )
    snap_u = snap.get("unique_total")
    if snap_u is not None and log_gallery is not None and snap_u != log_gallery:
        diffs.append(
            f"COUNT_SNAPSHOT_unique_total: gallery={log_gallery} snapshot={snap_u}"
        )

    notes.extend(log_counts.warnings)
    notes.extend(db_counts.warnings)
    return {
        "consistent": len(diffs) == 0,
        "diffs": diffs,
        "notes": notes,
        "db": db_counts.to_dict(),
        "log": log_counts.to_dict(),
    }


def _log_self_check(log_counts: Optional[Counts], diffs: List[str], notes: List[str]) -> None:
    if not log_counts:
        return
    for w in log_counts.warnings:
        if w.startswith("duplicate_FACE_REGISTERED") or w.startswith("EXIT_without_ENTRY"):
            if w not in diffs:
                diffs.append(w)
        else:
            if w not in notes:
                notes.append(w)


def load_sessions(
    log_path: Union[str, Path],
    include_rotated: bool = True,
) -> Tuple[Dict[str, _SessionAcc], List[str]]:
    """Parse log files into session accumulators + order of SYSTEM_START appearance."""
    paths = _iter_log_paths(log_path, include_rotated=include_rotated)
    sessions: Dict[str, _SessionAcc] = {}
    order: List[str] = []
    for _path, line in _read_lines(paths):
        rec = parse_log_line(line)
        if rec is None:
            continue
        _accumulate(sessions, order, rec)
    return sessions, order


def verify_log_invariants(
    log_path: Union[str, Path],
    session: str = "latest",
    db_path: Optional[Union[str, Path]] = None,
) -> List[str]:
    """Return human-readable violation strings (empty = OK)."""
    paths = _iter_log_paths(log_path, include_rotated=True)
    if not paths:
        return ["log_unavailable"]
    sessions, order = load_sessions(log_path, include_rotated=True)
    if not order:
        return ["no_sessions"]

    want = (session or "latest").strip().lower()
    if want in ("latest", ""):
        targets = [sessions[order[-1]]]
    elif want == "all":
        targets = [sessions[s] for s in order]
    else:
        if session not in sessions:
            return [f"session_not_found:{session}"]
        targets = [sessions[session]]

    by_sid: Dict[str, List[dict]] = {s: [] for s in order}
    for _p, line in _read_lines(paths):
        rec = parse_log_line(line)
        if rec is None:
            continue
        sid = rec.get("sid") or rec["fields"].get("session")
        if sid in by_sid:
            by_sid[sid].append(rec)

    violations: List[str] = []
    now = datetime.now()
    for acc in targets:
        label = acc.session_id
        if not acc.has_start:
            violations.append(f"{label}: missing SYSTEM_START")
        if acc.has_start and not acc.has_stop:
            violations.append(f"{label}: missing SYSTEM_STOP")
        if acc.has_stop and not acc.has_start:
            violations.append(f"{label}: SYSTEM_STOP without SYSTEM_START")
        for vid in sorted(acc.entry_visits):
            if vid not in acc.exit_visits:
                violations.append(f"{label}: ENTRY visit_id={vid} missing EXIT")
        for vid in sorted(acc.exit_visits):
            if vid not in acc.entry_visits:
                violations.append(f"{label}: EXIT visit_id={vid} without ENTRY")
        for tid in sorted(acc.track_ended - acc.track_started):
            violations.append(f"{label}: TRACK_ENDED track={tid} without TRACK_STARTED")
        for w in acc.warnings:
            if w.startswith("duplicate_FACE_REGISTERED") or w.startswith("EXIT_without"):
                violations.append(f"{label}: {w}")
        start = _parse_iso(acc.start_ts)
        stop = _parse_iso(acc.stop_ts)
        for rec in by_sid.get(acc.session_id, []):
            ts = _parse_iso(rec["ts"])
            if ts is None:
                continue
            if ts > now + timedelta(seconds=120):
                violations.append(f"{label}: future timestamp {rec['ts']} on {rec['event']}")
            if start and stop and rec["event"] not in ("SYSTEM_START", "SYSTEM_STOP"):
                if ts < start - timedelta(hours=1) or ts > stop + timedelta(hours=1):
                    violations.append(
                        f"{label}: timestamp {rec['ts']} outside session window "
                        f"[{acc.start_ts} .. {acc.stop_ts}] event={rec['event']}"
                    )
        if db_path and want != "all" and len(targets) == 1:
            db_c = count_from_db(db_path)
            if db_c.available and len(acc.face_set) != db_c.unique_visitors:
                violations.append(
                    f"{label}: FACE_REGISTERED={len(acc.face_set)} "
                    f"!= faces rows={db_c.unique_visitors}"
                )

    seen = set()
    out: List[str] = []
    for v in violations:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", ""))
    except ValueError:
        return None


def audit_payload(
    db_path: Optional[Union[str, Path]] = None,
    log_path: Optional[Union[str, Path]] = None,
    session: Optional[str] = "latest",
    db: Any = None,
) -> Dict[str, Any]:
    """Convenience for API/CLI: fetch both sides and reconcile."""
    db_counts = count_from_database(db) if db is not None else (
        count_from_db(db_path) if db_path else Counts(available=False, warnings=["db_unavailable"])
    )
    log_counts = count_from_log(log_path, session=session) if log_path else Counts(
        available=False, warnings=["log_unavailable"])
    result = reconcile(db_counts, log_counts)
    primary = db_counts if db_counts.available else log_counts
    source = "db" if db_counts.available else ("log" if log_counts.available else "none")
    return {
        "unique_visitors": primary.unique_visitors if primary.available else None,
        "entries": primary.entries if primary.available else None,
        "exits": primary.exits if primary.available else None,
        "currently_inside": primary.open_visits if primary.available else None,
        "returning_visitors": primary.returning_visitors if primary.available else None,
        "open_visits": primary.open_visits if primary.available else None,
        "source": source,
        "derived_from_log": log_counts.to_dict() if log_counts.available else None,
        "db_counts": db_counts.to_dict() if db_counts.available else None,
        "consistent": result["consistent"],
        "diffs": result["diffs"],
        "notes": result.get("notes", []),
        "session_id": log_counts.session_id if log_counts.available else None,
    }
