#!/usr/bin/env python3
"""Print unique visitor counts from DB and/or events.log and reconcile them.

Exit codes: 0 = consistent (or single-source OK), 1 = mismatch, 2 = could not compute.

    python scripts/count.py
    python scripts/count.py --source both --json
    python scripts/count.py --source log --session latest
    python scripts/count.py --session all --log logs/events.log
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.counting import (  # noqa: E402
    Counts, count_from_db, count_from_database, count_from_log, reconcile,
)


def _default_paths(args):
    db = args.db
    log = args.log
    storage = None
    if db is None or log is None:
        try:
            from src.config import load_config
            storage = load_config("config.json")["storage"]
            if db is None:
                db = st_path = storage.get("db_path") or "data/visitors.db"
            if log is None:
                log = storage.get("events_log") or "logs/events.log"
        except Exception:  # noqa: BLE001
            db = db or "data/visitors.db"
            log = log or "logs/events.log"
    return db, log, storage


def _load_db_counts(args, db_path, storage) -> Counts:
    """Prefer config storage (Postgres) unless --db points at an explicit SQLite file."""
    if args.db:
        return count_from_db(db_path)
    if storage is not None:
        try:
            from src.database import Database
            db = Database.from_storage(storage, reset=False)
            try:
                return count_from_database(db)
            finally:
                db.close()
        except Exception as exc:  # noqa: BLE001
            # Fall back to SQLite path from config.
            c = count_from_db(db_path)
            if not c.available:
                c.notes.append(f"storage_open_failed:{exc}")
            return c
    return count_from_db(db_path)


def _print_table(rows: list) -> None:
    headers = ["source", "unique", "entries", "exits", "returning", "open"]
    widths = [max(len(h), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for r in rows:
        print(fmt.format(*r))


def main() -> int:
    ap = argparse.ArgumentParser(description="Reconcile unique visitor counts (DB vs log)")
    ap.add_argument("--source", choices=("db", "log", "both"), default="both")
    ap.add_argument("--db", default=None, help="SQLite DB path (overrides config storage)")
    ap.add_argument("--log", default=None, help="events.log path")
    ap.add_argument("--session", default="latest",
                    help="latest | all | <session_id>")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-rotated", action="store_true",
                    help="Do not read events.log.N / logs/archive/")
    args = ap.parse_args()
    db_path, log_path, storage = _default_paths(args)

    db_c = Counts(available=False, warnings=["skipped"])
    log_c = Counts(available=False, warnings=["skipped"])
    if args.source in ("db", "both"):
        db_c = _load_db_counts(args, db_path, storage)
    if args.source in ("log", "both"):
        log_c = count_from_log(
            log_path, session=args.session, include_rotated=not args.no_rotated)

    if args.source == "db":
        result = reconcile(db_c, None)
    elif args.source == "log":
        result = reconcile(None, log_c)
    else:
        result = reconcile(db_c, log_c)

    backend = (storage or {}).get("backend", "postgres") if not args.db else "sqlite"
    if args.json:
        print(json.dumps({
            "db": db_c.to_dict() if args.source != "log" else None,
            "log": log_c.to_dict() if args.source != "db" else None,
            "reconcile": result,
            "db_path": db_path,
            "db_backend": backend,
            "log_path": log_path,
            "session": args.session,
        }, indent=2))
    else:
        print(f"DB backend: {backend}  log: {log_path}  session: {args.session}")
        rows = []
        if args.source in ("db", "both"):
            if db_c.available:
                rows.append(("db", db_c.unique_visitors, db_c.entries, db_c.exits,
                             db_c.returning_visitors, db_c.open_visits))
            else:
                rows.append(("db", "—", "—", "—", "—", "—"))
        if args.source in ("log", "both"):
            if log_c.available:
                rows.append((f"log:{log_c.session_id or args.session}",
                             log_c.unique_visitors, log_c.entries, log_c.exits,
                             log_c.returning_visitors, log_c.open_visits))
            else:
                rows.append(("log", "—", "—", "—", "—", "—"))
        _print_table(rows)
        consistent = result.get("consistent")
        if consistent is True:
            print("\nVerdict: CONSISTENT")
        elif consistent is False:
            print("\nVerdict: MISMATCH")
            for d in result.get("diffs", []):
                print(f"  - {d}")
        else:
            print("\nVerdict: SINGLE-SOURCE / INCOMPLETE")
            for n in result.get("notes", []):
                print(f"  note: {n}")
        if result.get("notes") and consistent is False:
            for n in result["notes"]:
                print(f"  note: {n}")
        for w in (db_c.warnings + log_c.warnings):
            if w not in (result.get("notes") or []):
                print(f"  warn: {w}")

    if not db_c.available and not log_c.available:
        return 2
    if args.source == "both" and result.get("consistent") is False:
        return 1
    if args.source == "log" and any(
        w.startswith("duplicate_") or w.startswith("EXIT_without")
        for w in log_c.warnings
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
