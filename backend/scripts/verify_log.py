#!/usr/bin/env python3
"""Check session log invariants. Exit 1 on violations, 2 if log unreadable.

    python backend/scripts/verify_log.py
    python backend/scripts/verify_log.py --log logs/events.log --session latest
    python backend/scripts/verify_log.py --db data/db/visitors.db
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import os
_BACKEND = Path(__file__).resolve().parents[1]
_PROJECT = _BACKEND.parent
sys.path.insert(0, str(_BACKEND))
os.chdir(_PROJECT)

from src.counting import verify_log_invariants  # noqa: E402


def _default_paths(args):
    log = args.log
    db = args.db
    if log is None or db is None:
        try:
            from src.config import load_config
            st = load_config("config.json")["storage"]
            log = log or st.get("events_log") or "logs/events.log"
            db = db or st.get("db_path") or "data/db/visitors.db"
        except Exception:  # noqa: BLE001
            log = log or "logs/events.log"
            db = db or "data/db/visitors.db"
    return db, log


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify events.log session invariants")
    ap.add_argument("--log", default=None)
    ap.add_argument("--db", default=None, help="Optional SQLite DB for FACE_REGISTERED vs faces")
    ap.add_argument("--session", default="latest")
    args = ap.parse_args()
    db_path, log_path = _default_paths(args)

    violations = verify_log_invariants(log_path, session=args.session, db_path=db_path)
    if violations == ["log_unavailable"]:
        print("ERROR: could not read log", log_path)
        return 2
    if not violations:
        print(f"OK — invariants hold for session={args.session} ({log_path})")
        return 0
    print(f"FAIL — {len(violations)} violation(s):")
    for v in violations:
        print(f"  - {v}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
