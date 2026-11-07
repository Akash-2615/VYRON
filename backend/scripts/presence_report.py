#!/usr/bin/env python3
"""Report visits / gaps / suspicious short re-entries.

    python backend/scripts/presence_report.py
    python backend/scripts/presence_report.py --db data/db/visitors.db
    python backend/scripts/presence_report.py --dsn postgresql://facetrack:facetrack@localhost:5432/facetrack
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import os
_BACKEND = Path(__file__).resolve().parents[1]
_PROJECT = _BACKEND.parent
sys.path.insert(0, str(_BACKEND))
os.chdir(_PROJECT)
from src.database import Database  # noqa: E402


def parse_ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def _open(args) -> Database:
    if args.dsn:
        return Database(dsn=args.dsn, backend="postgres", reset=False)
    if args.db:
        return Database(path=args.db, backend="sqlite", reset=False)
    from src.config import load_config
    return Database.from_storage(load_config("config.json")["storage"], reset=False)


def main() -> int:
    p = argparse.ArgumentParser(description="Presence / visit quality report")
    p.add_argument("--db", default=None, help="SQLite path")
    p.add_argument("--dsn", default=None, help="Postgres DSN")
    p.add_argument("--grace", type=float, default=10.0,
                   help="Flag re-entry within 2x this many seconds of prior exit")
    args = p.parse_args()
    if args.db and not Path(args.db).exists():
        print(f"DB not found: {args.db}")
        return 1
    db = _open(args)
    faces = db.fetch_dicts("SELECT face_id FROM faces ORDER BY face_id")
    entries = db.event_counts().get("entry", 0)
    exits = db.event_counts().get("exit", 0)
    label = args.db or args.dsn or f"config:{db.backend}"
    print(f"DB: {label} ({db.backend})")
    print(f"Unique faces: {len(faces)}  | entries={entries} exits={exits}  "
          f"{'OK' if entries == exits else 'MISMATCH'}")
    suspicious = 0
    try:
        visits = db.fetch_dicts("SELECT * FROM visits ORDER BY face_id, visit_id")
    except Exception as e:  # noqa: BLE001
        print(f"No visits table yet — {e}")
        db.close()
        return 0
    print(f"Visits: {len(visits)}")
    by_face = {}
    for v in visits:
        by_face.setdefault(v["face_id"], []).append(v)
    for fid, vs in by_face.items():
        print(f"\nFace #{fid}  ({len(vs)} visit{'s' if len(vs) != 1 else ''})")
        prev_exit = None
        for v in vs:
            tif = float(v["visible_seconds"] or 0) + float(v["occluded_seconds"] or 0)
            print(f"  visit {v['visit_id']}: in={tif:.1f}s visible={v['visible_seconds']:.1f}s "
                  f"occluded={v['occluded_seconds']:.1f}s resumes={v['resume_count']} "
                  f"reason={v['exit_reason'] or 'open'}")
            if prev_exit and v["entry_ts"]:
                gap = parse_ts(v["entry_ts"]) - parse_ts(prev_exit)
                if 0 <= gap <= 2 * args.grace:
                    print(f"    !! suspicious short re-entry ({gap:.1f}s after prior exit)")
                    suspicious += 1
            if v["exit_ts"]:
                prev_exit = v["exit_ts"]
    print(f"\nSuspicious short re-entries: {suspicious}")
    db.close()
    return 0 if entries == exits and suspicious == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
