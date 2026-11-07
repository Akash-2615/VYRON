"""Print / export results from the visitors DB (Postgres or SQLite).

    python scripts/report.py
    python scripts/report.py --db data/visitors.db
    python scripts/report.py --dsn postgresql://facetrack:facetrack@localhost:5432/facetrack
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.database import Database  # noqa: E402


def _open(args) -> Database:
    if args.dsn:
        return Database(dsn=args.dsn, backend="postgres", reset=False)
    if args.db and (args.db.endswith(".db") or Path(args.db).exists()):
        return Database(path=args.db, backend="sqlite", reset=False)
    # Default: config storage
    from src.config import load_config
    return Database.from_storage(load_config("config.json")["storage"], reset=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None, help="SQLite path (optional)")
    ap.add_argument("--dsn", default=None, help="Postgres DSN (optional)")
    ap.add_argument("--csv-dir", help="Export faces.csv and events.csv here")
    a = ap.parse_args()
    db = _open(a)
    print(f"Backend: {db.backend}")
    print("Unique visitors:", db.unique_count())
    print("Events:", db.event_counts())
    print("\nface_id | first_seen | last_seen | visits | image_path")
    for r in db.fetch_dicts("SELECT face_id, first_seen, last_seen, visit_count, image_path FROM faces ORDER BY face_id"):
        print(" | ".join(str(r[k]) for k in ("face_id", "first_seen", "last_seen", "visit_count", "image_path")))
    print("\nLast 15 events:")
    for r in db.fetch_dicts(
        "SELECT id, face_id, event_type, timestamp, image_path FROM events ORDER BY id DESC LIMIT 15"
    ):
        print(" | ".join(str(r[k]) for k in ("id", "face_id", "event_type", "timestamp", "image_path")))
    if a.csv_dir:
        out = Path(a.csv_dir)
        out.mkdir(parents=True, exist_ok=True)
        for table in ("faces", "events"):
            rows = db.fetch_dicts(f"SELECT * FROM {table}")
            if not rows:
                (out / f"{table}.csv").write_text("")
                continue
            with open(out / f"{table}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                # bytes columns (embeddings vectors) aren't in these tables
                w.writerows(rows)
        print(f"\nExported CSVs to {out}/")
    db.close()


if __name__ == "__main__":
    main()
