"""Print / export results from the SQLite DB.
   python scripts/report.py --db data/visitors.db [--csv-dir samples]"""
import argparse
import csv
import sqlite3
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/visitors.db")
    ap.add_argument("--csv-dir", help="Export faces.csv and events.csv here")
    a = ap.parse_args()
    con = sqlite3.connect(a.db)
    print("Unique visitors:", con.execute("SELECT COUNT(*) FROM faces").fetchone()[0])
    print("Events:", dict(con.execute("SELECT event_type, COUNT(*) FROM events GROUP BY 1").fetchall()))
    print("\nface_id | first_seen | last_seen | visits")
    for r in con.execute("SELECT * FROM faces ORDER BY face_id"):
        print(" | ".join(map(str, r)))
    print("\nLast 15 events:")
    for r in con.execute("SELECT id, face_id, event_type, timestamp, image_path FROM events ORDER BY id DESC LIMIT 15"):
        print(" | ".join(map(str, r)))
    if a.csv_dir:
        out = Path(a.csv_dir)
        out.mkdir(parents=True, exist_ok=True)
        for table in ("faces", "events"):
            cur = con.execute(f"SELECT * FROM {table}")
            with open(out / f"{table}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([d[0] for d in cur.description])
                w.writerows(cur.fetchall())
        print(f"\nExported CSVs to {out}/")


if __name__ == "__main__":
    main()
