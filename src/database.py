"""SQLite persistence (WAL mode, commit-per-write for crash resilience)."""
import sqlite3
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS faces (
    face_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL,
    visit_count INTEGER NOT NULL DEFAULT 0,
    image_path  TEXT,
    appearance_summary TEXT,
    appearance_json    TEXT
);
CREATE TABLE IF NOT EXISTS embeddings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id    INTEGER NOT NULL REFERENCES faces(face_id),
    vector     BLOB NOT NULL,
    created_at TEXT NOT NULL,
    image_path TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id    INTEGER NOT NULL REFERENCES faces(face_id),
    event_type TEXT NOT NULL CHECK (event_type IN ('entry', 'exit')),
    timestamp  TEXT NOT NULL,
    image_path TEXT,
    track_id   INTEGER,
    frame_idx  INTEGER,
    note       TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_face ON events(face_id);
CREATE INDEX IF NOT EXISTS idx_emb_face ON embeddings(face_id);

CREATE TABLE IF NOT EXISTS away_visitors (
    away_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    first_seen          TEXT NOT NULL,
    last_seen           TEXT NOT NULL,
    visit_count         INTEGER NOT NULL DEFAULT 0,
    image_path          TEXT,
    appearance_summary  TEXT,
    appearance_json     TEXT
);
CREATE TABLE IF NOT EXISTS away_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    away_id    INTEGER NOT NULL REFERENCES away_visitors(away_id),
    event_type TEXT NOT NULL CHECK (event_type IN ('entry', 'exit')),
    timestamp  TEXT NOT NULL,
    image_path TEXT,
    track_id   INTEGER,
    frame_idx  INTEGER,
    note       TEXT
);
CREATE INDEX IF NOT EXISTS idx_away_events ON away_events(away_id);
"""


class Database:
    def __init__(self, path: str, reset: bool = False):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            for suffix in ("", "-wal", "-shm"):
                Path(str(p) + suffix).unlink(missing_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(p), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after the first schema (safe on existing DBs)."""
        for table in ("faces", "embeddings"):
            cols = {r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")}
            if "image_path" not in cols:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN image_path TEXT")
        face_cols = {r[1] for r in self._conn.execute("PRAGMA table_info(faces)")}
        if "appearance_summary" not in face_cols:
            self._conn.execute("ALTER TABLE faces ADD COLUMN appearance_summary TEXT")
        if "appearance_json" not in face_cols:
            self._conn.execute("ALTER TABLE faces ADD COLUMN appearance_json TEXT")

    # ---- writes (each is its own transaction) ----
    def add_face(self, ts: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO faces(first_seen, last_seen, visit_count) VALUES (?,?,0)", (ts, ts))
            return int(cur.lastrowid)

    def add_embedding(self, face_id: int, vec: np.ndarray, ts: str, image_path: Optional[str] = None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO embeddings(face_id, vector, created_at, image_path) VALUES (?,?,?,?)",
                (face_id, vec.astype(np.float32).tobytes(), ts, image_path))

    def set_face_image(self, face_id: int, image_path: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE faces SET image_path=? WHERE face_id=?", (image_path, face_id))

    def set_appearance(self, face_id: int, summary: str, payload_json: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE faces SET appearance_summary=?, appearance_json=? WHERE face_id=?",
                (summary, payload_json, face_id))

    def get_appearance_summary(self, face_id: int) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT appearance_summary FROM faces WHERE face_id=?", (face_id,)).fetchone()
        return row[0] if row else None

    def add_away_visitor(self, ts: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO away_visitors(first_seen, last_seen, visit_count) VALUES (?,?,0)",
                (ts, ts))
            return int(cur.lastrowid)

    def set_away_appearance(self, away_id: int, summary: str, payload_json: str,
                            image_path: Optional[str] = None) -> None:
        with self._lock, self._conn:
            if image_path:
                self._conn.execute(
                    "UPDATE away_visitors SET appearance_summary=?, appearance_json=?, image_path=? "
                    "WHERE away_id=?", (summary, payload_json, image_path, away_id))
            else:
                self._conn.execute(
                    "UPDATE away_visitors SET appearance_summary=?, appearance_json=? WHERE away_id=?",
                    (summary, payload_json, away_id))

    def log_away_event(self, away_id: int, event_type: str, ts: str, image_path: Optional[str],
                       track_id: Optional[int], frame_idx: Optional[int], note: str = "") -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO away_events(away_id, event_type, timestamp, image_path, track_id, frame_idx, note) "
                "VALUES (?,?,?,?,?,?,?)",
                (away_id, event_type, ts, image_path, track_id, frame_idx, note))
            if event_type == "entry":
                self._conn.execute(
                    "UPDATE away_visitors SET last_seen=?, visit_count=visit_count+1 WHERE away_id=?",
                    (ts, away_id))
            else:
                self._conn.execute("UPDATE away_visitors SET last_seen=? WHERE away_id=?", (ts, away_id))

    def away_stats(self) -> dict:
        with self._lock:
            unique = int(self._conn.execute("SELECT COUNT(*) FROM away_visitors").fetchone()[0])
            rows = self._conn.execute(
                "SELECT event_type, COUNT(*) FROM away_events GROUP BY event_type").fetchall()
            clothed = int(self._conn.execute(
                "SELECT COUNT(*) FROM away_visitors WHERE appearance_summary IS NOT NULL "
                "AND appearance_summary != ''").fetchone()[0])
        counts = {k: v for k, v in rows}
        entries = counts.get("entry", 0)
        exits = counts.get("exit", 0)
        return {
            "away_unique": unique,
            "away_entries": entries,
            "away_exits": exits,
            "away_currently": max(0, entries - exits),
            "away_with_clothes": clothed,
        }

    def log_event(self, face_id: int, event_type: str, ts: str, image_path: Optional[str],
                  track_id: Optional[int], frame_idx: Optional[int], note: str = "") -> None:
        """Insert the event and refresh the face's last_seen/visit_count atomically."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO events(face_id, event_type, timestamp, image_path, track_id, frame_idx, note) "
                "VALUES (?,?,?,?,?,?,?)",
                (face_id, event_type, ts, image_path, track_id, frame_idx, note))
            if event_type == "entry":
                self._conn.execute(
                    "UPDATE faces SET last_seen=?, visit_count=visit_count+1 WHERE face_id=?", (ts, face_id))
            else:
                self._conn.execute("UPDATE faces SET last_seen=? WHERE face_id=?", (ts, face_id))

    # ---- reads ----
    def load_embeddings(self) -> List[Tuple[int, np.ndarray]]:
        with self._lock:
            rows = self._conn.execute("SELECT face_id, vector FROM embeddings").fetchall()
        return [(fid, np.frombuffer(blob, dtype=np.float32).copy()) for fid, blob in rows]

    def unique_count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0])

    def event_counts(self) -> dict:
        with self._lock:
            rows = self._conn.execute("SELECT event_type, COUNT(*) FROM events GROUP BY event_type").fetchall()
        return {k: v for k, v in rows}

    def query(self, sql: str, params: tuple = ()) -> list:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
                self._conn.close()
            except sqlite3.Error:
                pass
