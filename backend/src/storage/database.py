"""Persistence layer: PostgreSQL (runtime) or SQLite (unit tests only).

Runtime always opens Postgres via ``src.storage.bootstrap.open_database``.
SQLite remains available as ``Database(path=..., backend="sqlite")`` for pytest.
Commit-per-write. Schema migrations live in ``Database._migrate``.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from .schema import _PG_DROP, _PG_SCHEMA, _SQLITE_SCHEMA


def _adapt_sql(sql: str, backend: str) -> str:
    """Use ``?`` placeholders everywhere; expand to ``%s`` for psycopg."""
    if backend == "postgres":
        return sql.replace("?", "%s")
    return sql


class _DictRow(dict):
    """dict that also supports index access like sqlite3.Row."""

    def __getitem__(self, key: Union[str, int]) -> Any:
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


class Database:
    """Visitor / event store.

    Construction:
      * ``Database(path)`` — SQLite file (tests, seed_demo).
      * ``Database.from_storage(cfg["storage"], reset=...)`` — respects
        ``backend`` (``postgres`` | ``sqlite``).
    """

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        dsn: Optional[str] = None,
        reset: bool = False,
        backend: Optional[str] = None,
    ):
        self._lock = threading.Lock()
        if backend is None:
            backend = "postgres" if dsn else "sqlite"
        backend = backend.lower().strip()
        if backend not in ("sqlite", "postgres"):
            raise ValueError(f"Unknown storage backend: {backend}")
        self.backend = backend
        self.path = path
        self.dsn = dsn
        self._conn: Any

        if backend == "sqlite":
            if not path:
                raise ValueError("SQLite backend requires path")
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            if reset:
                # Keep a timestamped copy so a mistaken wipe (or kill mid-reset) is recoverable.
                if p.is_file() and p.stat().st_size > 0:
                    from datetime import datetime as _dt
                    bak_dir = p.parent / "backups"
                    bak_dir.mkdir(parents=True, exist_ok=True)
                    stamp = _dt.now().strftime("%Y%m%dT%H%M%S")
                    bak = bak_dir / f"{p.stem}_{stamp}{p.suffix}"
                    try:
                        import shutil
                        shutil.copy2(p, bak)
                    except OSError:
                        pass
                for suffix in ("", "-wal", "-shm"):
                    Path(str(p) + suffix).unlink(missing_ok=True)
            self._conn = sqlite3.connect(str(p), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SQLITE_SCHEMA)
            self._migrate_sqlite()
            self._conn.commit()
        else:
            if not dsn:
                raise ValueError("PostgreSQL backend requires dsn")
            try:
                import psycopg
                from psycopg.rows import dict_row
            except ImportError as e:
                raise ImportError(
                    "psycopg is required for PostgreSQL. "
                    "pip install 'psycopg[binary]>=3.1'"
                ) from e
            self._psycopg = psycopg
            self._dict_row = dict_row
            self._conn = psycopg.connect(dsn, row_factory=dict_row, autocommit=False)
            if reset:
                with self._conn.cursor() as cur:
                    cur.execute(_PG_DROP)
                self._conn.commit()
            with self._conn.cursor() as cur:
                cur.execute(_PG_SCHEMA)
            self._conn.commit()
            self._migrate_postgres()

    @classmethod
    def from_storage(cls, storage: dict, reset: bool = False) -> "Database":
        backend = str(storage.get("backend") or "postgres").lower().strip()
        if backend == "sqlite":
            # Unit tests / tooling only — runtime must use open_database (Postgres).
            return cls(path=storage["db_path"], backend="sqlite", reset=reset)
        pg = storage.get("postgres") or {}
        dsn = (
            storage.get("dsn")
            or pg.get("dsn")
            or _dsn_from_parts(pg)
        )
        if not dsn:
            raise ValueError(
                "storage.backend=postgres requires storage.postgres.dsn "
                "(or host/port/user/password/database)"
            )
        return cls(dsn=dsn, backend="postgres", reset=reset)

    # ---- internals ----
    def _sql(self, sql: str) -> str:
        return _adapt_sql(sql, self.backend)

    def _execute(self, sql: str, params: Sequence[Any] = ()):
        sql = self._sql(sql)
        if self.backend == "sqlite":
            return self._conn.execute(sql, params)
        cur = self._conn.cursor()
        cur.execute(sql, params)
        return cur

    def _insert_returning(self, sql: str, params: Sequence[Any], id_col: str) -> int:
        """INSERT and return the new primary key (works on both backends)."""
        if self.backend == "sqlite":
            cur = self._conn.execute(self._sql(sql), params)
            return int(cur.lastrowid)
        # postgres
        if "RETURNING" not in sql.upper():
            sql = sql.rstrip().rstrip(";") + f" RETURNING {id_col}"
        cur = self._conn.cursor()
        cur.execute(self._sql(sql), params)
        row = cur.fetchone()
        return int(row[id_col] if isinstance(row, dict) else row[0])

    def _migrate_sqlite(self) -> None:
        for table in ("faces", "embeddings"):
            cols = {r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")}
            if "image_path" not in cols:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN image_path TEXT")
        face_cols = {r[1] for r in self._conn.execute("PRAGMA table_info(faces)")}
        if "appearance_summary" not in face_cols:
            self._conn.execute("ALTER TABLE faces ADD COLUMN appearance_summary TEXT")
        if "appearance_json" not in face_cols:
            self._conn.execute("ALTER TABLE faces ADD COLUMN appearance_json TEXT")
        if "appearance_image_path" not in face_cols:
            self._conn.execute("ALTER TABLE faces ADD COLUMN appearance_image_path TEXT")
        away_cols = {r[1] for r in self._conn.execute("PRAGMA table_info(away_visitors)")}
        if "linked_face_id" not in away_cols:
            self._conn.execute("ALTER TABLE away_visitors ADD COLUMN linked_face_id INTEGER")
        evt_cols = {r[1] for r in self._conn.execute("PRAGMA table_info(events)")}
        if "visit_id" not in evt_cols:
            self._conn.execute("ALTER TABLE events ADD COLUMN visit_id INTEGER")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS visits ("
            "visit_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "face_id INTEGER NOT NULL REFERENCES faces(face_id),"
            "entry_ts TEXT NOT NULL,"
            "exit_ts TEXT,"
            "visible_seconds REAL NOT NULL DEFAULT 0,"
            "occluded_seconds REAL NOT NULL DEFAULT 0,"
            "resume_count INTEGER NOT NULL DEFAULT 0,"
            "exit_reason TEXT)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_visits_face ON visits(face_id)")

    def _migrate_postgres(self) -> None:
        """Add columns introduced after the first schema (safe on existing DBs)."""
        def cols(table: str) -> set:
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema='public' AND table_name=%s",
                    (table,),
                )
                return {r["column_name"] if isinstance(r, dict) else r[0] for r in cur.fetchall()}

        face_cols = cols("faces")
        for col, typ in (
            ("appearance_summary", "TEXT"),
            ("appearance_json", "TEXT"),
            ("appearance_image_path", "TEXT"),
            ("image_path", "TEXT"),
        ):
            if col not in face_cols:
                with self._conn.cursor() as cur:
                    cur.execute(f"ALTER TABLE faces ADD COLUMN {col} {typ}")
        emb_cols = cols("embeddings")
        if "image_path" not in emb_cols:
            with self._conn.cursor() as cur:
                cur.execute("ALTER TABLE embeddings ADD COLUMN image_path TEXT")
        away_cols = cols("away_visitors")
        if "linked_face_id" not in away_cols:
            with self._conn.cursor() as cur:
                cur.execute("ALTER TABLE away_visitors ADD COLUMN linked_face_id INTEGER")
        evt_cols = cols("events")
        if "visit_id" not in evt_cols:
            with self._conn.cursor() as cur:
                cur.execute("ALTER TABLE events ADD COLUMN visit_id INTEGER")
        self._conn.commit()

    def exists(self) -> bool:
        if self.backend == "sqlite":
            return bool(self.path and Path(self.path).exists())
        try:
            with self._lock:
                self._execute("SELECT 1").fetchone()
            return True
        except Exception:  # noqa: BLE001
            return False

    @contextmanager
    def readonly(self) -> Iterator[Any]:
        """Yield a connection suitable for read-only API queries (dict rows)."""
        if self.backend == "sqlite":
            if not self.path or not Path(self.path).exists():
                raise FileNotFoundError(self.path)
            con = sqlite3.connect(self.path, timeout=5)
            con.execute("PRAGMA query_only=ON")
            con.row_factory = sqlite3.Row
            try:
                yield _SqliteReadConn(con)
            finally:
                con.close()
        else:
            con = self._psycopg.connect(
                self.dsn, row_factory=self._dict_row, autocommit=True)
            try:
                with con.cursor() as cur:
                    cur.execute("SET default_transaction_read_only = on")
                yield _PgReadConn(con)
            finally:
                con.close()

    # ---- writes (each is its own transaction) ----
    def add_face(self, ts: str) -> int:
        with self._lock:
            fid = self._insert_returning(
                "INSERT INTO faces(first_seen, last_seen, visit_count) VALUES (?,?,0)",
                (ts, ts), "face_id")
            self._conn.commit()
            return fid

    def delete_face(self, face_id: int) -> bool:
        """Remove one gallery identity and all related rows (embeddings / events / visits)."""
        fid = int(face_id)
        with self._lock:
            row = self._execute("SELECT face_id FROM faces WHERE face_id=?", (fid,)).fetchone()
            if row is None:
                return False
            self._execute(
                "UPDATE away_visitors SET linked_face_id=NULL WHERE linked_face_id=?", (fid,))
            self._execute("DELETE FROM embeddings WHERE face_id=?", (fid,))
            self._execute("DELETE FROM events WHERE face_id=?", (fid,))
            self._execute("DELETE FROM visits WHERE face_id=?", (fid,))
            self._execute("DELETE FROM faces WHERE face_id=?", (fid,))
            self._conn.commit()
            return True

    def delete_faces(self, face_ids) -> List[int]:
        removed = []
        for fid in face_ids:
            if self.delete_face(int(fid)):
                removed.append(int(fid))
        return removed

    def add_embedding(self, face_id: int, vec: np.ndarray, ts: str,
                      image_path: Optional[str] = None) -> None:
        blob = vec.astype(np.float32).tobytes()
        with self._lock:
            self._execute(
                "INSERT INTO embeddings(face_id, vector, created_at, image_path) VALUES (?,?,?,?)",
                (face_id, blob, ts, image_path))
            self._conn.commit()

    def set_face_image(self, face_id: int, image_path: str) -> None:
        with self._lock:
            self._execute("UPDATE faces SET image_path=? WHERE face_id=?", (image_path, face_id))
            self._conn.commit()

    def set_appearance(self, face_id: int, summary: str, payload_json: str,
                       image_path: Optional[str] = None) -> None:
        with self._lock:
            if image_path:
                self._execute(
                    "UPDATE faces SET appearance_summary=?, appearance_json=?, appearance_image_path=? "
                    "WHERE face_id=?",
                    (summary, payload_json, image_path, face_id))
            else:
                self._execute(
                    "UPDATE faces SET appearance_summary=?, appearance_json=? WHERE face_id=?",
                    (summary, payload_json, face_id))
            self._conn.commit()

    def get_appearance_summary(self, face_id: int) -> Optional[str]:
        with self._lock:
            row = self._execute(
                "SELECT appearance_summary FROM faces WHERE face_id=?", (face_id,)).fetchone()
        if row is None:
            return None
        return row["appearance_summary"] if isinstance(row, dict) else row[0]

    def get_appearance_image(self, face_id: int) -> Optional[str]:
        with self._lock:
            row = self._execute(
                "SELECT appearance_image_path FROM faces WHERE face_id=?", (face_id,)).fetchone()
        if not row:
            return None
        val = row["appearance_image_path"] if isinstance(row, dict) else row[0]
        return val or None

    def add_away_visitor(self, ts: str) -> int:
        with self._lock:
            aid = self._insert_returning(
                "INSERT INTO away_visitors(first_seen, last_seen, visit_count) VALUES (?,?,0)",
                (ts, ts), "away_id")
            self._conn.commit()
            return aid

    def set_away_appearance(self, away_id: int, summary: str, payload_json: str,
                            image_path: Optional[str] = None) -> None:
        with self._lock:
            if image_path:
                self._execute(
                    "UPDATE away_visitors SET appearance_summary=?, appearance_json=?, image_path=? "
                    "WHERE away_id=?", (summary, payload_json, image_path, away_id))
            else:
                self._execute(
                    "UPDATE away_visitors SET appearance_summary=?, appearance_json=? WHERE away_id=?",
                    (summary, payload_json, away_id))
            self._conn.commit()

    def set_away_linked_face(self, away_id: int, face_id: int) -> None:
        with self._lock:
            self._execute(
                "UPDATE away_visitors SET linked_face_id=? WHERE away_id=? AND "
                "(linked_face_id IS NULL OR linked_face_id=?)",
                (face_id, away_id, face_id))
            self._conn.commit()

    def log_away_event(self, away_id: int, event_type: str, ts: str, image_path: Optional[str],
                       track_id: Optional[int], frame_idx: Optional[int], note: str = "") -> None:
        with self._lock:
            self._execute(
                "INSERT INTO away_events(away_id, event_type, timestamp, image_path, track_id, frame_idx, note) "
                "VALUES (?,?,?,?,?,?,?)",
                (away_id, event_type, ts, image_path, track_id, frame_idx, note))
            if event_type == "entry":
                self._execute(
                    "UPDATE away_visitors SET last_seen=?, visit_count=visit_count+1 WHERE away_id=?",
                    (ts, away_id))
            else:
                self._execute("UPDATE away_visitors SET last_seen=? WHERE away_id=?", (ts, away_id))
            self._conn.commit()

    def away_stats(self) -> dict:
        with self._lock:
            unique = int(self._scalar("SELECT COUNT(*) FROM away_visitors"))
            rows = self._execute(
                "SELECT event_type, COUNT(*) AS c FROM away_events GROUP BY event_type").fetchall()
            clothed = int(self._scalar(
                "SELECT COUNT(*) FROM away_visitors WHERE appearance_summary IS NOT NULL "
                "AND appearance_summary != ''"))
        counts = {}
        for r in rows:
            if isinstance(r, dict):
                counts[r["event_type"]] = int(r["c"])
            else:
                counts[r[0]] = int(r[1])
        entries = counts.get("entry", 0)
        exits = counts.get("exit", 0)
        return {
            "away_unique": unique,
            "away_entries": entries,
            "away_exits": exits,
            "away_currently": max(0, entries - exits),
            "away_with_clothes": clothed,
        }

    def _scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = self._execute(sql, params).fetchone()
        if row is None:
            return 0
        if isinstance(row, dict):
            return next(iter(row.values()))
        return row[0]

    def log_event(self, face_id: int, event_type: str, ts: str, image_path: Optional[str],
                  track_id: Optional[int], frame_idx: Optional[int], note: str = "",
                  visit_id: Optional[int] = None) -> None:
        with self._lock:
            self._execute(
                "INSERT INTO events(face_id, event_type, timestamp, image_path, track_id, frame_idx, note, visit_id) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (face_id, event_type, ts, image_path, track_id, frame_idx, note, visit_id))
            if event_type == "entry":
                self._execute(
                    "UPDATE faces SET last_seen=?, visit_count=visit_count+1 WHERE face_id=?",
                    (ts, face_id))
            else:
                self._execute("UPDATE faces SET last_seen=? WHERE face_id=?", (ts, face_id))
            self._conn.commit()

    def start_visit(self, face_id: int, entry_ts: str) -> int:
        with self._lock:
            vid = self._insert_returning(
                "INSERT INTO visits(face_id, entry_ts) VALUES (?,?)",
                (face_id, entry_ts), "visit_id")
            self._conn.commit()
            return vid

    def close_visit(self, visit_id: int, exit_ts: str, visible_seconds: float,
                    occluded_seconds: float, resume_count: int, exit_reason: str) -> None:
        with self._lock:
            self._execute(
                "UPDATE visits SET exit_ts=?, visible_seconds=?, occluded_seconds=?, "
                "resume_count=?, exit_reason=? WHERE visit_id=?",
                (exit_ts, float(visible_seconds), float(occluded_seconds),
                 int(resume_count), exit_reason, int(visit_id)))
            self._conn.commit()

    def update_visit_progress(self, visit_id: int, visible_seconds: float,
                              occluded_seconds: float, resume_count: int) -> None:
        with self._lock:
            self._execute(
                "UPDATE visits SET visible_seconds=?, occluded_seconds=?, resume_count=? WHERE visit_id=?",
                (float(visible_seconds), float(occluded_seconds), int(resume_count), int(visit_id)))
            self._conn.commit()

    def visit_stats(self) -> dict:
        with self._lock:
            rows = self._execute(
                "SELECT visible_seconds + occluded_seconds AS tif FROM visits WHERE exit_ts IS NOT NULL"
            ).fetchall()
            open_n = int(self._scalar("SELECT COUNT(*) FROM visits WHERE exit_ts IS NULL"))
        dwell = []
        for r in rows:
            dwell.append(float((r["tif"] if isinstance(r, dict) else r[0]) or 0))
        return {
            "avg_time_in_frame_seconds": round(sum(dwell) / len(dwell), 1) if dwell else None,
            "closed_visits": len(dwell),
            "open_visits": open_n,
        }

    def open_visits(self) -> list:
        """Visits with no exit_ts (crash leftovers or in-progress)."""
        with self._lock:
            rows = self._execute(
                "SELECT visit_id, face_id, entry_ts, visible_seconds, occluded_seconds, "
                "resume_count FROM visits WHERE exit_ts IS NULL ORDER BY visit_id"
            ).fetchall()
        keys = ("visit_id", "face_id", "entry_ts", "visible_seconds",
                "occluded_seconds", "resume_count")
        out = []
        for r in rows:
            if isinstance(r, dict):
                out.append({k: r[k] for k in keys})
            else:
                out.append(dict(zip(keys, r)))
        return out

    def last_event_timestamp(self) -> Optional[str]:
        with self._lock:
            row = self._execute(
                "SELECT timestamp FROM events ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        return row["timestamp"] if isinstance(row, dict) else row[0]

    def visits_for_face(self, face_id: int) -> list:
        with self._lock:
            rows = self._execute(
                "SELECT visit_id, face_id, entry_ts, exit_ts, visible_seconds, occluded_seconds, "
                "resume_count, exit_reason FROM visits WHERE face_id=? ORDER BY visit_id",
                (face_id,)).fetchall()
        keys = ("visit_id", "face_id", "entry_ts", "exit_ts", "visible_seconds",
                "occluded_seconds", "resume_count", "exit_reason")
        out = []
        for r in rows:
            if isinstance(r, dict):
                out.append({k: r[k] for k in keys})
            else:
                out.append(dict(zip(keys, r)))
        return out

    # ---- reads ----
    def load_embeddings(self) -> List[Tuple[int, np.ndarray]]:
        with self._lock:
            rows = self._execute("SELECT face_id, vector FROM embeddings").fetchall()
        out = []
        for r in rows:
            if isinstance(r, dict):
                fid, blob = r["face_id"], r["vector"]
            else:
                fid, blob = r[0], r[1]
            if isinstance(blob, memoryview):
                blob = blob.tobytes()
            out.append((fid, np.frombuffer(blob, dtype=np.float32).copy()))
        return out

    def unique_count(self) -> int:
        with self._lock:
            return int(self._scalar("SELECT COUNT(*) FROM faces"))

    def event_counts(self) -> dict:
        with self._lock:
            rows = self._execute(
                "SELECT event_type, COUNT(*) AS c FROM events GROUP BY event_type").fetchall()
        out = {}
        for r in rows:
            if isinstance(r, dict):
                out[r["event_type"]] = r["c"]
            else:
                out[r[0]] = r[1]
        return out

    def query(self, sql: str, params: tuple = ()) -> list:
        with self._lock:
            rows = self._execute(sql, params).fetchall()
        # Normalize to tuples for callers that index by position.
        out = []
        for r in rows:
            if isinstance(r, dict):
                out.append(tuple(r.values()))
            else:
                out.append(tuple(r))
        return out

    def fetch_dicts(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._execute(sql, params)
            rows = cur.fetchall()
            # SQLite: recover column names from cursor description when needed.
            colnames = None
            if self.backend == "sqlite" and cur.description:
                colnames = [d[0] for d in cur.description]
        out: List[Dict[str, Any]] = []
        for r in rows:
            if isinstance(r, dict):
                out.append(dict(r))
            elif hasattr(r, "keys"):
                out.append({k: r[k] for k in r.keys()})
            elif colnames is not None:
                out.append(dict(zip(colnames, r)))
            else:
                raise TypeError(f"Cannot convert DB row: {type(r)}")
        return out

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass


def _dsn_from_parts(pg: dict) -> Optional[str]:
    host = pg.get("host")
    database = pg.get("database") or pg.get("dbname")
    if not host or not database:
        return None
    user = pg.get("user", "facetrack")
    password = pg.get("password", "")
    port = int(pg.get("port", 5432))
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


class _SqliteReadConn:
    def __init__(self, con: sqlite3.Connection):
        self._con = con

    def execute(self, sql: str, params: Sequence[Any] = ()):
        return self._con.execute(_adapt_sql(sql, "sqlite"), params)


class _PgReadConn:
    def __init__(self, con: Any):
        self._con = con

    def execute(self, sql: str, params: Sequence[Any] = ()):
        cur = self._con.cursor()
        cur.execute(_adapt_sql(sql, "postgres"), params)
        return cur
