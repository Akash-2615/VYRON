"""SQL DDL / drop scripts for visitor storage (SQLite + Postgres).

Schema changes must stay in sync with Database migrations. Do not change
column names or semantics without a coordinated migration.
"""
from __future__ import annotations

# Shared logical schema. Dialects differ only in PK / BLOB types.
_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS faces (
    face_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL,
    visit_count INTEGER NOT NULL DEFAULT 0,
    image_path  TEXT,
    appearance_summary TEXT,
    appearance_json    TEXT,
    appearance_image_path TEXT
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
    note       TEXT,
    visit_id   INTEGER
);
CREATE TABLE IF NOT EXISTS visits (
    visit_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id           INTEGER NOT NULL REFERENCES faces(face_id),
    entry_ts          TEXT NOT NULL,
    exit_ts           TEXT,
    visible_seconds   REAL NOT NULL DEFAULT 0,
    occluded_seconds  REAL NOT NULL DEFAULT 0,
    resume_count      INTEGER NOT NULL DEFAULT 0,
    exit_reason       TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_face ON events(face_id);
CREATE INDEX IF NOT EXISTS idx_emb_face ON embeddings(face_id);
CREATE INDEX IF NOT EXISTS idx_visits_face ON visits(face_id);

CREATE TABLE IF NOT EXISTS away_visitors (
    away_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    first_seen          TEXT NOT NULL,
    last_seen           TEXT NOT NULL,
    visit_count         INTEGER NOT NULL DEFAULT 0,
    image_path          TEXT,
    appearance_summary  TEXT,
    appearance_json     TEXT,
    linked_face_id      INTEGER
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

_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS faces (
    face_id     SERIAL PRIMARY KEY,
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL,
    visit_count INTEGER NOT NULL DEFAULT 0,
    image_path  TEXT,
    appearance_summary TEXT,
    appearance_json    TEXT,
    appearance_image_path TEXT
);
CREATE TABLE IF NOT EXISTS embeddings (
    id         SERIAL PRIMARY KEY,
    face_id    INTEGER NOT NULL REFERENCES faces(face_id),
    vector     BYTEA NOT NULL,
    created_at TEXT NOT NULL,
    image_path TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id         SERIAL PRIMARY KEY,
    face_id    INTEGER NOT NULL REFERENCES faces(face_id),
    event_type TEXT NOT NULL CHECK (event_type IN ('entry', 'exit')),
    timestamp  TEXT NOT NULL,
    image_path TEXT,
    track_id   INTEGER,
    frame_idx  INTEGER,
    note       TEXT,
    visit_id   INTEGER
);
CREATE TABLE IF NOT EXISTS visits (
    visit_id          SERIAL PRIMARY KEY,
    face_id           INTEGER NOT NULL REFERENCES faces(face_id),
    entry_ts          TEXT NOT NULL,
    exit_ts           TEXT,
    visible_seconds   DOUBLE PRECISION NOT NULL DEFAULT 0,
    occluded_seconds  DOUBLE PRECISION NOT NULL DEFAULT 0,
    resume_count      INTEGER NOT NULL DEFAULT 0,
    exit_reason       TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_face ON events(face_id);
CREATE INDEX IF NOT EXISTS idx_emb_face ON embeddings(face_id);
CREATE INDEX IF NOT EXISTS idx_visits_face ON visits(face_id);

CREATE TABLE IF NOT EXISTS away_visitors (
    away_id             SERIAL PRIMARY KEY,
    first_seen          TEXT NOT NULL,
    last_seen           TEXT NOT NULL,
    visit_count         INTEGER NOT NULL DEFAULT 0,
    image_path          TEXT,
    appearance_summary  TEXT,
    appearance_json     TEXT,
    linked_face_id      INTEGER
);
CREATE TABLE IF NOT EXISTS away_events (
    id         SERIAL PRIMARY KEY,
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

_PG_DROP = """
DROP TABLE IF EXISTS away_events CASCADE;
DROP TABLE IF EXISTS away_visitors CASCADE;
DROP TABLE IF EXISTS events CASCADE;
DROP TABLE IF EXISTS visits CASCADE;
DROP TABLE IF EXISTS embeddings CASCADE;
DROP TABLE IF EXISTS faces CASCADE;
"""


