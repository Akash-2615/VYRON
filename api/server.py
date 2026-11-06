"""Read-only dashboard API over the tracker's SQLite DB + image store.

    uvicorn api.server:app --port 8000          (run from the repo root)

Env overrides: TRACKER_CONFIG, TRACKER_DB, TRACKER_LOG_DIR  (handy for demo data).
If web/dist exists (npm run build) it is served at "/", so one process serves everything.
"""
import asyncio
import json
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

EPOCH = datetime(1970, 1, 1)
BUCKETS = [1, 5, 10, 30, 60, 300, 900, 3600, 21600, 86400]


def _secs(ts: str) -> float:
    return (datetime.fromisoformat(ts) - EPOCH).total_seconds()


LOG_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),\d+ \| (\w+)\s*\| (\S+)\s*(.*)$")
FIELD_RE = re.compile(r"(\w+)=(\S+)")


def create_app(db_path: str, log_dir: str, web_dist: Optional[str] = "web/dist",
               events_log: Optional[str] = None, match_threshold: float = 0.45,
               live_preview: Optional[str] = "output/live.jpg") -> FastAPI:
    app = FastAPI(title="Face Tracker Analytics API")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])
    logs = Path(log_dir).resolve()
    logs.mkdir(parents=True, exist_ok=True)
    log_file = Path(events_log) if events_log else logs / "events.log"
    live_path = Path(live_preview) if live_preview else None

    def connect() -> sqlite3.Connection:
        con = sqlite3.connect(db_path, timeout=5)
        con.execute("PRAGMA query_only=ON")
        con.row_factory = sqlite3.Row
        return con

    def media(p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        try:
            return "/media/" + Path(p).resolve().relative_to(logs).as_posix()
        except ValueError:
            return None

    def snapshot():
        """Load faces + events once and derive per-visitor stats, dwell times and occupancy."""
        if not Path(db_path).exists():
            return [], [], {}, []
        with connect() as con:
            faces = [dict(r) for r in con.execute("SELECT * FROM faces ORDER BY face_id")]
            events = [dict(r) for r in con.execute("SELECT * FROM events ORDER BY timestamp, id")]
        stats = {f["face_id"]: {"total_dwell": 0.0, "inside": False, "thumb": None, "latest": None,
                                "entry_ts": None} for f in faces}
        dwell, inside, peak = [], set(), 0
        for e in events:
            s = stats.get(e["face_id"])
            if s is None:
                continue
            img = media(e["image_path"])
            if e["event_type"] == "entry":
                inside.add(e["face_id"])
                peak = max(peak, len(inside))
                s["inside"], s["entry_ts"] = True, e["timestamp"]
                s["thumb"] = s["thumb"] or img
            else:
                inside.discard(e["face_id"])
                s["inside"] = False
                if s["entry_ts"]:
                    d = _secs(e["timestamp"]) - _secs(s["entry_ts"])
                    s["total_dwell"] += d
                    dwell.append(d)
                    s["entry_ts"] = None
            s["latest"] = img or s["latest"]
        return faces, events, stats, dwell + [peak]  # peak travels as last element

    def visitor_row(f: dict, s: dict) -> dict:
        appearance = None
        raw = f.get("appearance_json")
        if raw:
            try:
                appearance = json.loads(raw)
            except (TypeError, ValueError):
                appearance = None
        return {"face_id": f["face_id"], "first_seen": f["first_seen"], "last_seen": f["last_seen"],
                "visit_count": f["visit_count"], "total_dwell_seconds": round(s["total_dwell"], 1),
                "status": "inside" if s["inside"] else "left",
                "face_image": media(f.get("image_path")),
                "thumbnail": media(f.get("image_path")) or s["thumb"] or s["latest"],
                "latest_image": s["latest"],
                "appearance_summary": f.get("appearance_summary"),
                "appearance": appearance}

    @app.get("/api/health")
    def health():
        return {"ok": True, "db_exists": Path(db_path).exists()}

    @app.get("/api/live/status")
    def live_status():
        """Whether main.py is publishing annotated frames for the dashboard."""
        if live_path is None:
            return {"active": False, "path": None, "age_sec": None}
        if not live_path.is_file():
            return {"active": False, "path": str(live_path), "age_sec": None}
        age = time.time() - live_path.stat().st_mtime
        return {"active": age < 5.0, "path": str(live_path), "age_sec": round(age, 2),
                "bytes": live_path.stat().st_size}

    @app.get("/api/live.jpg")
    def live_jpeg():
        """Latest annotated frame (single JPEG). Prefer /api/live for continuous MJPEG."""
        if live_path is None or not live_path.is_file():
            raise HTTPException(404, "Live preview not available — is main.py running?")
        return Response(live_path.read_bytes(), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/live")
    async def live_mjpeg():
        """MJPEG stream of the annotated monitor view (boxes + confidence)."""
        boundary = b"frame"

        async def frames():
            last_mtime = -1.0
            while True:
                if live_path is not None and live_path.is_file():
                    try:
                        mtime = live_path.stat().st_mtime
                        if mtime != last_mtime:
                            data = live_path.read_bytes()
                            last_mtime = mtime
                            yield (b"--" + boundary + b"\r\nContent-Type: image/jpeg\r\n"
                                   b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                                   + data + b"\r\n")
                    except OSError:
                        pass
                await asyncio.sleep(0.08)

        return StreamingResponse(
            frames(),
            media_type=f"multipart/x-mixed-replace; boundary={boundary.decode()}",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                     "Pragma": "no-cache", "Connection": "keep-alive"},
        )

    def _away_rows():
        if not Path(db_path).exists():
            return []
        with connect() as con:
            try:
                return [dict(r) for r in con.execute(
                    "SELECT * FROM away_visitors ORDER BY away_id DESC")]
            except sqlite3.OperationalError:
                return []

    def _away_stats_sql():
        if not Path(db_path).exists():
            return {"away_unique": 0, "away_entries": 0, "away_exits": 0,
                    "away_currently": 0, "away_with_clothes": 0}
        with connect() as con:
            try:
                unique = con.execute("SELECT COUNT(*) FROM away_visitors").fetchone()[0]
                rows = con.execute(
                    "SELECT event_type, COUNT(*) FROM away_events GROUP BY event_type").fetchall()
                clothed = con.execute(
                    "SELECT COUNT(*) FROM away_visitors WHERE appearance_summary IS NOT NULL "
                    "AND appearance_summary != ''").fetchone()[0]
            except sqlite3.OperationalError:
                return {"away_unique": 0, "away_entries": 0, "away_exits": 0,
                        "away_currently": 0, "away_with_clothes": 0}
        counts = {r[0]: r[1] for r in rows}
        entries, exits = counts.get("entry", 0), counts.get("exit", 0)
        return {"away_unique": unique, "away_entries": entries, "away_exits": exits,
                "away_currently": max(0, entries - exits), "away_with_clothes": clothed}

    @app.get("/api/summary")
    def summary():
        faces, events, stats, extra = snapshot()
        dwell, peak = (extra[:-1], extra[-1]) if extra else ([], 0)
        entries = sum(1 for e in events if e["event_type"] == "entry")
        away = _away_stats_sql()
        return {
            "unique_visitors": len(faces),
            "total_entries": entries,
            "total_exits": len(events) - entries,
            "currently_inside": sum(1 for s in stats.values() if s["inside"]),
            "returning_visitors": sum(1 for f in faces if f["visit_count"] > 1),
            "avg_dwell_seconds": round(sum(dwell) / len(dwell), 1) if dwell else None,
            "peak_occupancy": peak,
            "first_event": events[0]["timestamp"] if events else None,
            "last_event": events[-1]["timestamp"] if events else None,
            **away,
        }

    @app.get("/api/away-visitors")
    def away_visitors(limit: int = Query(40, ge=1, le=200)):
        rows = []
        for r in _away_rows()[:limit]:
            appearance = None
            raw = r.get("appearance_json")
            if raw:
                try:
                    appearance = json.loads(raw)
                except (TypeError, ValueError):
                    appearance = None
            rows.append({
                "away_id": r["away_id"],
                "first_seen": r["first_seen"],
                "last_seen": r["last_seen"],
                "visit_count": r["visit_count"],
                "appearance_summary": r.get("appearance_summary"),
                "appearance": appearance,
                "thumbnail": media(r.get("image_path")),
            })
        return rows

    @app.get("/api/timeline")
    def timeline():
        _, events, _, _ = snapshot()
        if not events:
            return {"bucket_seconds": 60, "points": []}
        t0, t1 = _secs(events[0]["timestamp"]), _secs(events[-1]["timestamp"])
        span = max(t1 - t0, 1)
        size = next((b for b in BUCKETS if span / b <= 60), BUCKETS[-1])
        start = int(t0 // size) * size
        n = int((t1 - start) // size) + 1
        entries, exits = [0] * n, [0] * n
        for e in events:
            i = int((_secs(e["timestamp"]) - start) // size)
            (entries if e["event_type"] == "entry" else exits)[i] += 1
        pts, occ = [], 0
        for i in range(n):
            occ = max(0, occ + entries[i] - exits[i])
            pts.append({"t": (EPOCH + timedelta(seconds=start + i * size)).isoformat(),
                        "entries": entries[i], "exits": exits[i], "occupancy": occ})
        return {"bucket_seconds": size, "points": pts}

    @app.get("/api/visitors")
    def visitors():
        faces, _, stats, _ = snapshot()
        rows = [visitor_row(f, stats[f["face_id"]]) for f in faces]
        return sorted(rows, key=lambda r: r["last_seen"], reverse=True)

    @app.get("/api/visitors/{face_id}")
    def visitor(face_id: int):
        faces, events, stats, _ = snapshot()
        f = next((f for f in faces if f["face_id"] == face_id), None)
        if f is None:
            raise HTTPException(404, "Visitor not found")
        evs = [{"id": e["id"], "event_type": e["event_type"], "timestamp": e["timestamp"],
                "image": media(e["image_path"]), "note": e["note"]}
               for e in events if e["face_id"] == face_id]
        return {**visitor_row(f, stats[face_id]), "events": evs}

    @app.get("/api/events")
    def list_events(type: Optional[str] = Query(None, pattern="^(entry|exit)$"),
                    face_id: Optional[int] = None, limit: int = Query(25, ge=1, le=200), offset: int = 0):
        if not Path(db_path).exists():
            return {"total": 0, "items": []}
        where, params = [], []
        if type:
            where.append("event_type=?"); params.append(type)
        if face_id is not None:
            where.append("face_id=?"); params.append(face_id)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        with connect() as con:
            total = con.execute(f"SELECT COUNT(*) FROM events {clause}", params).fetchone()[0]
            rows = con.execute(f"SELECT * FROM events {clause} ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?",
                               params + [limit, offset]).fetchall()
        items = [{"id": r["id"], "face_id": r["face_id"], "event_type": r["event_type"],
                  "timestamp": r["timestamp"], "image": media(r["image_path"]),
                  "track_id": r["track_id"], "frame_idx": r["frame_idx"], "note": r["note"]} for r in rows]
        return {"total": total, "items": items}

    @app.get("/api/visitors/{face_id}/embeddings")
    def visitor_embeddings(face_id: int):
        if not Path(db_path).exists():
            raise HTTPException(404, "No database yet")
        with connect() as con:
            rows = con.execute("SELECT id, face_id, vector, created_at, image_path FROM embeddings ORDER BY id").fetchall()
            face_imgs = {r["face_id"]: r["image_path"] for r in con.execute("SELECT face_id, image_path FROM faces")}
        if face_id not in face_imgs:
            raise HTTPException(404, "Visitor not found")
        by = defaultdict(list)
        for r in rows:
            by[r["face_id"]].append((r, np.frombuffer(r["vector"], dtype=np.float32)))

        def unit(v: np.ndarray) -> np.ndarray:
            return v / max(float(np.linalg.norm(v)), 1e-9)

        means = {fid: unit(np.mean([v for _, v in items], axis=0)) for fid, items in by.items()}
        samples = [{"id": r["id"], "created_at": r["created_at"], "image": media(r["image_path"]),
                    "norm": round(float(np.linalg.norm(v)), 4), "vector": [round(float(x), 5) for x in v]}
                   for r, v in by.get(face_id, [])]
        sims = None
        vecs = [unit(v) for _, v in by.get(face_id, [])]
        if len(vecs) > 1:
            m = np.stack(vecs) @ np.stack(vecs).T
            sims = round(float((m.sum() - len(vecs)) / (len(vecs) * (len(vecs) - 1))), 4)
        mine = means.get(face_id)
        similar = []
        if mine is not None:
            similar = sorted(({"face_id": fid, "similarity": round(float(mine @ m), 4),
                               "thumbnail": media(face_imgs.get(fid))}
                              for fid, m in means.items() if fid != face_id),
                             key=lambda d: d["similarity"], reverse=True)[:6]
        return {"face_id": face_id, "dim": int(mine.shape[0]) if mine is not None else 0,
                "count": len(samples), "match_threshold": match_threshold, "intra_similarity": sims,
                "face_image": media(face_imgs.get(face_id)),
                "mean": [round(float(x), 5) for x in mine] if mine is not None else [],
                "samples": samples, "similar": similar}

    @app.get("/api/logs")
    def logs_tail(limit: int = Query(300, ge=1, le=2000), event: Optional[str] = None,
                  level: Optional[str] = None, q: Optional[str] = None):
        """Parsed tail of events.log, newest first."""
        if not log_file.exists():
            return {"path": str(log_file), "exists": False, "total": 0, "counts": {}, "items": []}
        size = log_file.stat().st_size
        with open(log_file, "rb") as f:
            f.seek(max(0, size - 2_000_000))
            lines = f.read().decode("utf-8", errors="replace").splitlines()
        if size > 2_000_000:
            lines = lines[1:]                      # first line may be cut in half
        parsed = []
        for line in lines:
            m = LOG_RE.match(line)
            if not m:
                continue
            day, clock, lvl, ev, rest = m.groups()
            parsed.append({"ts": f"{day}T{clock}", "level": lvl, "event": ev, "message": rest,
                           "fields": dict(FIELD_RE.findall(rest))})
        counts = Counter(p["event"] for p in parsed)
        needle = q.lower() if q else None
        items = [p for p in reversed(parsed)
                 if (not event or p["event"] == event) and (not level or p["level"] == level)
                 and (not needle or needle in (p["event"] + " " + p["message"]).lower())]
        return {"path": str(log_file), "exists": True, "total": len(items), "counts": dict(counts),
                "items": items[:limit]}

    app.mount("/media", StaticFiles(directory=str(logs)), name="media")
    if web_dist and Path(web_dist).is_dir():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
    return app


def _default_app() -> FastAPI:
    from src.config import load_config
    cfg = load_config(os.environ.get("TRACKER_CONFIG", "config.json"))
    log_dir = os.environ.get("TRACKER_LOG_DIR", cfg["storage"]["log_dir"])
    default_log = cfg["storage"]["events_log"] if "TRACKER_LOG_DIR" not in os.environ else str(Path(log_dir) / "events.log")
    return create_app(os.environ.get("TRACKER_DB", cfg["storage"]["db_path"]), log_dir,
                      events_log=os.environ.get("TRACKER_EVENTS_LOG", default_log),
                      match_threshold=cfg["recognition"]["match_threshold"],
                      live_preview=os.environ.get(
                          "TRACKER_LIVE", cfg["output"].get("live_preview", "output/live.jpg")))


app = _default_app()
