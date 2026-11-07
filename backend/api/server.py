"""Read-only dashboard API over the tracker DB + image store.

    uvicorn api.server:app --app-dir backend --port 8000  (repo root)

Requires DATABASE_URL / TRACKER_DSN (local Postgres). Face crops and
events.log stay under the local logs/ folder. Env: TRACKER_CONFIG,
TRACKER_LOG_DIR, TRACKER_EVENTS_LOG, GROQ_*. If frontend/dist exists it is served
at "/". Unit tests may inject a SQLite Database via create_app(db=...).

Route groups inside ``create_app`` (closures over DB/log paths — no extra hops):
  health / videos / tracker job / live preview / count+summary /
  away visitors / timeline+visitors+events / embeddings / logs / static mounts
Analytics helpers live in ``api.services``; assistant chat in ``api.assistant``.
"""
import asyncio
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.storage import Database

EPOCH = datetime(1970, 1, 1)
BUCKETS = [1, 5, 10, 30, 60, 300, 900, 3600, 21600, 86400]


def _secs(ts) -> float:
    """Parse ISO timestamps from SQLite/Postgres TEXT (or datetime objects)."""
    if ts is None:
        return 0.0
    if isinstance(ts, datetime):
        return (ts.replace(tzinfo=None) - EPOCH).total_seconds()
    s = str(ts).replace("Z", "")
    # Postgres sometimes returns "YYYY-MM-DD HH:MM:SS[.mmm]"
    if " " in s and "T" not in s:
        s = s.replace(" ", "T", 1)
    return (datetime.fromisoformat(s) - EPOCH).total_seconds()


def _as_dict(row: Any) -> dict:
    if isinstance(row, dict):
        return dict(row)
    if hasattr(row, "keys"):
        return {k: row[k] for k in row.keys()}
    raise TypeError(f"Cannot convert row: {type(row)}")


def _blob(val: Any) -> bytes:
    if isinstance(val, memoryview):
        return val.tobytes()
    if isinstance(val, bytearray):
        return bytes(val)
    return val


LOG_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),\d+ \| (\w+)\s*\| (\S+)\s*(.*)$")
FIELD_RE = re.compile(r"(\w+)=(\S+)")


def create_app(db_path: Optional[str] = None, log_dir: str = "logs",
               web_dist: Optional[str] = "frontend/dist",
               events_log: Optional[str] = None, match_threshold: float = 0.45,
               live_preview: Optional[str] = "data/runtime/live.jpg",
               storage: Optional[dict] = None,
               db: Optional[Database] = None) -> FastAPI:
    app = FastAPI(title="Face Tracker Analytics API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    from src.core.paths import PROJECT_ROOT, resolve_under_root
    if web_dist and not Path(web_dist).is_absolute():
        web_dist = str(resolve_under_root(web_dist))
    if log_dir and not Path(log_dir).is_absolute():
        log_dir = str(resolve_under_root(log_dir))
    if live_preview and not Path(live_preview).is_absolute():
        live_preview = str(resolve_under_root(live_preview))
    if events_log and not Path(events_log).is_absolute():
        events_log = str(resolve_under_root(events_log))
    logs = Path(log_dir).resolve()
    logs.mkdir(parents=True, exist_ok=True)
    log_file = Path(events_log) if events_log else logs / "events.log"
    live_path = Path(live_preview) if live_preview else None

    if db is None:
        if storage is not None:
            db = Database.from_storage(storage, reset=False)
        elif db_path:
            # Explicit file path → SQLite (unit tests / create_app helpers only).
            db = Database(path=db_path, backend="sqlite", reset=False)
        else:
            raise ValueError("create_app requires db, storage, or db_path")

    def connect():
        return db.readonly()

    def media(p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        try:
            return "/media/" + Path(p).resolve().relative_to(logs).as_posix()
        except ValueError:
            return None

    def snapshot():
        """Load faces + events once and derive per-visitor stats, dwell times and occupancy."""
        if not db.exists():
            return [], [], {}, []
        try:
            with connect() as con:
                faces = [_as_dict(r) for r in con.execute("SELECT * FROM faces ORDER BY face_id")]
                events = [_as_dict(r) for r in con.execute("SELECT * FROM events ORDER BY timestamp, id")]
        except Exception:  # noqa: BLE001
            return [], [], {}, []
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

    def _visit_metrics(face_id: int) -> dict:
        empty = {"time_in_frame_seconds": 0.0, "visible_seconds": 0.0,
                 "occluded_seconds": 0.0, "resume_count": 0, "hidden_now": False, "visits": []}
        if not db.exists():
            return empty
        try:
            with connect() as con:
                rows = [_as_dict(r) for r in con.execute(
                    "SELECT visit_id, face_id, entry_ts, exit_ts, visible_seconds, occluded_seconds, "
                    "resume_count, exit_reason FROM visits WHERE face_id=? ORDER BY visit_id",
                    (face_id,))]
        except Exception:  # noqa: BLE001
            return empty
        vis = sum(float(r["visible_seconds"] or 0) for r in rows)
        occ = sum(float(r["occluded_seconds"] or 0) for r in rows)
        resumes = sum(int(r["resume_count"] or 0) for r in rows)
        open_rows = [r for r in rows if not r.get("exit_ts")]
        return {
            "time_in_frame_seconds": round(vis + occ, 1),
            "visible_seconds": round(vis, 1),
            "occluded_seconds": round(occ, 1),
            "resume_count": resumes,
            "hidden_now": bool(open_rows),
            "visits": [{
                "visit_id": r["visit_id"], "entry_ts": r["entry_ts"], "exit_ts": r["exit_ts"],
                "visible_seconds": r["visible_seconds"], "occluded_seconds": r["occluded_seconds"],
                "resume_count": r["resume_count"], "exit_reason": r["exit_reason"],
            } for r in rows],
        }

    def visitor_row(f: dict, s: dict) -> dict:
        appearance = None
        raw = f.get("appearance_json")
        if raw:
            try:
                appearance = json.loads(raw)
            except (TypeError, ValueError):
                appearance = None
        evidence = None
        if appearance and isinstance(appearance, dict):
            evidence = appearance.get("evidence_image")
        evidence = f.get("appearance_image_path") or evidence
        metrics = _visit_metrics(int(f["face_id"]))
        hidden_now = bool(metrics["visits"]) and any(v["exit_ts"] is None for v in metrics["visits"]) and s["inside"]
        dwell = metrics["time_in_frame_seconds"] or round(s["total_dwell"], 1)
        return {"face_id": f["face_id"], "first_seen": f["first_seen"], "last_seen": f["last_seen"],
                "visit_count": f["visit_count"], "total_dwell_seconds": dwell,
                "time_in_frame_seconds": metrics["time_in_frame_seconds"],
                "visible_seconds": metrics["visible_seconds"],
                "occluded_seconds": metrics["occluded_seconds"],
                "resume_count": metrics["resume_count"],
                "hidden_now": hidden_now,
                "status": "inside" if s["inside"] else "left",
                "face_image": media(f.get("image_path")),
                "thumbnail": media(f.get("image_path")) or s["thumb"] or s["latest"],
                "latest_image": s["latest"],
                "appearance_summary": f.get("appearance_summary"),
                "appearance_image": media(evidence),
                "appearance": appearance}

    # --- Health & local video catalogue ------------------------------------------
    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "db_exists": db.exists(),
            "backend": db.backend,
            "fallback": False,
        }

    @app.get("/api/videos")
    def list_videos():
        """Local videos from config.json (video_sources) for the dashboard picker."""
        from src.pipeline.tracker_job import list_local_videos
        cfg_path = os.environ.get("TRACKER_CONFIG", "config.json")
        items = list_local_videos(cfg_path)
        return {
            "items": items,
            "available": [v for v in items if v.get("exists")],
            "count": len(items),
            "available_count": sum(1 for v in items if v.get("exists")),
        }

    # --- Tracker job control (dashboard Start / Stop) ----------------------------
    @app.get("/api/tracker/status")
    def tracker_status():
        """Background RTSP tracker job state (started from the dashboard)."""
        from src.pipeline.tracker_job import get_tracker_job
        return get_tracker_job().status()

    @app.post("/api/tracker/start")
    def tracker_start(payload: dict = Body(default_factory=dict)):
        """Start detect → track → embed → log on RTSP, one local file, or all video_sources."""
        from src.pipeline.tracker_job import get_tracker_job
        body = payload or {}
        url = str(body.get("url") or body.get("path") or "").strip()
        mode = str(body.get("mode") or "rtsp").strip().lower()
        playlist = str(body.get("playlist") or "one").strip().lower()
        reset = bool(body.get("reset", False))
        raw_sources = body.get("sources")
        sources = None
        if isinstance(raw_sources, list):
            sources = [str(s).strip() for s in raw_sources if str(s).strip()]
        if mode in ("local", "file", "video", "mp4") and playlist in ("all", "playlist", "sources"):
            # url optional when playing the full configured playlist
            pass
        elif mode in ("webcam", "cam", "camera", "device"):
            url = url or "0"
        elif not url and not sources:
            raise HTTPException(400, "Paste an RTSP URL, choose a local video, or pick a webcam")
        try:
            st = get_tracker_job().start(
                url,
                config_path=os.environ.get("TRACKER_CONFIG", "config.json"),
                reset=reset,
                mode=mode,
                playlist=playlist,
                sources=sources,
            )
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e
        return {"ok": True, **st}

    @app.post("/api/tracker/stop")
    def tracker_stop():
        from src.pipeline.tracker_job import get_tracker_job
        st = get_tracker_job().stop()
        return {"ok": True, **st}

    # --- Live preview JPEG + stream status ---------------------------------------
    @app.get("/api/live/status")
    def live_status():
        """Whether main.py is publishing annotated frames + in-frame person counts."""
        empty = {"active": False, "path": None, "age_sec": None,
                 "persons_in_frame": None, "faces_in_frame": None, "no_face_in_frame": None,
                 "fps": None, "latency_ms": None, "width": None, "height": None}
        if live_path is None:
            return empty
        if not live_path.is_file():
            return {**empty, "path": str(live_path)}
        age = time.time() - live_path.stat().st_mtime
        out = {"active": age < 5.0, "path": str(live_path), "age_sec": round(age, 2),
               "bytes": live_path.stat().st_size,
               "persons_in_frame": None, "faces_in_frame": None, "no_face_in_frame": None,
               "fps": None, "latency_ms": int(round(age * 1000)) if age < 5 else None,
               "width": None, "height": None}
        stats_file = live_path.with_name("live_stats.json")
        if stats_file.is_file():
            try:
                import json as _json
                st = _json.loads(stats_file.read_text(encoding="utf-8"))
                out["persons_in_frame"] = st.get("persons")
                out["faces_in_frame"] = st.get("faces")
                out["faces_confirmed"] = st.get("faces_confirmed")
                out["no_face_in_frame"] = st.get("no_face")
                out["faces_turned_away"] = st.get("faces_turned_away")
                out["fps"] = st.get("fps")
                # Prefer preview size when live JPEG is downscaled for the dashboard.
                out["width"] = st.get("preview_width") or st.get("width")
                out["height"] = st.get("preview_height") or st.get("height")
                if st.get("latency_ms") is not None:
                    out["latency_ms"] = st.get("latency_ms")
                if st.get("ts"):
                    out["stats_age_sec"] = round(time.time() - float(st["ts"]), 2)
            except (OSError, ValueError, TypeError):
                pass
        return out

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
                updated = False
                if live_path is not None and live_path.is_file():
                    try:
                        mtime = live_path.stat().st_mtime
                        if mtime != last_mtime:
                            data = live_path.read_bytes()
                            last_mtime = mtime
                            updated = True
                            yield (b"--" + boundary + b"\r\nContent-Type: image/jpeg\r\n"
                                   b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                                   + data + b"\r\n")
                    except OSError:
                        pass
                # Poll faster right after a new frame; idle a bit longer otherwise.
                await asyncio.sleep(0.03 if updated else 0.06)

        return StreamingResponse(
            frames(),
            media_type=f"multipart/x-mixed-replace; boundary={boundary.decode()}",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                     "Pragma": "no-cache", "Connection": "keep-alive"},
        )

    def _away_rows():
        if not db.exists():
            return []
        try:
            with connect() as con:
                return [_as_dict(r) for r in con.execute(
                    "SELECT * FROM away_visitors ORDER BY away_id DESC")]
        except Exception:  # noqa: BLE001
            return []

    def _away_stats_sql():
        if not db.exists():
            return {"away_unique": 0, "away_entries": 0, "away_exits": 0,
                    "away_currently": 0, "away_with_clothes": 0}
        try:
            with connect() as con:
                unique = con.execute("SELECT COUNT(*) AS c FROM away_visitors").fetchone()
                unique = int(_as_dict(unique)["c"] if isinstance(unique, dict) or hasattr(unique, "keys")
                             else unique[0])
                rows = con.execute(
                    "SELECT event_type, COUNT(*) AS c FROM away_events GROUP BY event_type").fetchall()
                clothed_row = con.execute(
                    "SELECT COUNT(*) AS c FROM away_visitors WHERE appearance_summary IS NOT NULL "
                    "AND appearance_summary != ''").fetchone()
                clothed = int(_as_dict(clothed_row)["c"] if isinstance(clothed_row, dict)
                              or hasattr(clothed_row, "keys") else clothed_row[0])
        except Exception:  # noqa: BLE001
            return {"away_unique": 0, "away_entries": 0, "away_exits": 0,
                    "away_currently": 0, "away_with_clothes": 0}
        counts = {}
        for r in rows:
            d = _as_dict(r) if not isinstance(r, tuple) else None
            if d is not None and "event_type" in d:
                counts[d["event_type"]] = int(d["c"])
            else:
                counts[r[0]] = int(r[1])
        entries, exits = counts.get("entry", 0), counts.get("exit", 0)
        return {"away_unique": unique, "away_entries": entries, "away_exits": exits,
                "away_currently": max(0, entries - exits), "away_with_clothes": clothed}

    _count_cache: dict = {"t": 0.0, "key": "", "val": None}

    # --- Counts, summary, visitors, events ---------------------------------------
    @app.get("/api/count")
    def api_count(session: str = Query("latest")):
        """Unique visitor count from DB (preferred) with log reconciliation.

        Cached for 2 seconds. Falls back to the log when the DB is missing.
        """
        from src.reporting import count_from_database, count_from_log, reconcile
        now = time.time()
        key = session or "latest"
        if (_count_cache["key"] == key and _count_cache["val"] is not None
                and now - _count_cache["t"] < 2.0):
            return _count_cache["val"]
        db_counts = count_from_database(db)
        log_counts = count_from_log(log_file, session=key, include_rotated=True)
        result = reconcile(
            db_counts if db_counts.available else None,
            log_counts if log_counts.available else None,
        )
        if db_counts.available:
            primary, source = db_counts, "db"
        elif log_counts.available:
            primary, source = log_counts, "log"
        else:
            primary, source = None, "none"
        payload = {
            "unique_visitors": primary.unique_visitors if primary else None,
            "entries": primary.entries if primary else None,
            "exits": primary.exits if primary else None,
            "currently_inside": primary.open_visits if primary else None,
            "returning_visitors": primary.returning_visitors if primary else None,
            "open_visits": primary.open_visits if primary else None,
            "source": source,
            "derived_from_log": log_counts.to_dict() if log_counts.available else None,
            "consistent": result.get("consistent"),
            "diffs": result.get("diffs", []),
            "notes": result.get("notes", []),
            "session_id": log_counts.session_id if log_counts.available else None,
            "as_of": datetime.now().isoformat(timespec="seconds"),
        }
        _count_cache.update(t=now, key=key, val=payload)
        return payload

    @app.get("/api/summary")
    def summary():
        faces, events, stats, extra = snapshot()
        dwell, peak = (extra[:-1], extra[-1]) if extra else ([], 0)
        entries = sum(1 for e in events if e["event_type"] == "entry")
        away = _away_stats_sql()
        inside = sum(1 for s in stats.values() if s["inside"])
        hidden_now = 0
        avg_tif = None
        if db.exists():
            try:
                with connect() as con:
                    rows = con.execute(
                        "SELECT visible_seconds + occluded_seconds AS tif FROM visits "
                        "WHERE exit_ts IS NOT NULL"
                    ).fetchall()
                    if rows:
                        vals = []
                        for r in rows:
                            d = _as_dict(r) if not isinstance(r, tuple) else None
                            vals.append(float((d["tif"] if d else r[0]) or 0))
                        avg_tif = round(sum(vals) / len(vals), 1)
                    open_row = con.execute(
                        "SELECT COUNT(*) AS c FROM visits WHERE exit_ts IS NULL").fetchone()
                    open_n = int(_as_dict(open_row)["c"] if isinstance(open_row, dict)
                                 or hasattr(open_row, "keys") else open_row[0])
                    inside = max(inside, int(open_n))
                    hidden_now = max(0, int(open_n) - sum(
                        1 for s in stats.values() if s["inside"] and s.get("entry_ts")))
            except Exception:  # noqa: BLE001
                pass
        return {
            "unique_visitors": len(faces),
            "total_entries": entries,
            "total_exits": len(events) - entries,
            "currently_inside": inside,
            "hidden_now": hidden_now,
            "returning_visitors": sum(1 for f in faces if f["visit_count"] > 1),
            "avg_dwell_seconds": round(sum(dwell) / len(dwell), 1) if dwell else None,
            "avg_time_in_frame_seconds": avg_tif if avg_tif is not None else (
                round(sum(dwell) / len(dwell), 1) if dwell else None),
            "peak_occupancy": peak,
            "first_event": events[0]["timestamp"] if events else None,
            "last_event": events[-1]["timestamp"] if events else None,
            **away,
        }

    @app.get("/api/away-visitors")
    def away_visitors(limit: int = Query(40, ge=1, le=200)):
        rows = []
        last_evt: dict = {}
        if db.exists():
            try:
                with connect() as con:
                    for er in con.execute(
                        "SELECT away_id, event_type FROM away_events ORDER BY id"
                    ):
                        d = _as_dict(er)
                        last_evt[int(d["away_id"])] = d["event_type"]
            except Exception:  # noqa: BLE001
                pass
        for r in _away_rows()[:limit]:
            appearance = None
            raw = r.get("appearance_json")
            if raw:
                try:
                    appearance = json.loads(raw)
                except (TypeError, ValueError):
                    appearance = None
            evidence = None
            if appearance and isinstance(appearance, dict):
                evidence = appearance.get("evidence_image")
            aid = int(r["away_id"])
            status = "inside" if last_evt.get(aid) == "entry" else "left"
            rows.append({
                "away_id": aid,
                "first_seen": r["first_seen"],
                "last_seen": r["last_seen"],
                "visit_count": r["visit_count"],
                "status": status,
                "appearance_summary": r.get("appearance_summary"),
                "appearance": appearance,
                "appearance_image": media(evidence or r.get("image_path")),
                "thumbnail": media(r.get("image_path")),
                "linked_face_id": r.get("linked_face_id"),
            })
        return rows

    @app.get("/api/away-visitors/{away_id}")
    def away_visitor(away_id: int):
        """Detail for one no-face / away walker (appearance + entry/exit timeline)."""
        row = next((r for r in _away_rows() if int(r["away_id"]) == int(away_id)), None)
        if row is None:
            raise HTTPException(404, "Away visitor not found")
        appearance = None
        raw = row.get("appearance_json")
        if raw:
            try:
                appearance = json.loads(raw)
            except (TypeError, ValueError):
                appearance = None
        evidence = None
        if appearance and isinstance(appearance, dict):
            evidence = appearance.get("evidence_image")
        events = []
        last_evt = None
        if db.exists():
            try:
                with connect() as con:
                    for er in con.execute(
                        "SELECT * FROM away_events WHERE away_id=? ORDER BY timestamp, id",
                        (int(away_id),),
                    ):
                        d = _as_dict(er)
                        last_evt = d["event_type"]
                        events.append({
                            "id": d["id"],
                            "event_type": d["event_type"],
                            "timestamp": d["timestamp"],
                            "image": media(d.get("image_path")),
                            "note": d.get("note"),
                            "track_id": d.get("track_id"),
                            "frame_idx": d.get("frame_idx"),
                        })
            except Exception:  # noqa: BLE001
                pass
        status = "inside" if last_evt == "entry" else "left"
        return {
            "away_id": int(row["away_id"]),
            "first_seen": row["first_seen"],
            "last_seen": row["last_seen"],
            "visit_count": row["visit_count"],
            "status": status,
            "appearance_summary": row.get("appearance_summary"),
            "appearance": appearance,
            "appearance_image": media(evidence or row.get("image_path")),
            "thumbnail": media(row.get("image_path")),
            "linked_face_id": row.get("linked_face_id"),
            "events": events,
        }

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
        row = visitor_row(f, stats[face_id])
        metrics = _visit_metrics(face_id)
        return {**row, "events": evs, "visits": metrics["visits"]}

    @app.get("/api/events")
    def list_events(type: Optional[str] = Query(None, pattern="^(entry|exit)$"),
                    face_id: Optional[int] = None, limit: int = Query(25, ge=1, le=200), offset: int = 0):
        if not db.exists():
            return {"total": 0, "items": []}
        where, params = [], []
        if type:
            where.append("event_type=?"); params.append(type)
        if face_id is not None:
            where.append("face_id=?"); params.append(face_id)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        with connect() as con:
            total_row = con.execute(f"SELECT COUNT(*) AS c FROM events {clause}", params).fetchone()
            total = int(_as_dict(total_row)["c"] if isinstance(total_row, dict)
                        or hasattr(total_row, "keys") else total_row[0])
            rows = con.execute(
                f"SELECT * FROM events {clause} ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?",
                list(params) + [limit, offset]).fetchall()
        items = []
        for r in rows:
            d = _as_dict(r)
            items.append({"id": d["id"], "face_id": d["face_id"], "event_type": d["event_type"],
                          "timestamp": d["timestamp"], "image": media(d["image_path"]),
                          "track_id": d["track_id"], "frame_idx": d["frame_idx"], "note": d["note"]})
        return {"total": total, "items": items}

    @app.get("/api/visitors/{face_id}/embeddings")
    def visitor_embeddings(face_id: int):
        if not db.exists():
            raise HTTPException(404, "No database yet")
        with connect() as con:
            rows = [_as_dict(r) for r in con.execute(
                "SELECT id, face_id, vector, created_at, image_path FROM embeddings ORDER BY id")]
            face_imgs = {r["face_id"]: r["image_path"] for r in (
                _as_dict(x) for x in con.execute("SELECT face_id, image_path FROM faces"))}
        if face_id not in face_imgs:
            raise HTTPException(404, "Visitor not found")
        by = defaultdict(list)
        for r in rows:
            by[r["face_id"]].append((r, np.frombuffer(_blob(r["vector"]), dtype=np.float32)))

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
                  level: Optional[str] = None, q: Optional[str] = None,
                  source: str = Query("events", pattern="^(events|away)$")):
        """Parsed tail of events.log (default) or away_events.log, newest first."""
        target = log_file if source != "away" else (logs / "away_events.log")
        if not target.exists():
            return {"path": str(target), "exists": False, "source": source,
                    "total": 0, "counts": {}, "items": []}
        size = target.stat().st_size
        with open(target, "rb") as f:
            f.seek(max(0, size - 2_000_000))
            lines = f.read().decode("utf-8", errors="replace").splitlines()
        if size > 2_000_000:
            lines = lines[1:]
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
        return {"path": str(target), "exists": True, "source": source,
                "total": len(items), "counts": dict(counts), "items": items[:limit]}

    # Optional Groq analytics assistant (read-only tools over AnalyticsService).
    try:
        from api.assistant import create_assistant_router
        from api.services import AnalyticsService
        cfg_summary = {
            "match_threshold": match_threshold,
            "detection_skip_frames": None,
            "backend": getattr(db, "backend", None),
        }
        try:
            from src.core.config import load_config as _lc
            _cfg = _lc(os.environ.get("TRACKER_CONFIG", "config.json"))
            cfg_summary.update({
                "detection_skip_frames": _cfg.get("detection_skip_frames"),
                "max_lost_frames": _cfg.get("tracker", {}).get("max_lost_frames"),
                "min_face_px": _cfg.get("quality", {}).get("min_face_px"),
                "min_blur_var": _cfg.get("quality", {}).get("min_blur_var"),
                "min_hits": _cfg.get("tracker", {}).get("min_hits"),
                "iou_threshold": _cfg.get("tracker", {}).get("iou_threshold"),
                "exit_grace_seconds": _cfg.get("presence", {}).get("exit_grace_seconds"),
                "edge_exit_grace_seconds": _cfg.get("presence", {}).get("edge_exit_grace_seconds"),
                "stationary_grace_seconds": _cfg.get("presence", {}).get("stationary_grace_seconds"),
                "backend": _cfg.get("storage", {}).get("backend") or cfg_summary.get("backend"),
            })
        except Exception:  # noqa: BLE001
            pass
        analytics = AnalyticsService(
            db, logs, events_log=str(log_file),
            match_threshold=match_threshold, config_summary=cfg_summary)
        app.include_router(create_assistant_router(analytics))
    except Exception as exc:  # noqa: BLE001
        import logging as _logging
        _logging.getLogger(__name__).warning("Assistant router not mounted: %s", exc)

    # --- Optional video preprocessing (settings file write only; never DB) --------
    from fastapi import Header, Request
    from fastapi.responses import FileResponse

    def _preprocess_cfg_block() -> dict:
        try:
            from src.core.config import load_config as _lc
            return (_lc(os.environ.get("TRACKER_CONFIG", "config.json")).get("preprocess") or {})
        except Exception:  # noqa: BLE001
            return {}

    @app.get("/api/preprocess")
    def preprocess_get(request: Request = None,
                       x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token")):
        from api.preprocess_service import current_payload
        host = getattr(getattr(request, "client", None), "host", None) if request else "127.0.0.1"
        return current_payload(
            _preprocess_cfg_block(), client_host=host, admin_header=x_admin_token)

    @app.put("/api/preprocess")
    def preprocess_put(body: dict = Body(...), request: Request = None,
                       x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token")):
        from api.preprocess_service import (
            atomic_write_override, check_write_allowed, current_payload, validate_and_merge,
        )
        host = getattr(getattr(request, "client", None), "host", None) if request else "127.0.0.1"
        ok, reason = check_write_allowed(host, x_admin_token)
        if not ok:
            raise HTTPException(403, reason)
        try:
            settings = validate_and_merge(body)
            atomic_write_override(settings)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, "failed to write preprocess settings") from exc
        return current_payload(
            _preprocess_cfg_block(), client_host=host, admin_header=x_admin_token)

    @app.post("/api/preprocess/reset")
    def preprocess_reset(request: Request = None,
                         x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token")):
        from api.preprocess_service import check_write_allowed, current_payload, reset_override
        host = getattr(getattr(request, "client", None), "host", None) if request else "127.0.0.1"
        ok, reason = check_write_allowed(host, x_admin_token)
        if not ok:
            raise HTTPException(403, reason)
        reset_override()
        return current_payload(
            _preprocess_cfg_block(), client_host=host, admin_header=x_admin_token)

    @app.get("/api/preprocess/preview")
    def preprocess_preview():
        from api.preprocess_service import preview_path
        p = preview_path()
        if not p.is_file():
            raise HTTPException(404, "no preprocess preview yet")
        return FileResponse(str(p), media_type="image/jpeg")

    app.mount("/media", StaticFiles(directory=str(logs)), name="media")
    if web_dist and Path(web_dist).is_dir():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
    return app


def _default_app() -> FastAPI:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:  # noqa: BLE001
        pass
    from src.core.config import load_config
    from src.core.paths import PROJECT_ROOT
    from src.storage import DatabaseError, open_database, redact_secrets
    _cfg_path = os.environ.get("TRACKER_CONFIG", str(PROJECT_ROOT / "config.json"))
    cfg = load_config(_cfg_path)
    log_dir = os.environ.get("TRACKER_LOG_DIR", cfg["storage"]["log_dir"])
    default_log = cfg["storage"]["events_log"] if "TRACKER_LOG_DIR" not in os.environ else str(Path(log_dir) / "events.log")
    # DATABASE_URL / TRACKER_DSN required — see src.storage.open_database.
    storage = dict(cfg["storage"])
    db, db_info = open_database(storage, reset=False)
    app = create_app(db=db, log_dir=log_dir,
                     events_log=os.environ.get("TRACKER_EVENTS_LOG", default_log),
                     match_threshold=cfg["recognition"]["match_threshold"],
                     live_preview=os.environ.get(
                         "TRACKER_LIVE", cfg["output"].get("live_preview", "data/runtime/live.jpg")))
    app.state.db_bootstrap = db_info  # type: ignore[attr-defined]
    return app


try:
    app = _default_app()
except Exception as _exc:  # noqa: BLE001
    # Keep `from api.server import create_app` working in tests / offline shells.
    # uvicorn api.server:app will serve a hard-fail health until DATABASE_URL works.
    import logging as _logging
    from src.storage import redact_secrets
    _err = redact_secrets(str(_exc))
    _logging.getLogger(__name__).error("Postgres required — connect DB: %s", _err)
    app = FastAPI(title="Face Tracker Analytics API (db unavailable)")

    @app.get("/api/health")
    def _offline_health():
        return {
            "ok": False,
            "db_exists": False,
            "backend": None,
            "fallback": False,
            "error": f"Connect local Postgres: {_err}",
        }

    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    def _offline_api(path: str):
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail=f"Database not connected. Set DATABASE_URL (local Postgres). {_err}",
        )
