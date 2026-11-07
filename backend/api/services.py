"""Shared read-only analytics helpers used by HTTP routes and the assistant tools."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.reporting import count_from_database, count_from_log, reconcile
from src.storage import Database

EPOCH = datetime(1970, 1, 1)
BUCKETS = [1, 5, 10, 30, 60, 300, 900, 3600, 21600, 86400]
LOG_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),\d+ \| (\w+)\s*\| (\S+)\s*(.*)$")
FIELD_RE = re.compile(r"(\w+)=(\S+)")
MAX_TOOL_ROWS = 50


def secs(ts) -> float:
    """Parse ISO timestamps from SQLite/Postgres TEXT (or datetime objects)."""
    if ts is None:
        return 0.0
    if isinstance(ts, datetime):
        return (ts.replace(tzinfo=None) - EPOCH).total_seconds()
    s = str(ts).replace("Z", "")
    if " " in s and "T" not in s:
        s = s.replace(" ", "T", 1)
    return (datetime.fromisoformat(s) - EPOCH).total_seconds()


def as_dict(row: Any) -> dict:
    if isinstance(row, dict):
        return dict(row)
    if hasattr(row, "keys"):
        return {k: row[k] for k in row.keys()}
    raise TypeError(f"Cannot convert row: {type(row)}")


def blob(val: Any) -> bytes:
    if isinstance(val, memoryview):
        return val.tobytes()
    if isinstance(val, bytearray):
        return bytes(val)
    return val


class AnalyticsService:
    """Read-only view over the visitors DB + events.log (no writes)."""

    def __init__(
        self,
        db: Database,
        log_dir: str | Path,
        events_log: Optional[str] = None,
        match_threshold: float = 0.45,
        config_summary: Optional[dict] = None,
    ):
        self.db = db
        self.logs = Path(log_dir).resolve()
        self.log_file = Path(events_log) if events_log else self.logs / "events.log"
        self.match_threshold = float(match_threshold)
        self.config_summary = dict(config_summary or {})
        self._count_cache: Dict[str, Any] = {"t": 0.0, "key": "", "val": None}

    def connect(self):
        return self.db.readonly()

    def get_count(self, session: str = "latest") -> dict:
        """DB-primary unique count with optional log reconciliation (cached 2s)."""
        now = datetime.now().timestamp()
        key = session or "latest"
        cached = self._count_cache
        if cached["key"] == key and cached["val"] is not None and now - cached["t"] < 2.0:
            return cached["val"]
        db_counts = count_from_database(self.db)
        log_counts = count_from_log(self.log_file, session=key, include_rotated=True)
        result = reconcile(db_counts if db_counts.available else None,
                           log_counts if log_counts.available else None)
        if db_counts.available:
            primary = db_counts
            source = "db"
        elif log_counts.available:
            primary = log_counts
            source = "log"
        else:
            primary = None
            source = "none"
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
        self._count_cache = {"t": now, "key": key, "val": payload}
        return payload

    def media(self, p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        try:
            return "/media/" + Path(p).resolve().relative_to(self.logs).as_posix()
        except ValueError:
            return None

    def snapshot(self) -> Tuple[list, list, dict, list]:
        if not self.db.exists():
            return [], [], {}, []
        try:
            with self.connect() as con:
                faces = [as_dict(r) for r in con.execute("SELECT * FROM faces ORDER BY face_id")]
                events = [as_dict(r) for r in con.execute(
                    "SELECT * FROM events ORDER BY timestamp, id")]
        except Exception:  # noqa: BLE001
            return [], [], {}, []
        stats = {f["face_id"]: {"total_dwell": 0.0, "inside": False, "thumb": None, "latest": None,
                                "entry_ts": None} for f in faces}
        dwell, inside, peak = [], set(), 0
        for e in events:
            s = stats.get(e["face_id"])
            if s is None:
                continue
            img = self.media(e.get("image_path"))
            if e["event_type"] == "entry":
                inside.add(e["face_id"])
                peak = max(peak, len(inside))
                s["inside"], s["entry_ts"] = True, e["timestamp"]
                s["thumb"] = s["thumb"] or img
            else:
                inside.discard(e["face_id"])
                s["inside"] = False
                if s["entry_ts"]:
                    d = secs(e["timestamp"]) - secs(s["entry_ts"])
                    s["total_dwell"] += d
                    dwell.append(d)
                    s["entry_ts"] = None
            s["latest"] = img or s["latest"]
        return faces, events, stats, dwell + [peak]

    def visit_metrics(self, face_id: int) -> dict:
        empty = {"time_in_frame_seconds": 0.0, "visible_seconds": 0.0,
                 "occluded_seconds": 0.0, "resume_count": 0, "hidden_now": False, "visits": []}
        if not self.db.exists():
            return empty
        try:
            with self.connect() as con:
                rows = [as_dict(r) for r in con.execute(
                    "SELECT visit_id, face_id, entry_ts, exit_ts, visible_seconds, occluded_seconds, "
                    "resume_count, exit_reason FROM visits WHERE face_id=? ORDER BY visit_id",
                    (face_id,))]
        except Exception:  # noqa: BLE001
            return empty
        vis = sum(float(r["visible_seconds"] or 0) for r in rows)
        occ = sum(float(r["occluded_seconds"] or 0) for r in rows)
        resumes = sum(int(r["resume_count"] or 0) for r in rows)
        return {
            "time_in_frame_seconds": round(vis + occ, 1),
            "visible_seconds": round(vis, 1),
            "occluded_seconds": round(occ, 1),
            "resume_count": resumes,
            "hidden_now": bool([r for r in rows if not r.get("exit_ts")]),
            "visits": [{
                "visit_id": r["visit_id"], "entry_ts": r["entry_ts"], "exit_ts": r["exit_ts"],
                "visible_seconds": r["visible_seconds"], "occluded_seconds": r["occluded_seconds"],
                "resume_count": r["resume_count"], "exit_reason": r["exit_reason"],
            } for r in rows],
        }

    def visitor_row(self, f: dict, s: dict) -> dict:
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
        metrics = self.visit_metrics(int(f["face_id"]))
        hidden_now = bool(metrics["visits"]) and any(
            v["exit_ts"] is None for v in metrics["visits"]) and s["inside"]
        dwell = metrics["time_in_frame_seconds"] or round(s["total_dwell"], 1)
        return {
            "face_id": f["face_id"], "first_seen": f["first_seen"], "last_seen": f["last_seen"],
            "visit_count": f["visit_count"], "total_dwell_seconds": dwell,
            "time_in_frame_seconds": metrics["time_in_frame_seconds"],
            "visible_seconds": metrics["visible_seconds"],
            "occluded_seconds": metrics["occluded_seconds"],
            "resume_count": metrics["resume_count"], "hidden_now": hidden_now,
            "status": "inside" if s["inside"] else "left",
            "appearance_summary": f.get("appearance_summary"),
            "appearance": appearance,
        }

    def away_stats(self) -> dict:
        base = {"away_unique": 0, "away_entries": 0, "away_exits": 0,
                "away_currently": 0, "away_with_clothes": 0,
                "note": "Away walkers are approximate and NOT part of unique visitor count."}
        if not self.db.exists():
            return base
        try:
            with self.connect() as con:
                unique = con.execute("SELECT COUNT(*) AS c FROM away_visitors").fetchone()
                unique = int(as_dict(unique)["c"] if isinstance(unique, dict) or hasattr(unique, "keys")
                             else unique[0])
                rows = con.execute(
                    "SELECT event_type, COUNT(*) AS c FROM away_events GROUP BY event_type").fetchall()
                clothed_row = con.execute(
                    "SELECT COUNT(*) AS c FROM away_visitors WHERE appearance_summary IS NOT NULL "
                    "AND appearance_summary != ''").fetchone()
                clothed = int(as_dict(clothed_row)["c"] if isinstance(clothed_row, dict)
                              or hasattr(clothed_row, "keys") else clothed_row[0])
        except Exception:  # noqa: BLE001
            return base
        counts = {}
        for r in rows:
            d = as_dict(r)
            counts[d["event_type"]] = int(d["c"])
        entries, exits = counts.get("entry", 0), counts.get("exit", 0)
        return {**base, "away_unique": unique, "away_entries": entries, "away_exits": exits,
                "away_currently": max(0, entries - exits), "away_with_clothes": clothed}

    def get_summary(self) -> dict:
        faces, events, stats, extra = self.snapshot()
        dwell, peak = (extra[:-1], extra[-1]) if extra else ([], 0)
        entries = sum(1 for e in events if e["event_type"] == "entry")
        inside = sum(1 for s in stats.values() if s["inside"])
        avg_tif = None
        if self.db.exists():
            try:
                with self.connect() as con:
                    rows = con.execute(
                        "SELECT visible_seconds + occluded_seconds AS tif FROM visits "
                        "WHERE exit_ts IS NOT NULL").fetchall()
                    if rows:
                        vals = [float((as_dict(r)["tif"] if not isinstance(r, tuple) else r[0]) or 0)
                                for r in rows]
                        avg_tif = round(sum(vals) / len(vals), 1)
                    open_row = con.execute(
                        "SELECT COUNT(*) AS c FROM visits WHERE exit_ts IS NULL").fetchone()
                    open_n = int(as_dict(open_row)["c"] if isinstance(open_row, dict)
                                 or hasattr(open_row, "keys") else open_row[0])
                    inside = max(inside, int(open_n))
            except Exception:  # noqa: BLE001
                pass
        first = events[0]["timestamp"] if events else None
        last = events[-1]["timestamp"] if events else None
        return {
            "unique_visitors": len(faces),
            "total_entries": entries,
            "total_exits": len(events) - entries,
            "currently_inside": inside,
            "returning_visitors": sum(1 for f in faces if f["visit_count"] > 1),
            "avg_dwell_seconds": round(sum(dwell) / len(dwell), 1) if dwell else None,
            "avg_time_in_frame_seconds": avg_tif if avg_tif is not None else (
                round(sum(dwell) / len(dwell), 1) if dwell else None),
            "peak_occupancy": peak,
            "first_event": first,
            "last_event": last,
            "data_start": first,
            "data_end": last,
            **{k: v for k, v in self.away_stats().items() if k != "note"},
        }

    def get_timeline(self, bucket_seconds: Optional[int] = None) -> dict:
        _, events, _, _ = self.snapshot()
        if not events:
            return {"bucket_seconds": 60, "points": [], "data_start": None, "data_end": None}
        t0, t1 = secs(events[0]["timestamp"]), secs(events[-1]["timestamp"])
        span = max(t1 - t0, 1)
        size = int(bucket_seconds) if bucket_seconds else next(
            (b for b in BUCKETS if span / b <= 60), BUCKETS[-1])
        size = max(1, min(size, 86400))
        start = int(t0 // size) * size
        n = int((t1 - start) // size) + 1
        n = min(n, 500)
        entries, exits = [0] * n, [0] * n
        for e in events:
            i = int((secs(e["timestamp"]) - start) // size)
            if 0 <= i < n:
                (entries if e["event_type"] == "entry" else exits)[i] += 1
        pts, occ = [], 0
        for i in range(n):
            occ = max(0, occ + entries[i] - exits[i])
            pts.append({"t": (EPOCH + timedelta(seconds=start + i * size)).isoformat(),
                        "entries": entries[i], "exits": exits[i], "occupancy": occ})
        return {"bucket_seconds": size, "points": pts,
                "data_start": events[0]["timestamp"], "data_end": events[-1]["timestamp"]}

    def list_visitors(self, status: Optional[str] = None, min_visits: int = 0,
                      sort_by: str = "last_seen", limit: int = 20) -> list:
        limit = max(1, min(int(limit or 20), MAX_TOOL_ROWS))
        min_visits = max(0, int(min_visits or 0))
        faces, _, stats, _ = self.snapshot()
        rows = [self.visitor_row(f, stats[f["face_id"]]) for f in faces]
        if status in ("inside", "left"):
            rows = [r for r in rows if r["status"] == status]
        if min_visits:
            rows = [r for r in rows if int(r["visit_count"]) >= min_visits]
        key = {
            "last_seen": lambda r: r["last_seen"] or "",
            "dwell": lambda r: float(r.get("time_in_frame_seconds") or 0),
            "visits": lambda r: int(r["visit_count"]),
            "face_id": lambda r: int(r["face_id"]),
        }.get(sort_by or "last_seen", lambda r: r["last_seen"] or "")
        reverse = sort_by != "face_id"
        rows = sorted(rows, key=key, reverse=reverse)[:limit]
        # Strip heavy appearance blobs for tool responses
        for r in rows:
            r.pop("appearance", None)
        return rows

    def get_visitor(self, face_id: int) -> Optional[dict]:
        faces, events, stats, _ = self.snapshot()
        f = next((x for x in faces if int(x["face_id"]) == int(face_id)), None)
        if f is None:
            return None
        row = self.visitor_row(f, stats[int(face_id)])
        metrics = self.visit_metrics(int(face_id))
        evs = [{
            "event_type": e["event_type"], "timestamp": e["timestamp"],
            "note": (e.get("note") or "")[:200],
        } for e in events if int(e["face_id"]) == int(face_id)]
        evs = evs[-MAX_TOOL_ROWS:]
        appearance = row.get("appearance")
        appearance_safe = None
        if isinstance(appearance, dict):
            appearance_safe = {k: appearance[k] for k in (
                "summary", "upper_color", "upper_type", "lower_type", "accessories", "hair"
            ) if k in appearance}
        return {
            **{k: v for k, v in row.items() if k != "appearance"},
            "appearance_tags": appearance_safe,
            "events": evs,
            "visits": metrics["visits"][-MAX_TOOL_ROWS:],
        }

    def query_events(self, type: Optional[str] = None, face_id: Optional[int] = None,
                     from_ts: Optional[str] = None, to_ts: Optional[str] = None,
                     limit: int = 25, offset: int = 0) -> dict:
        limit = max(1, min(int(limit or 25), MAX_TOOL_ROWS))
        offset = max(0, int(offset or 0))
        if not self.db.exists():
            return {"total": 0, "items": []}
        where, params = [], []
        if type in ("entry", "exit"):
            where.append("event_type=?"); params.append(type)
        if face_id is not None:
            where.append("face_id=?"); params.append(int(face_id))
        if from_ts:
            where.append("timestamp>=?"); params.append(from_ts[:40])
        if to_ts:
            where.append("timestamp<=?"); params.append(to_ts[:40])
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        try:
            with self.connect() as con:
                total_row = con.execute(f"SELECT COUNT(*) AS c FROM events {clause}", params).fetchone()
                total = int(as_dict(total_row)["c"] if isinstance(total_row, dict)
                            or hasattr(total_row, "keys") else total_row[0])
                rows = con.execute(
                    f"SELECT id, face_id, event_type, timestamp, note, track_id, frame_idx "
                    f"FROM events {clause} ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?",
                    list(params) + [limit, offset]).fetchall()
        except Exception:  # noqa: BLE001
            return {"total": 0, "items": []}
        items = []
        for r in rows:
            d = as_dict(r)
            items.append({
                "id": d["id"], "face_id": d["face_id"], "event_type": d["event_type"],
                "timestamp": d["timestamp"], "note": (d.get("note") or "")[:200],
                "track_id": d.get("track_id"), "frame_idx": d.get("frame_idx"),
            })
        return {"total": total, "items": items}

    def search_logs(self, event: Optional[str] = None, level: Optional[str] = None,
                    query: Optional[str] = None, limit: int = 30) -> dict:
        limit = max(1, min(int(limit or 30), MAX_TOOL_ROWS))
        if not self.log_file.exists():
            return {"exists": False, "total": 0, "items": []}
        size = self.log_file.stat().st_size
        with open(self.log_file, "rb") as f:
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
            # Treat log text as untrusted data — truncate
            parsed.append({
                "ts": f"{day}T{clock}", "level": lvl, "event": ev,
                "message": rest[:500],
                "fields": {k: v[:80] for k, v in FIELD_RE.findall(rest)[:20]},
            })
        needle = query.lower() if query else None
        items = [p for p in reversed(parsed)
                 if (not event or p["event"] == event)
                 and (not level or p["level"] == level)
                 and (not needle or needle in (p["event"] + " " + p["message"]).lower())]
        return {"exists": True, "total": len(items), "items": items[:limit]}

    def get_similar_identities(self, face_id: int, top_k: int = 6) -> dict:
        """Similarity scores only — never return raw embedding vectors."""
        top_k = max(1, min(int(top_k or 6), 10))
        if not self.db.exists():
            return {"face_id": face_id, "similar": [], "intra_similarity": None}
        try:
            with self.connect() as con:
                rows = [as_dict(r) for r in con.execute(
                    "SELECT id, face_id, vector FROM embeddings ORDER BY id")]
                face_ids = {int(as_dict(r)["face_id"]) for r in con.execute("SELECT face_id FROM faces")}
        except Exception:  # noqa: BLE001
            return {"face_id": face_id, "similar": [], "intra_similarity": None}
        if int(face_id) not in face_ids:
            return {"face_id": face_id, "similar": [], "error": "visitor_not_found"}

        def unit(v: np.ndarray) -> np.ndarray:
            return v / max(float(np.linalg.norm(v)), 1e-9)

        by = defaultdict(list)
        for r in rows:
            by[int(r["face_id"])].append(unit(np.frombuffer(blob(r["vector"]), dtype=np.float32)))
        means = {fid: unit(np.mean(vecs, axis=0)) for fid, vecs in by.items() if vecs}
        mine = means.get(int(face_id))
        similar = []
        if mine is not None:
            similar = sorted(
                ({"face_id": fid, "similarity": round(float(mine @ m), 4)}
                 for fid, m in means.items() if fid != int(face_id)),
                key=lambda d: d["similarity"], reverse=True)[:top_k]
        intra = None
        vecs = by.get(int(face_id), [])
        if len(vecs) > 1:
            m = np.stack(vecs) @ np.stack(vecs).T
            intra = round(float((m.sum() - len(vecs)) / (len(vecs) * (len(vecs) - 1))), 4)
        return {"face_id": int(face_id), "match_threshold": self.match_threshold,
                "intra_similarity": intra, "similar": similar,
                "note": "Scores only; raw embedding vectors are never exposed."}

    def get_config_summary(self) -> dict:
        allowed = {
            "detection_skip_frames", "match_threshold", "max_lost_frames",
            "min_face_px", "min_blur_var", "min_hits", "iou_threshold",
            "exit_grace_seconds", "edge_exit_grace_seconds", "stationary_grace_seconds",
            "backend",
        }
        out = {k: v for k, v in self.config_summary.items() if k in allowed}
        out["match_threshold"] = out.get("match_threshold", self.match_threshold)
        out["note"] = "Non-secret tuning values only."
        return out

    def data_window(self) -> Tuple[Optional[str], Optional[str]]:
        s = self.get_summary()
        return s.get("data_start"), s.get("data_end")
