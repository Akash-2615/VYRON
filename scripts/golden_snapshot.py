#!/usr/bin/env python3
"""Normalized before/after snapshots for the modular refactor safety net.

Usage (from repo root):
  python scripts/golden_snapshot.py --save before
  python scripts/golden_snapshot.py --compare before

Exit 0 if identical; exit 1 and print a readable diff if not.
Snapshots live under ``.golden/<name>/`` (gitignored). This script is the
only artifact that should be committed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GOLDEN_ROOT = ROOT / ".golden"

# --------------------------------------------------------------------------- normalize
_SID = re.compile(r"\bsid=[A-Za-z0-9_]+")
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?")
_ABS = re.compile(r"(?:/Users|/home|/tmp|/var|/private)[^\s\"']+")
_MEDIA = re.compile(r"/media/[^\s\"']+")
_HEX = re.compile(r"\b[0-9a-f]{8,}\b", re.I)


def _norm_text(s: str, *, tmp_root: Optional[str] = None) -> str:
    out = s.replace("\\", "/")
    if tmp_root:
        out = out.replace(Path(tmp_root).as_posix(), "<TMP>")
        out = out.replace(str(tmp_root), "<TMP>")
    out = out.replace(ROOT.as_posix(), "<ROOT>")
    out = out.replace(str(ROOT), "<ROOT>")
    out = _ABS.sub("<ABS>", out)
    out = _MEDIA.sub("/media/<PATH>", out)
    out = _SID.sub("sid=<SID>", out)
    out = _ISO.sub("<TS>", out)
    out = _HEX.sub("<HEX>", out)
    return out


def _norm_obj(obj: Any, *, tmp_root: Optional[str] = None) -> Any:
    if isinstance(obj, dict):
        return {k: _norm_obj(v, tmp_root=tmp_root) for k, v in sorted(obj.items(), key=lambda x: str(x[0]))}
    if isinstance(obj, list):
        return [_norm_obj(v, tmp_root=tmp_root) for v in obj]
    if isinstance(obj, str):
        return _norm_text(obj, tmp_root=tmp_root)
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- pipeline slice
def _capture_pipeline(out_dir: Path) -> None:
    import numpy as np
    from src.config import load_config
    from src.database import Database
    from src.event_logger import EventLogger
    from src.pipeline import FaceTrackingPipeline

    # Same fake scenario as tests/test_pipeline.py::run (skip=0).
    RED, GREEN = (0, 0, 255), (0, 255, 0)

    class FakeDetector:
        def detect(self, frame):
            out = []
            for colour in (RED, GREEN):
                ys, xs = np.where(np.all(frame == colour, axis=2))
                if len(xs):
                    out.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 0.9])
            return np.array(out, dtype=np.float32).reshape(-1, 5)

    class FakeEmbedder:
        def embed(self, crop):
            red = int(np.all(crop == RED, axis=2).sum())
            green = int(np.all(crop == GREEN, axis=2).sum())
            v = np.zeros(512, dtype=np.float32)
            v[0 if red >= green else 1] = 1.0
            return v, 0.99

    def make_frame(a_x=None, b_x=None):
        f = np.zeros((240, 320, 3), dtype=np.uint8)
        if a_x is not None:
            f[60:140, a_x:a_x + 80] = RED
        if b_x is not None:
            f[100:180, b_x:b_x + 80] = GREEN
        return f

    with tempfile.TemporaryDirectory(prefix="golden_pipe_") as td:
        tmp = Path(td)
        cfg = load_config(str(ROOT / "config.json"))
        cfg["quality"].update(min_face_px=10, min_blur_var=0)
        cfg["recognition"]["min_embeds_to_register"] = 2
        cfg["away_walkers"] = {"enabled": False}
        cfg["appearance"] = {"enabled": False}
        cfg["timestamps"] = "video"
        cfg["detection_skip_frames"] = 0
        cfg["tracker"]["max_lost_frames"] = 20
        cfg["presence"] = {
            "exit_grace_seconds": 0.4,
            "edge_exit_grace_seconds": 0.2,
            "stationary_grace_seconds": 0.4,
            "occlusion_grace_multiplier": 2.0,
            "edge_margin_px": 24,
            "stationary_speed_px_s": 15,
            "resume_spatial_iou": 0.2,
            "resume_threshold_margin": 0.05,
            "reverify_interval_seconds": 100,
            "occlusion_iou": 0.3,
            "mass_loss_ratio": 0.6,
            "max_stream_gap_seconds": 20,
            "exit_zones": [],
        }
        db_path = tmp / "t.db"
        log_dir = tmp / "logs"
        log_file = log_dir / "events.log"
        db = Database(str(db_path), reset=True)
        ev = EventLogger(str(log_dir), str(log_file), session_id="golden_sid_0001")
        pipe = FaceTrackingPipeline(cfg, FakeDetector(), FakeEmbedder(), db, ev)
        pipe.set_timing(25.0, live=False)
        for i in range(110):
            a = 20 + i if i <= 40 else (30 if 80 <= i <= 100 else None)
            b = 200 if 20 <= i <= 60 else None
            pipe.process_frame(make_frame(a if i <= 40 or 80 <= i <= 100 else None, b), i, ts=i / 25.0)
        pipe.finalize()

        tables: Dict[str, Any] = {
            "unique_faces": db.unique_count(),
            "event_counts": db.event_counts(),
            "away_stats": db.away_stats(),
        }
        for name, sql in (
            ("faces", "SELECT face_id, visit_count FROM faces ORDER BY face_id"),
            ("events", "SELECT face_id, event_type, note, track_id FROM events ORDER BY id"),
            ("embeddings", "SELECT face_id, COUNT(*) AS n FROM embeddings GROUP BY face_id ORDER BY face_id"),
            ("visits", "SELECT face_id, exit_reason, resume_count FROM visits ORDER BY visit_id"),
        ):
            try:
                rows = db.query(sql)
                tables[name] = [list(r) if not isinstance(r, dict) else r for r in rows]
            except Exception as exc:  # noqa: BLE001
                tables[name] = f"<error:{type(exc).__name__}>"

        try:
            emb_n = db.query("SELECT COUNT(*) FROM embeddings")[0][0]
        except Exception:  # noqa: BLE001
            emb_n = None
        tables["embeddings_count"] = emb_n

        raw_log = log_file.read_text(encoding="utf-8") if log_file.is_file() else ""
        # Drop absolute image paths from log lines but keep event names + key fields.
        lines = []
        for line in raw_log.splitlines():
            # Keep event token + key=value after the pipe fields.
            parts = line.split("|")
            if len(parts) >= 3:
                body = parts[2].strip()
            else:
                body = line
            body = re.sub(r"\bimage=\S+", "image=<IMG>", body)
            body = re.sub(r"\bcloseup=\S+", "closeup=<IMG>", body)
            lines.append(_norm_text(body, tmp_root=td))

        db.close()
        _write_json(out_dir / "pipeline" / "tables.json", _norm_obj(tables, tmp_root=td))
        _write_text(out_dir / "pipeline" / "events.log", "\n".join(lines) + ("\n" if lines else ""))


# --------------------------------------------------------------------------- API slice
def _seed_demo_into(demo_root: Path) -> Tuple[Path, Path]:
    """Run seed_demo logic into an isolated folder (never touches repo demo/)."""
    import random
    from datetime import datetime, timedelta

    import cv2
    import numpy as np

    from src.database import Database
    from src.event_logger import EventLogger

    demo_root.mkdir(parents=True, exist_ok=True)
    db_path = demo_root / "visitors.db"
    log_dir = demo_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def placeholder(face_id: int, size: int = 160) -> np.ndarray:
        rng = random.Random(face_id)
        img = np.full((160, 160, 3), (rng.randint(60, 140), rng.randint(60, 140), rng.randint(60, 140)), np.uint8)
        skin = (rng.randint(140, 200), rng.randint(150, 205), rng.randint(180, 235))
        cv2.circle(img, (80, 80), 52, skin, -1)
        cv2.circle(img, (62, 70), 6, (30, 30, 30), -1)
        cv2.circle(img, (98, 70), 6, (30, 30, 30), -1)
        cv2.ellipse(img, (80, 104), (18, 9), 0, 0, 180, (40, 40, 120), 3)
        cv2.putText(img, f"#{face_id}", (6, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return img if size == 160 else cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)

    def unit(v: np.ndarray) -> np.ndarray:
        return (v / np.linalg.norm(v)).astype(np.float32)

    # Keep CWD as demo_root so EventLogger relative crop paths stay under it.
    prev = os.getcwd()
    os.chdir(demo_root)
    try:
        db = Database(str(db_path), reset=True)
        ev = EventLogger("logs", "logs/events.log", session_id="golden_demo_0001")
        rng = random.Random(7)
        base = datetime(2026, 11, 6, 10, 0, 0)
        lines = []

        def log(ts, event, level="INFO", **f):
            lines.append((ts, level, event, " ".join(f"{k}={v}" for k, v in f.items())))

        unique = 0
        for fid in range(1, 19):
            nrng = np.random.default_rng(fid)
            identity = unit(nrng.normal(size=512))
            visits, t = [], rng.randint(0, 560)
            for _ in range(2 if rng.random() < 0.3 else 1):
                dur = rng.randint(20, 90)
                visits.append((t, dur))
                t += dur + rng.randint(40, 120)

            for n, (start, dur) in enumerate(visits):
                t_in, t_out = base + timedelta(seconds=start), base + timedelta(seconds=start + dur)
                track = fid * 10 + n
                log(t_in - timedelta(seconds=3), "TRACK_STARTED", track=track, frame=start * 25, conf="0.91")
                log(t_in - timedelta(seconds=2), "EMBEDDING_GENERATED", track=track, det_score="0.99", attempt=1)
                emb_ts = t_in - timedelta(seconds=1)
                closeup = ev.save_crop("face", fid, emb_ts, placeholder(fid, 224))
                if n == 0:
                    emb = unit(identity + 0.25 * nrng.normal(size=512) / np.sqrt(512) * 4)
                    unique += 1
                    assert db.add_face(t_in.isoformat(timespec="milliseconds")) == fid
                    db.add_embedding(fid, emb, emb_ts.isoformat(timespec="milliseconds"), closeup)
                    db.set_face_image(fid, closeup)
                    log(t_in - timedelta(seconds=2), "EMBEDDING_GENERATED", track=track, det_score="0.98", attempt=2)
                    log(emb_ts, "FACE_REGISTERED", face_id=fid, track=track, unique_total=unique, closeup=closeup)
                else:
                    emb = unit(identity + nrng.normal(size=512) / np.sqrt(512) * 0.9)
                    sim = float(emb @ unit(identity))
                    db.add_embedding(fid, emb, emb_ts.isoformat(timespec="milliseconds"), closeup)
                    log(emb_ts, "FACE_RECOGNIZED", face_id=fid, track=track, similarity=f"{sim:.3f}")
                for kind, ts, note in (("entry", t_in, "new" if n == 0 else "re-entry"), ("exit", t_out, "lost")):
                    path = ev.save_crop(kind, fid, ts, placeholder(fid))
                    db.log_event(fid, kind, ts.isoformat(timespec="milliseconds"), path, track, int(ts.timestamp()) % 100000, note)
                    if kind == "exit":
                        log(ts, "TRACK_ENDED", track=track, face_id=fid, reason=note)
                    log(ts, kind.upper(), face_id=fid, track=track, note=note, image=path)

        lines.sort(key=lambda x: x[0])
        with open("logs/events.log", "w", encoding="utf-8") as f:
            f.write(f"{base:%Y-%m-%d %H:%M:%S},000 | INFO    | SYSTEM_START         sid=golden_demo_0001 sources=1 skip=2 db_reset=True\n")
            for ts, level, event, rest in lines:
                f.write(f"{ts:%Y-%m-%d %H:%M:%S},{ts.microsecond // 1000:03d} | {level:<7} | {event:<20} {rest}\n")
        db.close()
    finally:
        os.chdir(prev)
    return db_path, log_dir


def _capture_api(out_dir: Path) -> None:
    pytest = __import__("pytest")
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from api.server import create_app

    with tempfile.TemporaryDirectory(prefix="golden_api_") as td:
        demo = Path(td) / "demo"
        db_path, log_dir = _seed_demo_into(demo)
        live = Path(td) / "missing.jpg"
        app = create_app(
            db_path=str(db_path),
            log_dir=str(log_dir),
            web_dist=None,
            events_log=str(log_dir / "events.log"),
            live_preview=str(live),
        )
        c = TestClient(app)
        endpoints = [
            ("health", "/api/health"),
            ("summary", "/api/summary"),
            ("timeline", "/api/timeline"),
            ("visitors", "/api/visitors"),
            ("visitor_1", "/api/visitors/1"),
            ("visitor_1_embeddings", "/api/visitors/1/embeddings"),
            ("events", "/api/events?limit=50&offset=0"),
            ("logs", "/api/logs?limit=50"),
            ("count", "/api/count"),
            ("live_status", "/api/live/status"),
            ("away_visitors", "/api/away-visitors"),
            ("tracker_status", "/api/tracker/status"),
            ("videos", "/api/videos"),
        ]
        bundle: Dict[str, Any] = {}
        for name, path in endpoints:
            r = c.get(path)
            payload: Any
            try:
                payload = r.json()
            except Exception:  # noqa: BLE001
                payload = {"_status": r.status_code, "_text": r.text[:500]}
            # Drop raw embedding vectors (huge + float noise) — keep shape/meta.
            if name.endswith("embeddings") and isinstance(payload, dict):
                samples = payload.get("samples") or []
                payload = {
                    **{k: v for k, v in payload.items() if k != "samples"},
                    "samples": [
                        {k: v for k, v in s.items() if k != "vector"}
                        | {"vector_len": len(s.get("vector") or [])}
                        for s in samples
                    ],
                }
            bundle[name] = {"status": r.status_code, "body": _norm_obj(payload, tmp_root=td)}
        _write_json(out_dir / "api" / "get_endpoints.json", bundle)


def save_snapshot(name: str) -> Path:
    dest = GOLDEN_ROOT / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    _capture_pipeline(dest)
    _capture_api(dest)
    meta = {
        "name": name,
        "repo": str(ROOT),
        "files": sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()),
    }
    _write_json(dest / "meta.json", meta)
    print(f"saved snapshot → {dest}")
    return dest


def _load_tree(root: Path) -> Dict[str, str]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            out[rel] = p.read_text(encoding="utf-8")
    return out


def compare_snapshot(name: str) -> int:
    before = GOLDEN_ROOT / name
    if not before.is_dir():
        print(f"missing snapshot: {before}", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="golden_cmp_") as td:
        after = Path(td) / "after"
        # Re-run into a temp folder then compare file-by-file against saved.
        after.mkdir()
        _capture_pipeline(after)
        _capture_api(after)
        a = _load_tree(before)
        b = _load_tree(after)
        # Ignore meta timestamps / absolute repo path in meta.json
        a.pop("meta.json", None)
        b.pop("meta.json", None)
        keys = sorted(set(a) | set(b))
        diffs: List[str] = []
        for k in keys:
            if k not in a:
                diffs.append(f"+ only in after: {k}")
                continue
            if k not in b:
                diffs.append(f"- only in before: {k}")
                continue
            if a[k] != b[k]:
                diffs.append(f"! differ: {k}")
                la, lb = a[k].splitlines(), b[k].splitlines()
                import difflib
                for line in difflib.unified_diff(la, lb, fromfile=f"before/{k}", tofile=f"after/{k}", lineterm="", n=2):
                    diffs.append(line)
        if not diffs:
            print(f"OK — snapshot identical to {before}")
            return 0
        print(f"DIFF vs {before} ({len(diffs)} lines)")
        print("\n".join(diffs[:400]))
        if len(diffs) > 400:
            print(f"... ({len(diffs) - 400} more lines)")
        return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", metavar="NAME", help="Write normalized snapshot under .golden/NAME")
    g.add_argument("--compare", metavar="NAME", help="Re-run and compare against .golden/NAME")
    args = ap.parse_args()
    if args.save:
        save_snapshot(args.save)
        return 0
    return compare_snapshot(args.compare)


if __name__ == "__main__":
    raise SystemExit(main())
