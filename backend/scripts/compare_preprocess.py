#!/usr/bin/env python3
"""Compare one clip with preprocess off vs on (prints counts / FPS; no claimed targets)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))


def _run(source: str, enabled: bool, max_frames: int) -> dict:
    from src.core.config import load_config
    from src.database import Database
    from src.detection import YoloFaceDetector
    from src.event_logger import EventLogger
    from src.pipeline import FaceTrackingPipeline
    from src.recognition import FaceEmbedder
    from main import run_source

    cfg = load_config(str(ROOT / "config.json"))
    cfg.setdefault("preprocess", {})["enabled"] = enabled
    cfg["away_walkers"] = {**(cfg.get("away_walkers") or {}), "enabled": False}
    cfg["appearance"] = {**(cfg.get("appearance") or {}), "enabled": False}
    db = Database(":memory:", reset=True) if False else Database(str(ROOT / "data/db/compare_pp.db"), reset=True)
    # Prefer sqlite for compare to avoid wiping postgres gallery
    from src.storage.database import Database as DB2
    db = DB2(path=str(ROOT / "data/db/compare_pp.db"), backend="sqlite", reset=True)
    ev = EventLogger(str(ROOT / "logs"), str(ROOT / "logs" / "compare_pp_events.log"))
    det = YoloFaceDetector(cfg["detector"])
    emb = FaceEmbedder(cfg["recognition"])
    pipe = FaceTrackingPipeline(cfg, det, emb, db, ev)
    stop = {"flag": False}
    t0 = time.time()
    n = run_source(source, cfg, pipe, db, ev, stop, False, max_frames, "", live_path="")
    dt = max(time.time() - t0, 1e-6)
    counts = db.event_counts()
    log = Path(ROOT / "logs" / "compare_pp_events.log").read_text(encoding="utf-8", errors="replace")
    return {
        "enabled": enabled,
        "frames": n,
        "unique": db.unique_count(),
        "entries": counts.get("entry", 0),
        "exits": counts.get("exit", 0),
        "embed_no_face": log.count("EMBED_NO_FACE"),
        "fps": n / dt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/videos/video_sample1.mp4")
    ap.add_argument("--max-frames", type=int, default=120)
    args = ap.parse_args()
    src = str((ROOT / args.source).resolve()) if not Path(args.source).is_absolute() else args.source
    print("OFF …")
    off = _run(src, False, args.max_frames)
    print(off)
    print("ON …")
    on = _run(src, True, args.max_frames)
    print(on)
    print("delta unique", on["unique"] - off["unique"],
          "entries", on["entries"] - off["entries"],
          "fps", f'{on["fps"]:.1f} vs {off["fps"]:.1f}')


if __name__ == "__main__":
    main()
