#!/usr/bin/env python3
"""Interview mode — paste the RTSP link; everything else starts automatically.

    python backend/scripts/run_rtsp.py "rtsp://user:pass@CAMERA_IP:554/stream"

What this does for you:
  1. Loads config.json + .env (DATABASE_URL required)
  2. Opens local Postgres (hard-fails if not connected)
  3. Loads YOLO + RetinaFace fallback + InsightFace embeddings (+ person/appearance if enabled)
  4. Connects to the RTSP URL (TCP), reconnects on drops
  5. Runs detect → track → embed → ENTRY/EXIT logging + live preview
  6. Starts the dashboard at http://localhost:8000 (if free)
  7. Opens a live OpenCV window (press q to quit)

Face crops and events.log stay under logs/ on the local filesystem.

No other flags required.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
ROOT = PROJECT_ROOT  # chdir / config / logs live at repo root
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _normalize_url(url: str) -> str:
    u = (url or "").strip().strip('"').strip("'")
    if not u:
        raise SystemExit("No RTSP URL given.")
    low = u.lower()
    if not (low.startswith("rtsp://") or low.startswith("rtsps://")):
        raise SystemExit(
            f"Expected an rtsp:// (or rtsps://) URL, got: {u!r}\n"
            "Example: python backend/scripts/run_rtsp.py \"rtsp://user:pass@192.168.1.10:554/stream\""
        )
    return u


def _start_dashboard(cfg: dict) -> Optional[subprocess.Popen]:
    """Best-effort: launch uvicorn so Live monitor is available immediately."""
    port = int(os.environ.get("TRACKER_DASHBOARD_PORT", "8000"))
    # Skip if something already answers on the port.
    try:
        import urllib.request
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=0.4)
        print(f"Dashboard already running → http://127.0.0.1:{port}/")
        return None
    except Exception:  # noqa: BLE001
        pass
    env = os.environ.copy()
    env.setdefault("TRACKER_CONFIG", str(ROOT / "config.json"))
    # Keep API on the same storage backend the tracker just opened.
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.server:app",
             "--app-dir", str(BACKEND_ROOT),
             "--host", "127.0.0.1", "--port", str(port)],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.2)
        if proc.poll() is not None:
            print("Dashboard did not start (install fastapi/uvicorn or free port 8000). "
                  "Tracker still runs.")
            return None
        print(f"Dashboard → http://127.0.0.1:{port}/  (Live monitor updates automatically)")
        return proc
    except Exception as exc:  # noqa: BLE001
        print(f"Dashboard skipped: {exc}")
        return None


def _open_db(storage: dict, reset: bool):
    from src.storage import open_database
    db, info = open_database(storage, reset=reset)
    print(f"Database: {db.backend}")
    db._bootstrap_info = info  # type: ignore[attr-defined]
    return db


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Interview RTSP runner — paste the camera link only.")
    ap.add_argument(
        "url", nargs="?",
        help="RTSP URL, e.g. rtsp://user:pass@host:554/Streaming/Channels/101")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--no-dashboard", action="store_true",
                    help="Do not auto-start the web dashboard")
    ap.add_argument("--no-reset", action="store_true",
                    help="Keep existing visitor identities")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Optional safety stop after N frames (0 = unlimited)")
    args = ap.parse_args()

    url = args.url
    if not url:
        try:
            url = input("Paste RTSP URL and press Enter:\n> ").strip()
        except EOFError:
            url = ""
    url = _normalize_url(url)

    os.chdir(ROOT)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    from src.core.config import load_config
    from src.pipeline import FaceTrackingPipeline
    from src.session_svc import start_session, stop_session
    from main import run_source  # reuse the same live processing loop

    cfg = load_config(args.config)
    # Interview defaults: always show window + always publish live.jpg for dashboard.
    cfg.setdefault("output", {})
    cfg["output"]["display"] = True
    cfg["output"]["live_preview"] = cfg["output"].get("live_preview") or "data/runtime/live.jpg"
    # Live RTSP: keep reconnecting forever unless config says otherwise.
    cfg.setdefault("stream", {})
    if int(cfg["stream"].get("max_reconnects", 0) or 0) == 0:
        cfg["stream"]["max_reconnects"] = 0  # 0 = unlimited (FrameSource)

    print("=" * 60)
    print("FaceTrack interview mode")
    print(f"  RTSP     : {url}")
    print(f"  Config   : {args.config}")
    print(f"  Preview  : {cfg['output']['live_preview']}")
    print(f"  Events   : {cfg['storage']['events_log']}")
    print("  Quit     : press q in the video window, or Ctrl+C")
    print("=" * 60)

    from src.storage import log_db_open
    st = dict(cfg["storage"])
    reset = bool(st.get("reset_on_start", False)) and not args.no_reset
    db = _open_db(st, reset=reset)
    info = getattr(db, "_bootstrap_info", {"backend": db.backend, "fallback": False})
    events = start_session(
        cfg, db, reset=reset, sources=[url],
        extra_fields={"mode": "interview_rtsp", "backend": db.backend},
    )
    log_db_open(events, info)

    # Load models (same stack as main.py).
    from src.detection import HybridFaceDetector, YoloFaceDetector
    from src.recognition import FaceEmbedder

    appearance = None
    if cfg.get("appearance", {}).get("enabled", False):
        try:
            from src.recognition import AppearanceAnalyzer
            appearance = AppearanceAnalyzer(cfg["appearance"])
        except Exception:  # noqa: BLE001
            logging.exception("Appearance model failed; continuing")

    person_detector = None
    if cfg.get("away_walkers", {}).get("enabled", False):
        try:
            from src.detection import YoloPersonDetector
            person_detector = YoloPersonDetector(cfg.get("person_detector", {}))
        except Exception:  # noqa: BLE001
            logging.exception("Person detector failed; continuing")

    retina_cfg = cfg.get("retina_face") or {}
    if retina_cfg.get("enabled", True):
        try:
            detector = HybridFaceDetector(cfg["detector"], retina_cfg)
            events.log("RETINA_FACE_READY")
        except Exception:  # noqa: BLE001
            logging.exception("RetinaFace fallback unavailable; YOLO only")
            detector = YoloFaceDetector(cfg["detector"])
    else:
        detector = YoloFaceDetector(cfg["detector"])

    print("Loading InsightFace embeddings (buffalo_l)…")
    embedder = FaceEmbedder(cfg["recognition"])
    pipeline = FaceTrackingPipeline(
        cfg, detector, embedder, db, events,
        appearance=appearance, person_detector=person_detector)

    dash = None if args.no_dashboard else _start_dashboard(cfg)

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    import cv2
    frames = 0
    try:
        events.log("SESSION_VIDEO", index="1/1", source=url, unique_so_far=db.unique_count())
        print("Connecting to RTSP (TCP)… first frame may take a few seconds.")
        out = cfg["output"]
        frames = run_source(
            url, cfg, pipeline, db, events, stop,
            display=True,
            max_frames=args.max_frames,
            out_path="",  # never write annotated MP4 on live
            live_path=out.get("live_preview") or "",
            live_quality=int(out.get("live_jpeg_quality", 92)),
            live_max_width=int(out.get("live_max_width", 0) or 0),
            live_every=int(out.get("live_every_n_frames", 2)),
        )
    finally:
        try:
            pipeline.finalize()
        except Exception:  # noqa: BLE001
            logging.exception("pipeline.finalize failed during shutdown")
        cv2.destroyAllWindows()
        stop_session(db, events, mode="interview_rtsp", frames=frames)
        counts = db.event_counts()
        print(f"\nDone. Unique visitors: {db.unique_count()}"
              f"  | entries={counts.get('entry', 0)} exits={counts.get('exit', 0)}")
        print(f"Session {events.session_id}  | crops → {st['log_dir']}/{{entries,exits,faces}}/"
              f"   log → {st['events_log']}")
        db.close()
        if dash is not None and dash.poll() is None:
            dash.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
