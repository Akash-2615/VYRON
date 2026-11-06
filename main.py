"""Entry point.

Single video / RTSP:   python main.py --source rtsp://...  [--display]
Several videos, ONE combined unique count (shared DB + face gallery):
                       python main.py --sources data/v1.mp4 data/v2.mp4 data/v3.mp4
Or list them in config.json as "video_sources": [...]
"""
import argparse
import logging
import signal
import time
from pathlib import Path

import cv2

from src.config import load_config
from src.database import Database
from src.event_logger import EventLogger
from src.pipeline import FaceTrackingPipeline
from src.stream import FrameSource
from src.preview import publish_preview
from src.visualize import draw_overlay


def parse_args():
    p = argparse.ArgumentParser(description="Intelligent face tracker + unique visitor counter")
    p.add_argument("--config", default="config.json")
    p.add_argument("--source", help="Override video_source (file path, rtsp:// URL or webcam index)")
    p.add_argument("--sources", nargs="+", help="Process several videos in order as ONE session (combined unique count)")
    p.add_argument("--no-reset", action="store_true", help="Keep existing DB/identities (overrides reset_on_start)")
    p.add_argument("--display", action="store_true", help="Show live window (press q to quit)")
    p.add_argument("--max-frames", type=int, default=0, help="Stop each source after N frames (0 = no limit)")
    return p.parse_args()


def annotated_path(base: str, source: str, multi: bool) -> str:
    """output/annotated.mp4 -> output/annotated_<video-stem>.mp4 when several sources are used."""
    if not base or not multi:
        return base
    b = Path(base)
    return str(b.with_name(f"{b.stem}_{Path(source).stem}{b.suffix}"))


def run_source(source, cfg, pipeline, db, events, stop, display, max_frames, out_path,
               live_path: str = "", live_quality: int = 55, live_max_width: int = 960,
               live_every: int = 2) -> int:
    src = FrameSource(source, cfg["stream"]).start()
    # Annotated MP4 at full CCTV resolution is a major lag source on live RTSP — skip it.
    if src.is_live:
        out_path = ""
    writer, idx, t0, fps = None, 0, time.time(), 0.0
    unique_cached = db.unique_count()
    away_cached = db.away_stats().get("away_unique", 0)
    events.log("SOURCE_START", source=source, live=src.is_live)
    try:
        while not stop["flag"]:
            ok, frame = src.read()
            if not ok:
                events.log("STREAM_END", source=source, frames=idx)
                break
            try:
                tracks, away_tracks = pipeline.process_frame(frame, idx)
            except Exception:                      # keep running on a bad frame
                logging.exception("Frame %d failed", idx)
                idx += 1
                continue
            idx += 1
            if idx % 30 == 0:
                fps = idx / max(time.time() - t0, 1e-6)
                unique_cached = db.unique_count()
                away_cached = db.away_stats().get("away_unique", 0)
            if idx % 300 == 0:
                events.log("STATS", source=source, frames=idx, fps=f"{fps:.1f}",
                           unique=unique_cached, away=away_cached)

            need_vis = bool(out_path or display or live_path)
            if need_vis:
                # Live preview: draw + encode only every N frames (dashboard does not need 25 FPS).
                do_live = bool(live_path) and (idx % max(1, live_every) == 0)
                if do_live or out_path or display:
                    vis = draw_overlay(frame, tracks, unique_cached, fps,
                                      away_tracks=away_tracks, away_unique=away_cached)
                    if do_live:
                        publish_preview(vis, live_path, live_quality, live_max_width)
                    if out_path:
                        if writer is None:
                            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                            h, w = vis.shape[:2]
                            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
                        writer.write(vis)
                    if display:
                        cv2.imshow("Face Tracker", vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            stop["flag"] = True
            if max_frames and idx >= max_frames:
                break
    finally:
        pipeline.finalize()      # close this video's open visits; identities/gallery are kept for the next video
        src.release()
        if writer:
            writer.release()
    return idx


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if args.sources:
        sources = args.sources
    elif args.source:
        sources = [args.source]
    else:
        sources = cfg.get("video_sources") or [cfg["video_source"]]
    multi = len(sources) > 1
    display = args.display or cfg["output"]["display"]

    st = cfg["storage"]
    reset = st["reset_on_start"] and not args.no_reset
    db = Database(st["db_path"], reset=reset)        # reset happens ONCE per session, never between videos
    events = EventLogger(st["log_dir"], st["events_log"])
    events.log("SYSTEM_START", sources=len(sources), skip=cfg["detection_skip_frames"], db_reset=reset)

    from src.detector import YoloFaceDetector      # heavy imports kept here so tests don't need them
    from src.recognizer import FaceEmbedder
    appearance = None
    if cfg.get("appearance", {}).get("enabled", False):
        try:
            from src.appearance import AppearanceAnalyzer
            appearance = AppearanceAnalyzer(cfg["appearance"])
        except Exception:  # noqa: BLE001
            logging.exception("Appearance model failed to load; continuing without clothing tags")
    person_detector = None
    if cfg.get("away_walkers", {}).get("enabled", True):
        try:
            from src.person_detector import YoloPersonDetector
            person_detector = YoloPersonDetector(cfg.get("person_detector", {}))
        except Exception:  # noqa: BLE001
            logging.exception("Person detector failed to load; away-walker KPI disabled")
    pipeline = FaceTrackingPipeline(cfg, YoloFaceDetector(cfg["detector"]),
                                    FaceEmbedder(cfg["recognition"]), db, events,
                                    appearance=appearance, person_detector=person_detector)

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    total = 0
    try:
        for i, source in enumerate(sources, 1):
            if stop["flag"]:
                break
            events.log("SESSION_VIDEO", index=f"{i}/{len(sources)}", source=source,
                       unique_so_far=db.unique_count())
            out = cfg["output"]
            total += run_source(
                source, cfg, pipeline, db, events, stop, display, args.max_frames,
                annotated_path(out.get("annotated_video"), source, multi),
                live_path=out.get("live_preview") or "",
                live_quality=int(out.get("live_jpeg_quality", 55)),
                live_max_width=int(out.get("live_max_width", 960)),
                live_every=int(out.get("live_every_n_frames", 2)),
            )
    finally:
        pipeline.finalize()
        cv2.destroyAllWindows()
        counts = db.event_counts()
        away = db.away_stats()
        events.log("SYSTEM_STOP", videos=len(sources), frames=total, unique_visitors=db.unique_count(),
                   entries=counts.get("entry", 0), exits=counts.get("exit", 0),
                   away_unique=away.get("away_unique", 0), away_entries=away.get("away_entries", 0))
        print(f"\nUnique visitors (all videos combined): {db.unique_count()}"
              f"  | entries={counts.get('entry', 0)} exits={counts.get('exit', 0)}")
        print(f"Away walkers (backs / facing away): {away.get('away_unique', 0)}"
              f"  | entries={away.get('away_entries', 0)} exits={away.get('away_exits', 0)}"
              f"  | with clothes logged={away.get('away_with_clothes', 0)}")
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
