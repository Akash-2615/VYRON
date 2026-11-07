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

from src.core.config import load_config
from src.pipeline import FaceTrackingPipeline
from src.session_svc import start_session, stop_session
from src.video_io import FrameSource, default_publisher, publish_live_stats
from src.viz import draw_overlay


def parse_args():
    p = argparse.ArgumentParser(description="Intelligent face tracker + unique visitor counter")
    p.add_argument("--config", default="config.json")
    p.add_argument("--source", help="Override video_source (file path, rtsp:// URL or webcam index)")
    p.add_argument("--sources", nargs="+", help="Process several videos in order as ONE session (combined unique count)")
    p.add_argument("--no-reset", action="store_true", help="Keep existing DB/identities (overrides reset_on_start)")
    p.add_argument(
        "--fresh", action="store_true",
        help="Archive events.log + away_events.log, reset the active DB, then start clean",
    )
    p.add_argument("--display", action="store_true", help="Show live window (press q to quit)")
    p.add_argument("--max-frames", type=int, default=0, help="Stop each source after N frames (0 = no limit)")
    return p.parse_args()


def annotated_path(base: str, source: str, multi: bool) -> str:
    """output/annotated.mp4 -> output/annotated_<video-stem>.mp4 when several sources are used."""
    if not base or not multi:
        return base
    b = Path(base)
    return str(b.with_name(f"{b.stem}_{Path(source).stem}{b.suffix}"))


def _scale_tracks(tracks, scale: float):
    """Shallow-copy tracks with boxes scaled for a downsized preview frame."""
    if not tracks or scale == 1.0:
        return tracks
    out = []
    for t in tracks:
        try:
            c = t.copy() if hasattr(t, "copy") else None
        except Exception:  # noqa: BLE001
            c = None
        if c is None:
            # Track objects are mutable dataclasses / simple namespaces — clone box only.
            import copy
            c = copy.copy(t)
        box = getattr(c, "box", None)
        if box is not None and len(box) >= 4:
            c.box = [float(box[0]) * scale, float(box[1]) * scale,
                     float(box[2]) * scale, float(box[3]) * scale]
        out.append(c)
    return out


def run_source(source, cfg, pipeline, db, events, stop, display, max_frames, out_path,
               live_path: str = "", live_quality: int = 92, live_max_width: int = 0,
               live_every: int = 2) -> int:
    src = FrameSource(source, cfg["stream"]).start()
    # Annotated MP4 at full CCTV resolution is a major lag source — skip on live / when unset.
    if src.is_live or not out_path:
        out_path = ""
    writer, idx, t0, fps = None, 0, time.time(), 0.0
    unique_cached = db.unique_count()
    away_cached = db.away_stats().get("away_unique", 0)
    # File playback at real wall-clock speed (match source FPS). Live RTSP already paces itself.
    realtime = (not src.is_live) and bool(cfg.get("stream", {}).get("realtime", True))
    video_fps = 25.0
    if (not src.is_live) and src._cap is not None:
        probed = float(src._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if probed > 1.0:
            video_fps = probed
    frame_dt = 1.0 / video_fps
    skipped = 0
    events.log("SOURCE_START", source=source, live=src.is_live, realtime=realtime,
               video_fps=f"{video_fps:.2f}")
    unique_at_start = db.unique_count()
    pipeline.set_timing(video_fps, live=src.is_live)
    gap_thr = float(cfg.get("stream", {}).get("stream_gap_seconds", 1.5))
    last_read_wall = time.time()
    end_reason = "eof"

    # Optional preprocess (detector input only). OFF => identical path to before.
    from src.detection.preprocess import get_preprocess_runtime, PreprocessSettings
    pp_rt = get_preprocess_runtime()
    pp_rt.store.set_base_from_config(cfg.get("preprocess"))
    pp_rt.start_worker()
    pp0 = pp_rt.store.get()
    events.log("PREPROCESS_ENABLED",
               enabled=bool(pp0.enabled), mode=pp0.mode,
               detector_res=bool(pp0.process_at_detector_resolution),
               version=pp0.version)
    det_imgsz = int((cfg.get("detector") or {}).get("imgsz") or 640)
    pp_fallback = 0
    pp_ms_avg = 0.0
    pp_ready = 100.0

    def _count_snapshot(**extra):
        counts = db.event_counts()
        events.log(
            "COUNT_SNAPSHOT",
            unique_total=db.unique_count(),
            entries=counts.get("entry", 0),
            exits=counts.get("exit", 0),
            **extra,
        )

    try:
        while not stop["flag"]:
            ok, frame = src.read()
            now_wall = time.time()
            gap = now_wall - last_read_wall
            if ok and gap >= gap_thr and src.is_live:
                pipeline.notify_stream_gap(gap)
            last_read_wall = now_wall
            if not ok:
                end_reason = "disconnect" if src.is_live else "eof"
                break
            # Downscale 4K/CCTV before YOLO so live FPS stays usable.
            proc_w = int((cfg.get("stream") or {}).get("process_max_width", 0) or 0)
            if proc_w > 0 and frame.shape[1] > proc_w:
                scale = proc_w / float(frame.shape[1])
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # Preprocess for detector only (never blocks live; file may wait max_wait_ms).
            detect_frame = None
            detect_scale = 1.0
            pp = pp_rt.store.get()
            pipeline._preprocess_settings = pp  # quality_assist reads this
            run_det = idx % (int(cfg.get("detection_skip_frames", 0)) + 1) == 0
            if pp.enabled and pp.any_step_on() and run_det:
                if pp.mode == "sync":
                    detect_frame, detect_scale, _gate = pp_rt.enhance_sync(
                        frame, pp, detector_imgsz=det_imgsz)
                else:
                    pp_rt.submit(idx, frame, pp, det_imgsz)
                    wait_ms = 0.0 if src.is_live else float(pp.max_wait_ms)
                    hit = pp_rt.take(idx, wait_ms=wait_ms)
                    if hit is None:
                        pp_rt.note_fallback()
                        pp_fallback += 1
                    else:
                        detect_frame, detect_scale, _gate = hit
                m = pp_rt.metrics
                pp_ms_avg = float(m.get("total_avg_ms") or 0.0)
                pp_ready = float(m.get("ready_pct") or 100.0)

            try:
                # Wall-clock processing time for files and live (mono used only for durations).
                tracks, away_tracks = pipeline.process_frame(
                    frame, idx, ts=None,
                    detect_frame=detect_frame, detect_scale=detect_scale)
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
                events.log(
                    "STATS", source=source, frames=idx, fps=f"{fps:.1f}",
                    unique=unique_cached, away=away_cached, skipped=skipped,
                    away_summary=events.away_summary().replace(" ", ","),
                    preprocess=("on" if pp.enabled else "off"),
                    pp_ms=f"{pp_ms_avg:.1f}",
                    ready=f"{pp_ready:.0f}",
                    fallback=pp_fallback,
                )
                _count_snapshot(source=source, frames=idx)

            # Real-time in-frame counts (cheap; every other frame is enough).
            if live_path and idx % 2 == 0:
                counts = pipeline.in_frame_counts(tracks, away_tracks)
                shown_fps = float(fps or video_fps) or 25.0
                counts["fps"] = round(shown_fps, 1)
                counts["width"] = int(frame.shape[1])
                counts["height"] = int(frame.shape[0])
                # Real catch-up lag vs source clock (more honest than 1/fps).
                if realtime:
                    lag_ms = max(0.0, (time.time() - t0) - (idx * frame_dt)) * 1000.0
                    counts["latency_ms"] = int(round(lag_ms))
                else:
                    counts["latency_ms"] = int(round(1000.0 / max(shown_fps, 1.0)))
                if live_max_width and frame.shape[1] > live_max_width:
                    sc = live_max_width / float(frame.shape[1])
                    counts["preview_width"] = int(frame.shape[1] * sc)
                    counts["preview_height"] = int(frame.shape[0] * sc)
                stats_path = str(Path(live_path).with_name("live_stats.json"))
                publish_live_stats(counts, stats_path)

            need_vis = bool(out_path or display or live_path)
            if need_vis:
                do_live = bool(live_path) and (idx % max(1, live_every) == 0)
                if do_live or out_path or display:
                    # Face: coast across detection skips. Away: person YOLO is rarer.
                    skip = int(cfg.get("detection_skip_frames", 0))
                    away_every = max(1, int(cfg.get("away_walkers", {}).get("detect_every_n", 1)))
                    box_age = max(1, skip)
                    away_age = max(box_age, (skip + 1) * away_every)
                    # Draw live preview on a downscaled copy — 4K overlay+JPEG was the lag.
                    draw_frame = frame
                    draw_tracks, draw_away = tracks, away_tracks
                    preview_max = int(live_max_width or 0)
                    if do_live and preview_max and frame.shape[1] > preview_max and not out_path and not display:
                        scale = preview_max / float(frame.shape[1])
                        draw_frame = cv2.resize(
                            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                        draw_tracks = _scale_tracks(tracks, scale)
                        draw_away = _scale_tracks(away_tracks, scale)
                    vis = draw_overlay(draw_frame, draw_tracks, unique_cached, fps,
                                      away_tracks=draw_away, away_unique=away_cached,
                                      max_box_age=box_age, away_max_box_age=away_age)
                    if do_live:
                        # Already sized for preview when we downscaled above.
                        pub_w = 0 if draw_frame is not frame else live_max_width
                        # Non-blocking: JPEG/disk on a background thread (drop stale).
                        default_publisher().publish(vis, live_path, live_quality, pub_w)
                    if out_path or display:
                        full = vis if draw_frame is frame else draw_overlay(
                            frame, tracks, unique_cached, fps,
                            away_tracks=away_tracks, away_unique=away_cached,
                            max_box_age=box_age, away_max_box_age=away_age)
                        if out_path:
                            if writer is None:
                                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                                h, w = full.shape[:2]
                                writer = cv2.VideoWriter(
                                    out_path, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (w, h))
                            writer.write(full)
                        if display:
                            cv2.imshow("Face Tracker", full)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                stop["flag"] = True

            # Pace to source FPS: sleep when ahead so we never play faster than 1×.
            # Mild catch-up when far behind keeps the live UI near realtime without
            # skipping entire scenes (hard cap on drops per step).
            if realtime:
                target = idx * frame_dt
                elapsed = time.time() - t0
                ahead = target - elapsed
                if ahead > 0.001:
                    time.sleep(min(ahead, frame_dt * 2))
                elif (bool((cfg.get("stream") or {}).get("catch_up_drop", True))
                      and elapsed - target > 1.0 and src._cap is not None):
                    drop = min(int((elapsed - target) * video_fps * 0.35) + 1, 8)
                    for _ in range(max(0, drop)):
                        if not src._cap.grab():
                            break
                        idx += 1
                        skipped += 1
                    # Jump tracker time so soft-exit grace still tracks wall-ish progress.
                    pipeline.frame_idx = idx - 1
            if max_frames and idx >= max_frames:
                end_reason = "stopped"
                break
        if stop["flag"]:
            end_reason = "stopped"
    except Exception:
        end_reason = "error"
        raise
    finally:
        # Finalize first so COUNT_SNAPSHOT / STREAM_END see closed visits for this source.
        try:
            pipeline.finalize()
        except Exception:  # noqa: BLE001
            logging.exception("pipeline.finalize failed at source end")
        unique_now = db.unique_count()
        events.log(
            "STREAM_END",
            source=source, frames=idx, skipped=skipped, reason=end_reason,
            new_faces=max(0, unique_now - unique_at_start),
            unique_so_far=unique_now,
        )
        _count_snapshot(source=source, frames=idx, reason=end_reason)
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
    if args.fresh:
        from src.session_svc import prepare_fresh
        report = prepare_fresh(cfg)
        print("--fresh: clean start", flush=True)
        for pth in report.get("archived") or []:
            print(f"  archived log → {pth}", flush=True)
        if not report.get("archived"):
            print("  archived log → (nothing to archive)", flush=True)
        db_info_pre = report.get("db_reset") or {}
        print(f"  reset DB → backend={db_info_pre.get('backend')} "
              f"path={db_info_pre.get('path')} "
              f"unique_after={db_info_pre.get('unique_after')}", flush=True)
        reset = True
    if args.no_reset and not args.fresh:
        reset = False
    # reset happens ONCE per session, never between videos
    from src.storage import log_db_open, open_database
    db, db_info = open_database(st, reset=reset)  # Postgres only; raises if unavailable
    events = start_session(cfg, db, reset=reset, sources=sources)
    log_db_open(events, db_info)

    from src.detection import HybridFaceDetector, YoloFaceDetector  # heavy imports stay here
    from src.recognition import FaceEmbedder
    appearance = None
    app_cfg = cfg.get("appearance", {}) or {}
    away_cfg = cfg.get("away_walkers", {}) or {}
    # Skip loading CLIP when nothing will call describe (saves RAM + startup on crowded runs).
    need_appearance = bool(app_cfg.get("enabled", False)) and (
        bool(app_cfg.get("describe_on_register", True))
        or bool(away_cfg.get("describe_away", False))
        or bool(away_cfg.get("require_facing_away", False))
    )
    if need_appearance:
        try:
            from src.recognition import AppearanceAnalyzer
            appearance = AppearanceAnalyzer(app_cfg)
        except Exception:  # noqa: BLE001
            logging.exception("Appearance model failed to load; continuing without clothing tags")
    person_detector = None
    if cfg.get("away_walkers", {}).get("enabled", False):
        try:
            from src.detection import YoloPersonDetector
            person_detector = YoloPersonDetector(cfg.get("person_detector", {}))
        except Exception:  # noqa: BLE001
            logging.exception("Person detector failed to load; away-walker KPI disabled")

    retina_cfg = cfg.get("retina_face") or {}
    if retina_cfg.get("enabled", True):
        try:
            detector = HybridFaceDetector(cfg["detector"], retina_cfg)
            events.log("RETINA_FACE_READY",
                       verify_below=retina_cfg.get("verify_below_conf", 0.55),
                       fallback_on_empty=retina_cfg.get("fallback_on_empty", True))
        except Exception:  # noqa: BLE001
            logging.exception("RetinaFace fallback failed to load; using YOLO only")
            events.log("RETINA_FACE_UNAVAILABLE", level=logging.WARNING)
            detector = YoloFaceDetector(cfg["detector"])
    else:
        detector = YoloFaceDetector(cfg["detector"])

    pipeline = FaceTrackingPipeline(cfg, detector,
                                    FaceEmbedder(cfg["recognition"]), db, events,
                                    appearance=appearance, person_detector=person_detector)

    stop = {"flag": False}

    def _request_stop(*_args):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

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
        stop_session(db, events, videos=len(sources), frames=total)
        print(f"\nUnique visitors (all videos combined): {db.unique_count()}")
        print(f"Session {events.session_id} stopped.")
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
