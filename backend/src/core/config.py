"""Configuration loading with defaults + light validation."""
import copy
import json
import logging
from pathlib import Path

DEFAULTS = {
    "video_source": "data/videos/video_sample1.mp4",
    "video_sources": [],
    "detection_skip_frames": 2,
    "detector": {
        "weights": "data/models/yolov8n-face.pt",
        "weights_url": "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt",
        "conf_threshold": 0.30, "imgsz": 640, "device": "auto",
    },
    # RetinaFace-family (InsightFace SCRFD/RetinaFace ONNX) as YOLO fallback + low-conf verifier.
    "retina_face": {
        "enabled": True,
        "conf_threshold": 0.5,
        "det_size": 640,
        "device": "auto",
        "verify_below_conf": 0.55,   # YOLO score below this must be confirmed by Retina
        "verify_iou": 0.30,
        "fallback_on_empty": True,  # if YOLO finds nothing, run Retina on full frame
        "model_path": "",           # empty → ~/.insightface/models/buffalo_l/det_10g.onnx
    },
    "recognition": {
        "model_pack": "buffalo_l", "det_size": 320, "match_threshold": 0.45,
        "soft_match_margin": 0.10, "register_min_det_score": 0.40,
        "mismatch_margin": 0.08,
        "max_embeddings_per_face": 5, "min_embeds_to_register": 2,
        "embed_retry_interval": 2, "max_embed_attempts": 10,
    },
    "quality": {"min_face_px": 45, "min_blur_var": 35.0},
    "appearance": {
        "enabled": False,
        "model": "ViT-B-32",
        "pretrained": "openai",
        "device": "auto",
        "min_crop_px": 48,
        "describe_on_register": True,
    },
    "away_walkers": {
        "enabled": False,
        "require_facing_away": False,
        "away_score_min": 0.35,
        # 1 = same cadence as face detect. Higher values need away_tracker interval
        # = (skip+1)*N (wired in FaceTrackingPipeline) or tracks die before confirm.
        "detect_every_n": 1,
        "describe_away": True,
    },
    "person_detector": {
        "weights": "data/models/yolov8n.pt",
        "conf_threshold": 0.45,
        "imgsz": 640,
        "device": "auto",
        "min_height_px": 70,
        "min_width_px": 24,
        "min_aspect": 1.15,
        "max_aspect": 5.0,
    },
    "away_tracker": {"high_conf": 0.4, "new_track_conf": 0.4, "iou_threshold": 0.25,
                     "min_hits": 2, "max_lost_frames": 45},
    "tracker": {"high_conf": 0.5, "new_track_conf": 0.5, "iou_threshold": 0.25,
                "min_hits": 2, "max_lost_frames": 45,
                "reentry_grace_frames": 75, "reentry_grace_sec": 25.0},
    "presence": {
        "exit_grace_seconds": 10,
        "edge_exit_grace_seconds": 2,
        "stationary_grace_seconds": 45,
        "occlusion_grace_multiplier": 2.0,
        "edge_margin_px": 24,
        "stationary_speed_px_s": 15,
        "resume_spatial_iou": 0.2,
        "resume_threshold_margin": 0.05,
        "reverify_interval_seconds": 5,
        "occlusion_iou": 0.3,
        "mass_loss_ratio": 0.6,
        "max_stream_gap_seconds": 20,
        "exit_zones": [],
    },
    "timestamps": "auto",  # auto | wall | video (tests); production uses wall processing time
    "logging": {
        "level": "INFO",                 # INFO | DEBUG
        "separate_away_log": True,       # AWAY_* → logs/away_events.log
        "debug_events": False,           # FACE_ROI_REFINE, TRACK_LOST, EMBED_NO_FACE, …
    },
    "storage": {
        "backend": "postgres",
        # Postgres DSN must come from DATABASE_URL / TRACKER_DSN — never commit a password.
        "postgres": {},
        "log_dir": "logs",
        "events_log": "logs/events.log",
        "reset_on_start": False,
    },
    "stream": {
        "reconnect_delay_sec": 3, "max_reconnects": 0, "read_timeout_sec": 10,
        "realtime": True, "stream_gap_seconds": 1.5,
        # When true, file playback skips ahead if the CPU is behind (keeps live UI near realtime).
        # Prefer false for crowded clips — aggressive dropping looks like fast-forward and breaks IDs.
        "catch_up_drop": False,
        # Downscale wide sources before detect/track (0 = full resolution).
        "process_max_width": 1280,
    },
    "output": {
        "display": False,
        "annotated_video": "data/runtime/annotated.mp4",
        "live_preview": "data/runtime/live.jpg",
        "live_jpeg_quality": 72,
        "live_max_width": 960,
        "live_every_n_frames": 2,
    },
    # Optional dull/dark assist for the face detector only (OFF by default).
    "preprocess": {
        "enabled": False,
        "process_at_detector_resolution": True,
        "mode": "async",
        "max_wait_ms": 50,
        "auto_gate": {
            "enabled": True,
            "brightness_below": 90,
            "contrast_below": 35,
            "mode": "any",
        },
        "denoise": {"enabled": False, "method": "gaussian", "strength": 0.35},
        "white_balance": {"enabled": False, "strength": 0.6, "max_gain": 1.5},
        "gamma": {"enabled": False, "gamma": 0.75, "auto_target_mean": 0},
        "clahe": {"enabled": False, "clip_limit": 2.0, "tile_grid": 8},
        "sharpen": {"enabled": False, "amount": 0.3, "sigma": 1.0},
        "quality_assist": {"enabled": False},
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _pick_existing(configured: str, *, preferred: str, legacy: list) -> str:
    """Keep config key names; prefer an existing file under PROJECT_ROOT."""
    from src.core.paths import resolve_under_root
    log = logging.getLogger(__name__)
    ordered = []
    for c in (configured, preferred, *legacy):
        if c and c not in ordered:
            ordered.append(c)
    for i, rel in enumerate(ordered):
        if resolve_under_root(rel).exists():
            if i > 0:
                log.warning(
                    "Path fallback: %s missing; using %s (update config.json)",
                    configured or preferred, rel)
            return Path(rel).as_posix()
    return Path(configured or preferred).as_posix()


def load_config(path: str = "config.json") -> dict:
    from src.core.paths import PROJECT_ROOT

    cfg = copy.deepcopy(DEFAULTS)
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if p.exists():
        with p.open() as f:
            cfg = _merge(cfg, json.load(f))
    else:
        logging.getLogger(__name__).warning("Config %s not found, using defaults", path)

    skip = int(cfg["detection_skip_frames"])
    if skip < 0:
        raise ValueError("detection_skip_frames must be >= 0")
    cfg["detection_skip_frames"] = skip

    # A track must be allowed to coast across at least two detection cycles,
    # otherwise a single missed detection would produce a false exit.
    min_lost = 2 * (skip + 1) + 2
    if cfg["tracker"]["max_lost_frames"] < min_lost:
        logging.getLogger(__name__).warning(
            "tracker.max_lost_frames raised %s -> %s to suit detection_skip_frames=%s",
            cfg["tracker"]["max_lost_frames"], min_lost, skip)
        cfg["tracker"]["max_lost_frames"] = min_lost

    presence = cfg.setdefault("presence", {})
    for key in ("exit_grace_seconds", "edge_exit_grace_seconds", "stationary_grace_seconds",
                "occlusion_grace_multiplier", "edge_margin_px", "stationary_speed_px_s",
                "resume_spatial_iou", "reverify_interval_seconds", "occlusion_iou",
                "mass_loss_ratio", "max_stream_gap_seconds"):
        if key in presence and float(presence[key]) <= 0:
            raise ValueError(f"presence.{key} must be > 0")
    eg = float(presence.get("edge_exit_grace_seconds", 2))
    xg = float(presence.get("exit_grace_seconds", 10))
    sg = float(presence.get("stationary_grace_seconds", 45))
    if not (sg >= xg >= eg):
        raise ValueError(
            "presence requires stationary_grace_seconds >= exit_grace_seconds >= edge_exit_grace_seconds")

    # One-release fallbacks when files still live at pre-structure paths.
    det = cfg.setdefault("detector", {})
    det["weights"] = _pick_existing(
        det.get("weights") or "",
        preferred="data/models/yolov8n-face.pt",
        legacy=["models/yolov8n-face.pt"],
    )
    person = cfg.setdefault("person_detector", {})
    person["weights"] = _pick_existing(
        person.get("weights") or "",
        preferred="data/models/yolov8n.pt",
        legacy=["yolov8n.pt", "models/yolov8n.pt"],
    )
    out = cfg.setdefault("output", {})
    if (out.get("live_preview") or "") in ("", "output/live.jpg"):
        out["live_preview"] = "data/runtime/live.jpg"
    if (out.get("annotated_video") or "") in ("output/annotated.mp4",):
        out["annotated_video"] = "data/runtime/annotated.mp4"
    legacy_src = cfg.get("video_source") or ""
    if legacy_src in ("data/sample_video.mp4", "data/videos/sample_video.mp4"):
        cfg["video_source"] = "data/videos/video_sample1.mp4"
    return cfg
