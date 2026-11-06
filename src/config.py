"""Configuration loading with defaults + light validation."""
import copy
import json
import logging
from pathlib import Path

DEFAULTS = {
    "video_source": "data/sample_video.mp4",
    "video_sources": [],
    "detection_skip_frames": 2,
    "detector": {
        "weights": "models/yolov8n-face.pt",
        "weights_url": "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt",
        "conf_threshold": 0.30, "imgsz": 640, "device": "auto",
    },
    "recognition": {
        "model_pack": "buffalo_l", "det_size": 320, "match_threshold": 0.45,
        "max_embeddings_per_face": 5, "min_embeds_to_register": 2,
        "embed_retry_interval": 2, "max_embed_attempts": 10,
    },
    "quality": {"min_face_px": 45, "min_blur_var": 35.0},
    "appearance": {
        "enabled": True,
        "model": "ViT-B-32",
        "pretrained": "openai",
        "device": "auto",
        "min_crop_px": 48,
    },
    "away_walkers": {
        "enabled": True,
        "require_facing_away": True,
        "away_score_min": 0.35,
    },
    "person_detector": {
        "weights": "yolov8n.pt",
        "conf_threshold": 0.35,
        "imgsz": 640,
        "device": "auto",
        "min_height_px": 60,
    },
    "away_tracker": {"high_conf": 0.4, "new_track_conf": 0.4, "iou_threshold": 0.25,
                     "min_hits": 2, "max_lost_frames": 25},
    "tracker": {"high_conf": 0.5, "new_track_conf": 0.5, "iou_threshold": 0.25,
                "min_hits": 2, "max_lost_frames": 20},
    "storage": {"db_path": "data/visitors.db", "log_dir": "logs",
                "events_log": "logs/events.log", "reset_on_start": True},
    "stream": {"reconnect_delay_sec": 3, "max_reconnects": 0, "read_timeout_sec": 10},
    "output": {
        "display": False,
        "annotated_video": "output/annotated.mp4",
        "live_preview": "output/live.jpg",
        "live_jpeg_quality": 55,
        "live_max_width": 960,
        "live_every_n_frames": 2,
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


def load_config(path: str = "config.json") -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    p = Path(path)
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
    return cfg
