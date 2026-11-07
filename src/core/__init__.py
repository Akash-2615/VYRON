"""Core utilities: config defaults/validation and the single time source.

Must not import detection, pipeline, storage, or api.
"""
from .clock import iso_ms, is_anomalous, mono, wall_from_mono, wall_now
from .config import DEFAULTS, load_config

__all__ = [
    "DEFAULTS",
    "load_config",
    "wall_now",
    "mono",
    "iso_ms",
    "wall_from_mono",
    "is_anomalous",
]
