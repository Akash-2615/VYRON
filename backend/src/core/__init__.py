"""Core utilities: config defaults/validation and the single time source.

Session helpers live in ``src.session_svc`` (shimmed as ``src.session``).
Must not import detection, pipeline, or api.
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

from . import paths  # noqa: F401
