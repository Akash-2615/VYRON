"""Deprecated shim — use ``src.video_io.preview`` instead."""
from __future__ import annotations

import sys

from src.video_io import preview as _impl

sys.modules[__name__] = _impl
