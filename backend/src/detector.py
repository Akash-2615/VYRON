"""Deprecated shim — use ``src.detection.detector`` instead."""
from __future__ import annotations

import sys

from src.detection import detector as _impl

sys.modules[__name__] = _impl
