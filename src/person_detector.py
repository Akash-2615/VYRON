"""Deprecated shim — use ``src.detection.person_detector`` instead."""
from __future__ import annotations

import sys

from src.detection import person_detector as _impl

sys.modules[__name__] = _impl
