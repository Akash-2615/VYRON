"""Deprecated shim — use ``src.detection.retina_face`` instead."""
from __future__ import annotations

import sys

from src.detection import retina_face as _impl

sys.modules[__name__] = _impl
