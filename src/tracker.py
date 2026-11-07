"""Deprecated shim — use ``src.tracking.tracker`` instead."""
from __future__ import annotations

import sys

from src.tracking import tracker as _impl

sys.modules[__name__] = _impl
