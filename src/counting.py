"""Deprecated shim — use ``src.reporting.counting`` instead."""
from __future__ import annotations

import sys

from src.reporting import counting as _impl

sys.modules[__name__] = _impl
