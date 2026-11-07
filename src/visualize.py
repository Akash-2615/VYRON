"""Deprecated shim — use ``src.viz.overlay`` instead."""
from __future__ import annotations

import sys

from src.viz import overlay as _impl

sys.modules[__name__] = _impl
