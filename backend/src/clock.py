"""Deprecated shim — use ``src.core.clock`` / ``src.core`` instead.

Aliases this module name to the real implementation so monkeypatches on
``src.clock`` still affect callers (same module object).
"""
from __future__ import annotations

import sys

from src.core import clock as _impl

sys.modules[__name__] = _impl
