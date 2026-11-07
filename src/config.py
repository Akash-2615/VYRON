"""Deprecated shim — use ``src.core.config`` / ``src.core`` instead.

Aliases this module name to the real implementation so monkeypatches on
``src.config`` still affect callers (same module object).
"""
from __future__ import annotations

import sys

from src.core import config as _impl

sys.modules[__name__] = _impl
