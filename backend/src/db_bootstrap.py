"""Deprecated shim — use ``src.storage.bootstrap`` / ``src.storage`` instead.

Aliases this module name to the real implementation so existing imports and
monkeypatches keep working.
"""
from __future__ import annotations

import sys

from src.storage import bootstrap as _impl

sys.modules[__name__] = _impl
