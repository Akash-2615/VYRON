"""Deprecated shim — use ``src.storage.database`` / ``src.storage`` instead.

Aliases this module name to the real implementation so ``from src.database
import Database`` and monkeypatches keep working.
"""
from __future__ import annotations

import sys

from src.storage import database as _impl

sys.modules[__name__] = _impl
