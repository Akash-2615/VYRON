"""Deprecated shim — use ``src.core.session`` instead."""
from __future__ import annotations

import sys

from src.core import session as _impl

sys.modules[__name__] = _impl
