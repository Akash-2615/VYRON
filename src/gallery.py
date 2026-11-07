"""Deprecated shim — use ``src.recognition.gallery`` instead."""
from __future__ import annotations

import sys

from src.recognition import gallery as _impl

sys.modules[__name__] = _impl
