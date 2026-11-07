"""Deprecated shim — use ``src.recognition.appearance`` instead."""
from __future__ import annotations

import sys

from src.recognition import appearance as _impl

sys.modules[__name__] = _impl
