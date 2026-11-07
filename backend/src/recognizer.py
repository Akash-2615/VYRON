"""Deprecated shim — use ``src.recognition.recognizer`` instead."""
from __future__ import annotations

import sys

from src.recognition import recognizer as _impl

sys.modules[__name__] = _impl
