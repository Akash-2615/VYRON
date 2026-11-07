"""Deprecated shim — use ``src.pipeline.tracker_job`` instead."""
from __future__ import annotations

import sys

from src.pipeline import tracker_job as _impl

sys.modules[__name__] = _impl
