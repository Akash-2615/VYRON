"""Deprecated shim — use ``src.event_logging.event_logger`` instead.

Aliases this module name to the real implementation so monkeypatches on
``src.event_logger`` still affect callers (same module object).
"""
from __future__ import annotations

import sys

from src.event_logging import event_logger as _impl

sys.modules[__name__] = _impl
