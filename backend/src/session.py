"""Deprecated shim — use ``src.session_svc.session`` instead."""
from __future__ import annotations
import sys
from src.session_svc import session as _impl
sys.modules[__name__] = _impl
