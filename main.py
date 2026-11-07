"""Thin shim so ``python main.py ...`` still works from the repo root.

Adds ``backend/`` to ``sys.path`` and runs ``backend/main.py`` with the same argv.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(_BACKEND))
sys.argv[0] = str(_BACKEND / "main.py")
runpy.run_path(str(_BACKEND / "main.py"), run_name="__main__")
