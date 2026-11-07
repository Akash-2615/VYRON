"""Project paths resolved from this file's location (cwd-independent).

``PROJECT_ROOT`` is the repo root (parent of ``backend/``). Relative paths
stored in the DB and logs stay forward-slash paths relative to PROJECT_ROOT,
exactly as before the top-level layout refactor.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

log = logging.getLogger(__name__)

# backend/src/core/paths.py → parents[2]=backend, parents[3]=repo root
BACKEND_ROOT: Path = Path(__file__).resolve().parents[2]
PROJECT_ROOT: Path = BACKEND_ROOT.parent


def resolve_under_root(path: Union[str, Path], *, root: Optional[Path] = None) -> Path:
    """Absolute path: leave absolute inputs alone; else join to PROJECT_ROOT."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (root or PROJECT_ROOT) / p


def rel_to_root(path: Union[str, Path], *, root: Optional[Path] = None) -> str:
    """Forward-slash path relative to PROJECT_ROOT when possible."""
    root = root or PROJECT_ROOT
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def legacy_fallback(new_rel: str, old_candidates: Sequence[str]) -> str:
    """Prefer ``new_rel`` under PROJECT_ROOT; else first existing legacy path.

    Logs a WARNING when falling back so configs migrate off old locations.
    Returns a project-relative forward-slash path (or the new_rel default).
    """
    new_abs = resolve_under_root(new_rel)
    if new_abs.exists():
        return Path(new_rel).as_posix()
    for old in old_candidates:
        old_abs = resolve_under_root(old)
        if old_abs.exists():
            log.warning(
                "Path fallback: %s missing; using legacy %s (update config.json)",
                new_rel, old,
            )
            return Path(old).as_posix()
    return Path(new_rel).as_posix()


# Canonical relative locations (config defaults)
MODELS_DIR = "data/models"
DB_DIR = "data/db"
RUNTIME_DIR = "data/runtime"
VIDEOS_DIR = "data/videos"
DEMO_DIR = "data/demo"
LOGS_DIR = "logs"
FRONTEND_DIST = "frontend/dist"


def models_dir() -> Path:
    return resolve_under_root(MODELS_DIR)


def db_path(name: str = "visitors.db") -> Path:
    return resolve_under_root(Path(DB_DIR) / name)


def runtime_path(name: str) -> Path:
    return resolve_under_root(Path(RUNTIME_DIR) / name)


def videos_dir() -> Path:
    return resolve_under_root(VIDEOS_DIR)


def logs_dir() -> Path:
    return resolve_under_root(LOGS_DIR)


def frontend_dist() -> Path:
    return resolve_under_root(FRONTEND_DIST)


def demo_dir() -> Path:
    return resolve_under_root(DEMO_DIR)
