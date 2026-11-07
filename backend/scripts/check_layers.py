#!/usr/bin/env python3
"""AST-based import layer checker for the modular layout.

Allowed edges (lower must not import higher):
  core <- storage, event_logging <- detection, recognition, tracking, video_io
       <- presence <- pipeline <- main/scripts
  api/routers -> api/services -> storage/reporting
  Nothing in src/ imports api/ or scripts/.

Exit 0 if clean; exit 1 with violations listed.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]

# Package rank: lower number = lower layer. Cross-import only toward lower/equal.
RANK: Dict[str, int] = {
    "src.core": 10,
    "src.storage": 20,
    "src.event_logging": 20,
    "src.session_svc": 25,
    "src.detection": 30,
    "src.recognition": 30,
    "src.tracking": 30,
    "src.video_io": 30,
    "src.viz": 35,
    "src.presence": 40,
    "src.reporting": 40,
    "src.pipeline": 50,
    "api": 60,
    "scripts": 70,
    "main": 70,
    "run_rtsp": 70,
}

# Legacy shim modules at src/*.py map to their owning package for ranking.
SHIM_TO_PKG = {
    "src.config": "src.core",
    "src.clock": "src.core",
    "src.session": "src.session_svc",
    "src.database": "src.storage",
    "src.db_bootstrap": "src.storage",
    "src.event_logger": "src.event_logging",
    "src.detector": "src.detection",
    "src.retina_face": "src.detection",
    "src.person_detector": "src.detection",
    "src.recognizer": "src.recognition",
    "src.gallery": "src.recognition",
    "src.appearance": "src.recognition",
    "src.tracker": "src.tracking",
    "src.stream": "src.video_io",
    "src.preview": "src.video_io",
    "src.presence": "src.presence",
    "src.counting": "src.reporting",
    "src.visualize": "src.viz",
    "src.pipeline": "src.pipeline",
    "src.tracker_job": "src.pipeline",
}


def _pkg_of(mod: str) -> Optional[str]:
    if mod in SHIM_TO_PKG:
        return SHIM_TO_PKG[mod]
    if mod in RANK:
        return mod
    for key in sorted(RANK, key=len, reverse=True):
        if mod == key or mod.startswith(key + "."):
            return key
    if mod.startswith("src."):
        # unknown src submodule — treat as pipeline-level (strict)
        return "src.pipeline"
    if mod.startswith("api"):
        return "api"
    if mod.startswith("scripts"):
        return "scripts"
    return None


def _module_name(path: Path) -> str:
    rel = path.resolve().relative_to(ROOT).with_suffix("")
    return ".".join(rel.parts)


def _imports(path: Path) -> Set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                found.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.add(node.module)
    return found


def check() -> List[str]:
    violations: List[str] = []
    py_files = list((ROOT / "src").rglob("*.py"))
    py_files += list((ROOT / "api").rglob("*.py"))
    for extra in (ROOT / "main.py", ROOT / "scripts" / "run_rtsp.py"):
        if extra.is_file():
            py_files.append(extra)
    for path in py_files:
        if path.name == "__init__.py" and path.stat().st_size == 0:
            continue
        # Skip deprecated shim files that only re-alias (they import higher by design).
        text = path.read_text(encoding="utf-8")
        if "Deprecated shim" in text and "sys.modules[__name__]" in text:
            continue
        me = _module_name(path)
        me_pkg = _pkg_of(me)
        if me_pkg is None:
            continue
        me_rank = RANK[me_pkg]
        for imp in _imports(path):
            if not (imp.startswith("src") or imp.startswith("api") or imp in ("main", "run_rtsp")
                    or imp.startswith("scripts")):
                continue
            other = _pkg_of(imp)
            if other is None:
                continue
            # src must never import api/scripts
            if me_pkg.startswith("src") and (other == "api" or other == "scripts"):
                violations.append(f"{me} imports {imp} (src must not import {other})")
                continue
            other_rank = RANK[other]
            # Dashboard job reuses main.run_source (lazy import); allow this edge.
            if me.endswith("tracker_job") and other == "main":
                continue
            if other_rank > me_rank:
                # allow api to import src.* (lower)
                if me_pkg == "api" and other.startswith("src"):
                    continue
                # allow main/run_rtsp/scripts to import anything in src/api
                if me_pkg in ("main", "run_rtsp", "scripts"):
                    continue
                violations.append(
                    f"{me} (rank {me_rank}) imports {imp} → {other} (rank {other_rank})"
                )
    return violations


def main() -> int:
    bad = check()
    if not bad:
        print("OK — layer imports look clean")
        return 0
    print(f"FAIL — {len(bad)} layer violation(s):")
    for line in bad:
        print(" ", line)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
