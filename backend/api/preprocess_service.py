"""Read/write helpers for optional video preprocessing settings.

The only API write path: ``data/runtime/preprocess.json`` (never the DB).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.core.paths import runtime_path
from src.detection.preprocess import (
    PRESETS,
    STEP_CATALOG,
    PreprocessSettings,
    default_preprocess_dict,
    get_preprocess_runtime,
)


def override_path() -> Path:
    return runtime_path("preprocess.json")


def status_path() -> Path:
    return runtime_path("preprocess_status.json")


def preview_path() -> Path:
    return runtime_path("preprocess_preview.jpg")


def load_status() -> dict:
    p = status_path()
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def current_payload(config_block: Optional[dict] = None,
                    *, client_host: Optional[str] = None,
                    admin_header: Optional[str] = None) -> dict:
    rt = get_preprocess_runtime()
    if config_block is not None:
        rt.store.set_base_from_config(config_block)
    else:
        rt.store.reload()
    settings = rt.store.get()
    source = "override" if override_path().is_file() else "config"
    defaults = default_preprocess_dict()
    # Writable probe must not consume the write rate-limit budget.
    writable = _writable_probe(client_host, admin_header)
    return {
        "enabled": settings.enabled,
        "settings": settings.to_dict(),
        "defaults": defaults,
        "step_catalog": STEP_CATALOG,
        "presets": PRESETS,
        "metrics": {**rt.metrics, **load_status()},
        "writable": writable,
        "version": settings.version,
        "source": source,
    }


_BOUNDS = {
    ("max_wait_ms",): (0, 500),
    ("auto_gate", "brightness_below"): (0, 255),
    ("auto_gate", "contrast_below"): (0, 128),
    ("denoise", "strength"): (0, 1),
    ("white_balance", "strength"): (0, 1),
    ("white_balance", "max_gain"): (1.0, 2.5),
    ("gamma", "gamma"): (0.5, 2.0),
    ("gamma", "auto_target_mean"): (0, 255),
    ("clahe", "clip_limit"): (1.0, 4.0),
    ("clahe", "tile_grid"): (4, 16),
    ("sharpen", "amount"): (0, 1),
    ("sharpen", "sigma"): (0.5, 2.0),
}


def _reject_out_of_bounds(merged: dict) -> None:
    for path, (lo, hi) in _BOUNDS.items():
        cur: Any = merged
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                cur = None
                break
            cur = cur[key]
        if cur is None:
            continue
        try:
            val = float(cur)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{'.'.join(path)} must be a number") from exc
        if val < lo or val > hi:
            raise ValueError(f"{'.'.join(path)}={val} out of bounds [{lo}, {hi}]")


def validate_and_merge(partial: dict, base: Optional[dict] = None) -> PreprocessSettings:
    """Merge partial update into current settings; raises ValueError on bad types."""
    if not isinstance(partial, dict):
        raise ValueError("body must be a JSON object")
    partial = dict(partial)
    # Reject unknown top-level keys (typos)
    allowed = set(default_preprocess_dict()) | {"preset"}
    unknown = [k for k in partial if k not in allowed]
    if unknown:
        raise ValueError(f"unknown keys: {unknown}")
    cur = dict(base or get_preprocess_runtime().store.get().to_dict())
    if "preset" in partial:
        name = str(partial.pop("preset"))
        if name not in PRESETS:
            raise ValueError(f"unknown preset: {name}")
        preset = PRESETS[name]
        for k, v in preset.items():
            if isinstance(v, dict) and isinstance(cur.get(k), dict):
                cur[k] = {**cur[k], **v}
            else:
                cur[k] = v
    for k, v in partial.items():
        if isinstance(v, dict) and isinstance(cur.get(k), dict):
            cur[k] = {**cur[k], **v}
        else:
            cur[k] = v
    _reject_out_of_bounds(cur)
    return PreprocessSettings.from_dict(cur)


def atomic_write_override(settings: PreprocessSettings) -> Path:
    path = override_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = settings.to_dict()
    payload["version"] = int(settings.version) + 1
    tmp = path.with_suffix(".tmp")
    data = json.dumps(payload, indent=2)
    if len(data.encode("utf-8")) > 64_000:
        raise ValueError("settings payload too large")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)
    get_preprocess_runtime().store.reload(force=True)
    return path


def reset_override() -> bool:
    path = override_path()
    if path.is_file():
        path.unlink()
        get_preprocess_runtime().store.reload(force=True)
        return True
    return False


# --- write protection ------------------------------------------------------------

_WRITE_TIMES: list = []


def _auth_ok(client_host: Optional[str], admin_header: Optional[str]) -> Tuple[bool, str]:
    token = os.environ.get("PREPROCESS_ADMIN_TOKEN", "").strip()
    if token:
        if not admin_header or admin_header.strip() != token:
            return False, "missing or invalid X-Admin-Token"
        return True, ""
    host = (client_host or "").split("%")[0]
    if host not in ("127.0.0.1", "::1", "localhost", "testclient"):
        return False, "writes allowed from loopback only (set PREPROCESS_ADMIN_TOKEN to open)"
    return True, ""


def _writable_probe(client_host: Optional[str], admin_header: Optional[str]) -> bool:
    ok, _ = _auth_ok(client_host, admin_header)
    return ok


def check_write_allowed(client_host: Optional[str], admin_header: Optional[str]) -> Tuple[bool, str]:
    ok, reason = _auth_ok(client_host, admin_header)
    if not ok:
        return False, reason
    now = time.time()
    global _WRITE_TIMES
    _WRITE_TIMES = [t for t in _WRITE_TIMES if now - t < 60.0]
    if len(_WRITE_TIMES) >= 20:
        return False, "rate limit: max 20 writes per minute"
    _WRITE_TIMES.append(now)
    return True, ""
