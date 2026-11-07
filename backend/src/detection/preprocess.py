"""Optional video preprocessing for dull / dark footage (detector assist).

Frame filters (denoise / WB / gamma / CLAHE / sharpen) run in an async worker on a
downscaled copy for the face detector only. ``quality_assist`` is NOT a frame filter —
it only contrast-normalises the 112×112 grayscale crop used by the blur gate, cut from
the original frame. Auto-gate measures brightness + contrast (dull / dark) only — not noise.

Embeddings, crops, preview, away detection, and disturbance stats always use the original
frame. All steps are OFF by default — when disabled the pipeline path is a no-op
(byte-identical to pre-feature behaviour).
"""
from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.paths import PROJECT_ROOT, resolve_under_root, runtime_path

log = logging.getLogger(__name__)

OVERRIDE_REL = "data/runtime/preprocess.json"
STATUS_REL = "data/runtime/preprocess_status.json"
PREVIEW_REL = "data/runtime/preprocess_preview.jpg"

STEP_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "auto_gate",
        "kind": "gate",
        "label": "Auto gate (dull / dark)",
        "what": "Measures brightness and contrast on a small grayscale preview; enables frame filters only when the frame looks dull or dark. Does not detect noise.",
        "why": "Avoids wasting work and changing already-good daylight footage.",
        "params": [
            {"key": "brightness_below", "label": "Brightness below", "type": "number", "min": 0, "max": 255, "default": 90, "unit": "mean"},
            {"key": "contrast_below", "label": "Contrast below", "type": "number", "min": 0, "max": 128, "default": 35, "unit": "std"},
            {"key": "mode", "label": "Gate mode", "type": "select", "options": ["any", "all"], "default": "any"},
        ],
    },
    {
        "name": "denoise",
        "kind": "frame",
        "label": "Denoise",
        "what": "Light Gaussian or bilateral blur to calm grain.",
        "why": "Grainy low-light CCTV; runs first so later steps do not amplify noise. Optional — not part of the auto-gate metric.",
        "params": [
            {"key": "method", "label": "Method", "type": "select", "options": ["gaussian", "bilateral"], "default": "gaussian"},
            {"key": "strength", "label": "Strength", "type": "slider", "min": 0, "max": 1, "default": 0.35, "unit": ""},
        ],
    },
    {
        "name": "white_balance",
        "kind": "frame",
        "label": "White balance",
        "what": "Gray-world channel gain correction.",
        "why": "Colour-cast cameras; auto-skips near-grayscale / IR frames.",
        "params": [
            {"key": "strength", "label": "Strength", "type": "slider", "min": 0, "max": 1, "default": 0.6, "unit": ""},
            {"key": "max_gain", "label": "Max gain", "type": "number", "min": 1.0, "max": 2.5, "default": 1.5, "unit": "x"},
        ],
    },
    {
        "name": "gamma",
        "kind": "frame",
        "label": "Gamma",
        "what": "LUT brightness curve (gamma < 1 brightens).",
        "why": "Cheapest fix for under/over-exposed footage.",
        "params": [
            {"key": "gamma", "label": "Gamma", "type": "slider", "min": 0.5, "max": 2.0, "default": 0.75, "unit": ""},
            {"key": "auto_target_mean", "label": "Auto target mean", "type": "number", "min": 0, "max": 255, "default": 0, "unit": "0=off"},
        ],
    },
    {
        "name": "clahe",
        "kind": "frame",
        "label": "CLAHE",
        "what": "Contrast-limited adaptive histogram equalisation on luminance (LAB L-channel).",
        "why": "Dull, hazy, or shadowed faces gain local contrast.",
        "params": [
            {"key": "clip_limit", "label": "Clip limit", "type": "slider", "min": 1, "max": 4, "default": 2.0, "unit": ""},
            {"key": "tile_grid", "label": "Tile grid", "type": "number", "min": 4, "max": 16, "default": 8, "unit": ""},
        ],
    },
    {
        "name": "sharpen",
        "kind": "frame",
        "label": "Sharpen",
        "what": "Mild unsharp mask.",
        "why": "Soft / compressed video; last so it does not amplify earlier noise.",
        "params": [
            {"key": "amount", "label": "Amount", "type": "slider", "min": 0, "max": 1, "default": 0.3, "unit": ""},
            {"key": "sigma", "label": "Sigma", "type": "slider", "min": 0.5, "max": 2.0, "default": 1.0, "unit": "px"},
        ],
    },
    {
        "name": "quality_assist",
        "kind": "blur_gate",
        "label": "Quality-gate assist",
        "what": "Not a frame filter. Contrast-normalises the 112×112 grayscale crop (from the original frame) used by the blur gate only.",
        "why": "Dull-but-real faces are less often rejected as blurry; embeddings still use the original crop.",
        "params": [],
    },
]


def default_preprocess_dict() -> dict:
    return {
        "enabled": False,
        "process_at_detector_resolution": True,
        "mode": "async",
        "max_wait_ms": 50,
        "auto_gate": {
            "enabled": True,
            "brightness_below": 90,
            "contrast_below": 35,
            "mode": "any",
        },
        "denoise": {"enabled": False, "method": "gaussian", "strength": 0.35},
        "white_balance": {"enabled": False, "strength": 0.6, "max_gain": 1.5},
        "gamma": {"enabled": False, "gamma": 0.75, "auto_target_mean": 0},
        "clahe": {"enabled": False, "clip_limit": 2.0, "tile_grid": 8},
        "sharpen": {"enabled": False, "amount": 0.3, "sigma": 1.0},
        "quality_assist": {"enabled": False},
        "version": 0,
    }


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return default


@dataclass(frozen=True)
class PreprocessSettings:
    """Immutable, bounds-checked preprocess configuration."""

    enabled: bool = False
    process_at_detector_resolution: bool = True
    mode: str = "async"  # async | sync
    max_wait_ms: int = 50
    auto_gate_enabled: bool = True
    brightness_below: float = 90.0
    contrast_below: float = 35.0
    gate_mode: str = "any"
    denoise_enabled: bool = False
    denoise_method: str = "gaussian"
    denoise_strength: float = 0.35
    wb_enabled: bool = False
    wb_strength: float = 0.6
    wb_max_gain: float = 1.5
    gamma_enabled: bool = False
    gamma: float = 0.75
    gamma_auto_target: float = 0.0
    clahe_enabled: bool = False
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    sharpen_enabled: bool = False
    sharpen_amount: float = 0.3
    sharpen_sigma: float = 1.0
    quality_assist: bool = False
    version: int = 0

    def any_step_on(self) -> bool:
        return bool(
            self.denoise_enabled or self.wb_enabled or self.gamma_enabled
            or self.clahe_enabled or self.sharpen_enabled
        )

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "process_at_detector_resolution": self.process_at_detector_resolution,
            "mode": self.mode,
            "max_wait_ms": self.max_wait_ms,
            "auto_gate": {
                "enabled": self.auto_gate_enabled,
                "brightness_below": self.brightness_below,
                "contrast_below": self.contrast_below,
                "mode": self.gate_mode,
            },
            "denoise": {
                "enabled": self.denoise_enabled,
                "method": self.denoise_method,
                "strength": self.denoise_strength,
            },
            "white_balance": {
                "enabled": self.wb_enabled,
                "strength": self.wb_strength,
                "max_gain": self.wb_max_gain,
            },
            "gamma": {
                "enabled": self.gamma_enabled,
                "gamma": self.gamma,
                "auto_target_mean": self.gamma_auto_target,
            },
            "clahe": {
                "enabled": self.clahe_enabled,
                "clip_limit": self.clahe_clip,
                "tile_grid": self.clahe_tile,
            },
            "sharpen": {
                "enabled": self.sharpen_enabled,
                "amount": self.sharpen_amount,
                "sigma": self.sharpen_sigma,
            },
            "quality_assist": {"enabled": self.quality_assist},
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, raw: Optional[dict], *, version: Optional[int] = None) -> "PreprocessSettings":
        d = default_preprocess_dict()
        if raw:
            # shallow + nested merge
            for k, v in raw.items():
                if k in d and isinstance(d[k], dict) and isinstance(v, dict):
                    d[k] = {**d[k], **v}
                else:
                    d[k] = v
        ag = d.get("auto_gate") or {}
        dn = d.get("denoise") or {}
        wb = d.get("white_balance") or {}
        gm = d.get("gamma") or {}
        cl = d.get("clahe") or {}
        sh = d.get("sharpen") or {}
        qa = d.get("quality_assist") or {}
        mode = str(d.get("mode") or "async").lower()
        if mode not in ("async", "sync"):
            mode = "async"
        gate_mode = str(ag.get("mode") or "any").lower()
        if gate_mode not in ("any", "all"):
            gate_mode = "any"
        method = str(dn.get("method") or "gaussian").lower()
        if method not in ("gaussian", "bilateral"):
            method = "gaussian"
        ver = int(version if version is not None else d.get("version") or 0)
        return cls(
            enabled=_as_bool(d.get("enabled"), False),
            process_at_detector_resolution=_as_bool(d.get("process_at_detector_resolution"), True),
            mode=mode,
            max_wait_ms=int(_clamp(float(d.get("max_wait_ms", 50)), 0, 500)),
            auto_gate_enabled=_as_bool(ag.get("enabled"), True),
            brightness_below=_clamp(float(ag.get("brightness_below", 90)), 0, 255),
            contrast_below=_clamp(float(ag.get("contrast_below", 35)), 0, 128),
            gate_mode=gate_mode,
            denoise_enabled=_as_bool(dn.get("enabled"), False),
            denoise_method=method,
            denoise_strength=_clamp(float(dn.get("strength", 0.35)), 0, 1),
            wb_enabled=_as_bool(wb.get("enabled"), False),
            wb_strength=_clamp(float(wb.get("strength", 0.6)), 0, 1),
            wb_max_gain=_clamp(float(wb.get("max_gain", 1.5)), 1.0, 2.5),
            gamma_enabled=_as_bool(gm.get("enabled"), False),
            gamma=_clamp(float(gm.get("gamma", 0.75)), 0.5, 2.0),
            gamma_auto_target=_clamp(float(gm.get("auto_target_mean", 0)), 0, 255),
            clahe_enabled=_as_bool(cl.get("enabled"), False),
            clahe_clip=_clamp(float(cl.get("clip_limit", 2.0)), 1.0, 4.0),
            clahe_tile=int(_clamp(float(cl.get("tile_grid", 8)), 4, 16)),
            sharpen_enabled=_as_bool(sh.get("enabled"), False),
            sharpen_amount=_clamp(float(sh.get("amount", 0.3)), 0, 1),
            sharpen_sigma=_clamp(float(sh.get("sigma", 1.0)), 0.5, 2.0),
            quality_assist=_as_bool(qa.get("enabled"), False),
            version=ver,
        )


@dataclass
class FramePacket:
    frame: np.ndarray
    enhanced: Optional[np.ndarray] = None
    scale: float = 1.0  # enhanced_width / original_width (1.0 = same size)
    seq: int = 0
    ts: float = 0.0
    gate_active: bool = False
    fallback: bool = False


# ---- pure step functions (unit-testable) ------------------------------------

def step_denoise(bgr: np.ndarray, method: str, strength: float) -> np.ndarray:
    if strength <= 0:
        return bgr
    if method == "bilateral":
        # d=5 fixed (fast); sigma scaled by strength
        s = max(1.0, 25.0 * strength)
        return cv2.bilateralFilter(bgr, d=5, sigmaColor=s, sigmaSpace=s)
    # gaussian 3x3 or 5x5
    k = 3 if strength < 0.55 else 5
    sigma = 0.3 + 1.2 * strength
    return cv2.GaussianBlur(bgr, (k, k), sigmaX=sigma)


def step_white_balance(bgr: np.ndarray, strength: float, max_gain: float) -> np.ndarray:
    if strength <= 0:
        return bgr
    # Skip near-grayscale / IR (low chroma)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if float(hsv[:, :, 1].mean()) < 12.0:
        return bgr
    f = bgr.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)
    gray = float(means.mean()) + 1e-6
    gains = np.clip(gray / (means + 1e-6), 1.0 / max_gain, max_gain)
    gains = 1.0 + (gains - 1.0) * strength
    out = f * gains.reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def _gamma_lut(gamma: float) -> np.ndarray:
    # gamma < 1 brightens; gamma > 1 darkens (power-law on normalised pixels).
    g = max(float(gamma), 1e-6)
    table = np.array([((i / 255.0) ** g) * 255.0 for i in range(256)], dtype=np.uint8)
    return table


def step_gamma(bgr: np.ndarray, gamma: float, auto_target_mean: float = 0.0) -> np.ndarray:
    g = float(gamma)
    if auto_target_mean and auto_target_mean > 0:
        mean = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean()) + 1e-6
        # nudge gamma so mean moves toward target (cheap heuristic)
        ratio = float(auto_target_mean) / mean
        # brighter needed → smaller gamma
        g = _clamp(g / max(ratio, 1e-3), 0.5, 2.0)
    return cv2.LUT(bgr, _gamma_lut(g))


def step_clahe(bgr: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def step_sharpen(bgr: np.ndarray, amount: float, sigma: float) -> np.ndarray:
    if amount <= 0:
        return bgr
    blur = cv2.GaussianBlur(bgr, (0, 0), sigmaX=float(sigma))
    return cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)


def quality_assist_gray(gray_112: np.ndarray) -> np.ndarray:
    """Contrast-normalise a 112x112 grayscale crop for the blur gate only."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(gray_112)


def resize_for_detector(bgr: np.ndarray, imgsz: int) -> Tuple[np.ndarray, float]:
    """Resize so long side == imgsz. Returns (resized, scale=new_w/orig_w)."""
    h, w = bgr.shape[:2]
    long = max(h, w)
    if long <= 0 or imgsz <= 0 or long == imgsz:
        return bgr, 1.0
    scale = float(imgsz) / float(long)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    out = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return out, scale


def map_boxes_to_original(dets: Optional[np.ndarray], scale: float) -> Optional[np.ndarray]:
    """Map detector boxes from enhanced/downscaled space back to original pixels."""
    if dets is None or len(dets) == 0:
        return dets
    if scale == 1.0 or scale <= 0:
        return dets
    out = np.asarray(dets, dtype=np.float32).copy()
    inv = 1.0 / float(scale)
    out[:, 0:4] *= inv
    return out


def frame_metrics(bgr: np.ndarray) -> Tuple[float, float]:
    """mean luminance, contrast (std) on a downscaled gray copy."""
    small = cv2.resize(bgr, (160, 90), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(gray.mean()), float(gray.std())


class AutoGate:
    """EMA + hysteresis so the dull-gate does not flicker."""

    def __init__(self):
        self._ema_bri: Optional[float] = None
        self._ema_con: Optional[float] = None
        self._active = False
        self._hold = 0

    def reset(self) -> None:
        self._ema_bri = self._ema_con = None
        self._active = False
        self._hold = 0

    def update(self, bri: float, con: float, settings: PreprocessSettings) -> bool:
        a = 0.25
        self._ema_bri = bri if self._ema_bri is None else (1 - a) * self._ema_bri + a * bri
        self._ema_con = con if self._ema_con is None else (1 - a) * self._ema_con + a * con
        b_thr, c_thr = settings.brightness_below, settings.contrast_below
        enter_b, exit_b = b_thr, b_thr * 1.10
        enter_c, exit_c = c_thr, c_thr * 1.10
        dull_b = self._ema_bri < (enter_b if not self._active else exit_b)
        dull_c = self._ema_con < (enter_c if not self._active else exit_c)
        want = (dull_b or dull_c) if settings.gate_mode == "any" else (dull_b and dull_c)
        if want == self._active:
            self._hold = 0
            return self._active
        self._hold += 1
        if self._hold >= 15:
            self._active = want
            self._hold = 0
        return self._active


class Preprocessor:
    """Applies the fixed step order; records per-step timings."""

    def __init__(self):
        self.gate = AutoGate()
        self.last_step_ms: Dict[str, float] = {}
        self.last_total_ms: float = 0.0
        self.last_gate_active: bool = False

    def enhance(self, bgr: np.ndarray, settings: PreprocessSettings,
                *, detector_imgsz: int = 640) -> Tuple[np.ndarray, float, bool]:
        """Return (enhanced_bgr, scale, gate_active). Never mutates ``bgr``."""
        t0 = time.perf_counter()
        self.last_step_ms = {}
        if not settings.enabled or not settings.any_step_on():
            self.last_gate_active = False
            self.last_total_ms = (time.perf_counter() - t0) * 1000.0
            return bgr, 1.0, False

        work = bgr
        scale = 1.0
        if settings.process_at_detector_resolution:
            work, scale = resize_for_detector(bgr, int(detector_imgsz))

        bri, con = frame_metrics(work)
        gate_on = True
        if settings.auto_gate_enabled:
            gate_on = self.gate.update(bri, con, settings)
        self.last_gate_active = gate_on
        if not gate_on:
            self.last_total_ms = (time.perf_counter() - t0) * 1000.0
            return work if scale != 1.0 else bgr, scale, False

        out = work
        def _run(name, fn):
            nonlocal out
            s = time.perf_counter()
            out = fn(out)
            self.last_step_ms[name] = (time.perf_counter() - s) * 1000.0

        if settings.denoise_enabled:
            _run("denoise", lambda x: step_denoise(x, settings.denoise_method, settings.denoise_strength))
        if settings.wb_enabled:
            _run("white_balance", lambda x: step_white_balance(x, settings.wb_strength, settings.wb_max_gain))
        if settings.gamma_enabled:
            _run("gamma", lambda x: step_gamma(x, settings.gamma, settings.gamma_auto_target))
        if settings.clahe_enabled:
            _run("clahe", lambda x: step_clahe(x, settings.clahe_clip, settings.clahe_tile))
        if settings.sharpen_enabled:
            _run("sharpen", lambda x: step_sharpen(x, settings.sharpen_amount, settings.sharpen_sigma))

        self.last_total_ms = (time.perf_counter() - t0) * 1000.0
        return out, scale, True


class SettingsStore:
    """config.json defaults < runtime override file; hot-reload by mtime."""

    def __init__(self, base: Optional[dict] = None):
        self._base = PreprocessSettings.from_dict(base or default_preprocess_dict())
        self._settings = self._base
        self._mtime: float = -1.0
        self._lock = threading.Lock()
        self._version = 0
        self.reload(force=True)

    @property
    def override_path(self) -> Path:
        return runtime_path("preprocess.json")

    def get(self) -> PreprocessSettings:
        with self._lock:
            return self._settings

    def reload(self, force: bool = False) -> bool:
        path = self.override_path
        try:
            mtime = path.stat().st_mtime if path.is_file() else -1.0
        except OSError:
            mtime = -1.0
        if not force and mtime == self._mtime:
            return False
        self._mtime = mtime
        if mtime < 0:
            with self._lock:
                old = self._settings
                self._settings = self._base
            diff = _settings_diff(old, self._base)
            if diff:
                log.info("PREPROCESS_CONFIG_CHANGED version=%s diff=%s",
                         self._base.version, diff)
            return True
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("preprocess override must be a JSON object")
            new = PreprocessSettings.from_dict(raw, version=max(self._version, int(raw.get("version") or 0)))
        except Exception as exc:  # noqa: BLE001
            # Keep last-good settings; record mtime so we only warn again on change.
            log.warning("PREPROCESS_CONFIG_INVALID path=%s err=%s", path, exc)
            return False
        with self._lock:
            old = self._settings
            self._settings = new
            self._version = max(self._version, int(new.version))
        diff = _settings_diff(old, new)
        if diff:
            log.info("PREPROCESS_CONFIG_CHANGED version=%s diff=%s", new.version, diff)
        return True

    def set_base_from_config(self, cfg_block: Optional[dict]) -> None:
        new_base = PreprocessSettings.from_dict(cfg_block or default_preprocess_dict())
        # Avoid re-applying identical config.json defaults on every API poll.
        if new_base.to_dict() != self._base.to_dict():
            self._base = new_base
        self.reload(force=False)


def _settings_diff(a: PreprocessSettings, b: PreprocessSettings) -> str:
    da, db = a.to_dict(), b.to_dict()
    changed = []

    def walk(x, y, prefix=""):
        if isinstance(x, dict) and isinstance(y, dict):
            keys = set(x) | set(y)
            for k in sorted(keys):
                walk(x.get(k), y.get(k), f"{prefix}{k}.")
        elif x != y:
            changed.append(prefix.rstrip("."))

    walk(da, db)
    return ",".join(changed[:12])


class PreprocessRuntime:
    """Shared runtime: settings, metrics, preview, optional async worker."""

    def __init__(self):
        self.store = SettingsStore()
        self.preprocessor = Preprocessor()
        self._metrics_lock = threading.Lock()
        self.metrics: Dict[str, Any] = {
            "version": 0,
            "enabled": False,
            "step_ms": {},
            "total_avg_ms": 0.0,
            "total_p95_ms": 0.0,
            "ready_pct": 100.0,
            "fallback": 0,
            "dropped": 0,
            "gate_active": False,
            "frames_enhanced": 0,
            "updated_at": None,
        }
        self._totals: List[float] = []
        self._ready = 0
        self._seen = 0
        self._last_status_write = 0.0
        self._last_preview_write = 0.0
        self._last_backlog_log = 0.0
        self._pending: Optional[Tuple[int, np.ndarray, float, PreprocessSettings]] = None
        self._result: Dict[int, Tuple[np.ndarray, float, bool]] = {}
        self._result_lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._wake = threading.Event()

    def start_worker(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._worker_loop, name="preprocess", daemon=True)
        self._worker.start()

    def stop_worker(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._worker:
            self._worker.join(timeout=2.0)
            self._worker = None

    def _worker_loop(self) -> None:
        last_reload = 0.0
        while not self._stop.is_set():
            now = time.time()
            if now - last_reload >= 1.0:
                self.store.reload()
                last_reload = now
            self._wake.wait(timeout=0.05)
            self._wake.clear()
            job = None
            with self._result_lock:
                if self._pending is not None:
                    job = self._pending
                    self._pending = None
            if job is None:
                continue
            seq, frame, imgsz, settings = job
            try:
                enh, scale, gate = self.preprocessor.enhance(
                    frame, settings, detector_imgsz=int(imgsz))
            except Exception:  # noqa: BLE001
                log.exception("preprocess enhance failed")
                continue
            with self._result_lock:
                self._result = {seq: (enh, scale, gate)}
            self._note_timing(settings)
            self._maybe_write_status(settings)
            self._maybe_write_preview(frame, enh if gate or settings.enabled else frame, settings)

    def submit(self, seq: int, frame: np.ndarray, settings: PreprocessSettings,
               detector_imgsz: int) -> None:
        with self._result_lock:
            if self._pending is not None:
                self.metrics["dropped"] = int(self.metrics.get("dropped", 0)) + 1
                now = time.time()
                if now - self._last_backlog_log >= 10.0:
                    log.info("PREPROCESS_BACKLOG dropped=%s", self.metrics["dropped"])
                    self._last_backlog_log = now
            self._pending = (seq, frame, float(detector_imgsz), settings)
        self._wake.set()

    def take(self, seq: int, wait_ms: float = 0.0) -> Optional[Tuple[np.ndarray, float, bool]]:
        # Live (wait_ms≈0): never block; may accept a drifted latest result.
        # File (wait_ms>0): wait only for this seq, up to max_wait_ms — never steal
        # an older seq's result (that would inflate wait past the budget).
        deadline = time.time() + max(0.0, wait_ms) / 1000.0
        allow_drift = wait_ms <= 0.0
        while True:
            with self._result_lock:
                hit = self._result.pop(seq, None)
                if hit is not None:
                    return hit
                if allow_drift and self._result:
                    _, val = self._result.popitem()
                    self._result.clear()
                    return val
            if time.time() >= deadline:
                with self._result_lock:
                    hit = self._result.pop(seq, None)
                    if hit is not None:
                        return hit
                    if allow_drift and self._result:
                        _, val = self._result.popitem()
                        self._result.clear()
                        return val
                return None
            time.sleep(0.001)

    def enhance_sync(self, frame: np.ndarray, settings: PreprocessSettings,
                     detector_imgsz: int) -> Tuple[np.ndarray, float, bool]:
        out = self.preprocessor.enhance(frame, settings, detector_imgsz=detector_imgsz)
        self._note_timing(settings)
        self._maybe_write_status(settings)
        self._maybe_write_preview(frame, out[0], settings)
        return out

    def _note_timing(self, settings: PreprocessSettings) -> None:
        ms = float(self.preprocessor.last_total_ms)
        with self._metrics_lock:
            self._totals.append(ms)
            if len(self._totals) > 120:
                self._totals = self._totals[-120:]
            self._seen += 1
            self._ready += 1
            self.metrics["step_ms"] = dict(self.preprocessor.last_step_ms)
            self.metrics["total_avg_ms"] = float(sum(self._totals) / max(len(self._totals), 1))
            ordered = sorted(self._totals)
            self.metrics["total_p95_ms"] = float(ordered[int(0.95 * (len(ordered) - 1))]) if ordered else 0.0
            self.metrics["ready_pct"] = 100.0 * self._ready / max(self._seen, 1)
            self.metrics["gate_active"] = bool(self.preprocessor.last_gate_active)
            self.metrics["frames_enhanced"] = int(self.metrics.get("frames_enhanced", 0)) + (
                1 if self.preprocessor.last_gate_active else 0)
            self.metrics["enabled"] = bool(settings.enabled)
            self.metrics["version"] = int(settings.version)
            self.metrics["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    def note_fallback(self) -> None:
        with self._metrics_lock:
            self.metrics["fallback"] = int(self.metrics.get("fallback", 0)) + 1
            self._seen += 1
            self.metrics["ready_pct"] = 100.0 * self._ready / max(self._seen, 1)

    def _maybe_write_status(self, settings: PreprocessSettings) -> None:
        now = time.time()
        if now - self._last_status_write < 2.0:
            return
        self._last_status_write = now
        path = runtime_path("preprocess_status.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.metrics)
        payload["settings_enabled"] = settings.enabled
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            log.debug("preprocess status write failed", exc_info=True)

    def _maybe_write_preview(self, original: np.ndarray, enhanced: np.ndarray,
                             settings: PreprocessSettings) -> None:
        if not settings.enabled:
            return
        now = time.time()
        if now - self._last_preview_write < 1.5:
            return
        self._last_preview_write = now
        try:
            def thumb(img):
                h, w = img.shape[:2]
                # Large enough for Config + Live compare panes.
                scale = 900.0 / max(w, 1)
                return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                                 interpolation=cv2.INTER_AREA)

            def label(img, text: str):
                out = img.copy()
                cv2.rectangle(out, (0, 0), (out.shape[1], 36), (18, 14, 28), -1)
                cv2.putText(out, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (220, 210, 255), 2, cv2.LINE_AA)
                return out

            a, b = thumb(original), thumb(enhanced)
            if a.shape[0] != b.shape[0]:
                h = min(a.shape[0], b.shape[0])
                a, b = a[:h], b[:h]
            if a.shape[1] != b.shape[1]:
                w = max(a.shape[1], b.shape[1])

                def pad(x):
                    if x.shape[1] == w:
                        return x
                    out = np.zeros((x.shape[0], w, 3), dtype=np.uint8)
                    out[:, :x.shape[1]] = x
                    return out

                a, b = pad(a), pad(b)
            a = label(a, "ORIGINAL  (identity · live · crops)")
            b = label(b, "ENHANCED  (detector input when ready)")
            gap = np.full((a.shape[0], 4, 3), 40, dtype=np.uint8)
            panel = np.hstack([a, gap, b])
            path = runtime_path("preprocess_preview.jpg")
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        except Exception:  # noqa: BLE001
            log.debug("preprocess preview write failed", exc_info=True)


_RUNTIME: Optional[PreprocessRuntime] = None
_RUNTIME_LOCK = threading.Lock()


def get_preprocess_runtime() -> PreprocessRuntime:
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = PreprocessRuntime()
        return _RUNTIME


PRESETS: Dict[str, dict] = {
    "off": {"enabled": False},
    "dull_low_contrast": {
        "enabled": True,
        "auto_gate": {"enabled": True},
        "clahe": {"enabled": True, "clip_limit": 2.0},
        "denoise": {"enabled": False},
        "gamma": {"enabled": False},
        "white_balance": {"enabled": False},
        "sharpen": {"enabled": False},
    },
    "low_light": {
        "enabled": True,
        "auto_gate": {"enabled": True},
        "gamma": {"enabled": True, "gamma": 0.7},
        "clahe": {"enabled": True, "clip_limit": 2.5},
        "denoise": {"enabled": True, "method": "gaussian", "strength": 0.35},
        "white_balance": {"enabled": False},
        "sharpen": {"enabled": False},
    },
    "noisy_cctv": {
        "enabled": True,
        "auto_gate": {"enabled": True},
        "denoise": {"enabled": True, "method": "bilateral", "strength": 0.45},
        "clahe": {"enabled": True, "clip_limit": 2.0},
        "sharpen": {"enabled": True, "amount": 0.3},
        "gamma": {"enabled": False},
        "white_balance": {"enabled": False},
    },
}
