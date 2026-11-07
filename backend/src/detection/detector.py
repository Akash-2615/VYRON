"""YOLOv8-face detector (Ultralytics) + optional RetinaFace-family hybrid."""
import logging
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


def _resolve_device(device: str):
    if device != "auto":
        return device
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def ensure_weights(path: str, url: str) -> str:
    from src.core.paths import resolve_under_root
    p = resolve_under_root(path)
    if p.exists():
        return str(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading YOLO face weights -> %s", p)
    try:
        urllib.request.urlretrieve(url, str(p))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not download weights from {url}: {exc}. "
            f"Download a YOLOv8 face model manually and put it at {p}") from exc
    return str(p)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    return inter / max(area_a + area_b - inter, 1e-9)


def merge_yolo_retina(
    yolo: np.ndarray,
    retina: np.ndarray,
    *,
    verify_below_conf: float = 0.55,
    verify_iou: float = 0.3,
) -> Tuple[np.ndarray, dict]:
    """Combine YOLO dets with RetinaFace fallback / low-conf verification.

    * Empty YOLO → use RetinaFace full-frame (fallback).
    * High-conf YOLO boxes kept as-is.
    * Low-conf YOLO boxes kept only if a RetinaFace box overlaps (IoU); Retina box wins.
    """
    empty = np.zeros((0, 5), dtype=np.float32)
    yolo = np.zeros((0, 5), dtype=np.float32) if yolo is None else np.asarray(yolo, dtype=np.float32).reshape(-1, 5)
    retina = empty if retina is None else np.asarray(retina, dtype=np.float32).reshape(-1, 5)
    meta = {"source": "yolo", "fallback": 0, "verified": 0, "dropped_low": 0}

    if len(yolo) == 0:
        if len(retina):
            meta["source"] = "retina_fallback"
            meta["fallback"] = len(retina)
            return retina.copy(), meta
        return empty, meta

    high = yolo[yolo[:, 4] >= verify_below_conf]
    low = yolo[yolo[:, 4] < verify_below_conf]
    if len(low) == 0:
        return yolo.copy(), meta

    verified = []
    for box in low:
        best_iou, best = 0.0, None
        for r in retina:
            iou = _iou_xyxy(box, r)
            if iou > best_iou:
                best_iou, best = iou, r
        if best is not None and best_iou >= verify_iou:
            verified.append(best)
            meta["verified"] += 1
        else:
            meta["dropped_low"] += 1

    parts = [high]
    if verified:
        parts.append(np.asarray(verified, dtype=np.float32).reshape(-1, 5))
        meta["source"] = "yolo+retina_verify"
    out = np.vstack(parts) if any(len(p) for p in parts) else empty
    return out.astype(np.float32), meta


class YoloFaceDetector:
    """detect(frame) -> float32 array (N, 5): x1, y1, x2, y2, confidence."""

    def __init__(self, cfg: dict):
        from ultralytics import YOLO
        weights = ensure_weights(cfg["weights"], cfg["weights_url"])
        self.model = YOLO(weights)
        self.conf = float(cfg["conf_threshold"])
        self.imgsz = int(cfg["imgsz"])
        self.device = _resolve_device(cfg["device"])
        self.half = self.device != "cpu"
        log.info("YOLO face detector ready (device=%s)", self.device)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                                 device=self.device, half=self.half, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy().reshape(-1, 1)
        return np.hstack([xyxy, conf]).astype(np.float32)


class HybridFaceDetector:
    """YOLO primary + RetinaFace-family fallback / low-confidence verification."""

    def __init__(self, yolo_cfg: dict, retina_cfg: Optional[dict] = None, retina=None):
        self.yolo = YoloFaceDetector(yolo_cfg)
        retina_cfg = retina_cfg or {}
        self.enabled = bool(retina_cfg.get("enabled", True))
        self.verify_below = float(retina_cfg.get("verify_below_conf", 0.55))
        self.verify_iou = float(retina_cfg.get("verify_iou", 0.30))
        self.fallback_on_empty = bool(retina_cfg.get("fallback_on_empty", True))
        self.retina = retina
        self.last_meta: dict = {"source": "yolo", "fallback": 0, "verified": 0, "dropped_low": 0}
        if self.enabled and self.retina is None:
            from .retina_face import RetinaFaceDetector
            self.retina = RetinaFaceDetector(retina_cfg)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        yolo = self.yolo.detect(frame)
        if not self.enabled or self.retina is None:
            self.last_meta = {"source": "yolo", "fallback": 0, "verified": 0, "dropped_low": 0}
            return yolo

        need_retina = (self.fallback_on_empty and len(yolo) == 0) or (
            len(yolo) > 0 and np.any(yolo[:, 4] < self.verify_below)
        )
        retina = self.retina.detect(frame) if need_retina else np.zeros((0, 5), dtype=np.float32)
        merged, meta = merge_yolo_retina(
            yolo, retina,
            verify_below_conf=self.verify_below,
            verify_iou=self.verify_iou,
        )
        self.last_meta = meta
        return merged
