"""COCO YOLOv8 person detector — finds full bodies (including backs / faces away)."""
import logging
from typing import Optional

import numpy as np

from .detector import _resolve_device

log = logging.getLogger(__name__)


class YoloPersonDetector:
    """detect(frame) -> float32 (N, 5) xyxy + conf for class 'person' only."""

    def __init__(self, cfg: Optional[dict] = None):
        from ultralytics import YOLO
        cfg = cfg or {}
        weights = cfg.get("weights", "yolov8n.pt")  # auto-downloads COCO nano
        self.model = YOLO(weights)
        self.conf = float(cfg.get("conf_threshold", 0.35))
        self.imgsz = int(cfg.get("imgsz", 640))
        self.device = _resolve_device(cfg.get("device", "auto"))
        self.half = self.device != "cpu"
        self.min_h = int(cfg.get("min_height_px", 60))
        log.info("YOLO person detector ready (weights=%s device=%s)", weights, self.device)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        res = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, device=self.device,
            half=self.half, classes=[0], verbose=False)[0]  # 0 = person
        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy().reshape(-1, 1)
        out = np.hstack([xyxy, conf]).astype(np.float32)
        keep = (out[:, 3] - out[:, 1]) >= self.min_h
        return out[keep] if keep.any() else np.zeros((0, 5), dtype=np.float32)
