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
        from src.core.paths import resolve_under_root
        weights = str(resolve_under_root(cfg.get("weights", "data/models/yolov8n.pt")))  # auto-downloads COCO nano
        self.model = YOLO(weights)
        self.conf = float(cfg.get("conf_threshold", 0.35))
        self.imgsz = int(cfg.get("imgsz", 640))
        self.device = _resolve_device(cfg.get("device", "auto"))
        # FP16 on CUDA when available. `quantize=` is invalid in current Ultralytics.
        self.half = self.device not in ("cpu", "mps")
        self.min_h = int(cfg.get("min_height_px", 60))
        self.min_w = int(cfg.get("min_width_px", 20))
        self.min_aspect = float(cfg.get("min_aspect", 1.15))   # h/w — reject flat blobs
        self.max_aspect = float(cfg.get("max_aspect", 5.5))    # reject ultra-thin floor streaks
        self.nms_iou = float(cfg.get("nms_iou", 0.55))
        log.info("YOLO person detector ready (weights=%s device=%s half=%s)",
                 weights, self.device, self.half)

    @staticmethod
    def _nms(dets: np.ndarray, iou_thr: float) -> np.ndarray:
        """Greedy NMS — crowded hallways otherwise keep overlapping partial bodies."""
        if len(dets) <= 1:
            return dets
        order = np.argsort(-dets[:, 4])
        keep = []
        while len(order):
            i = int(order[0])
            keep.append(i)
            if len(order) == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(dets[i, 0], dets[rest, 0])
            yy1 = np.maximum(dets[i, 1], dets[rest, 1])
            xx2 = np.minimum(dets[i, 2], dets[rest, 2])
            yy2 = np.minimum(dets[i, 3], dets[rest, 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            area_i = max(1e-6, (dets[i, 2] - dets[i, 0]) * (dets[i, 3] - dets[i, 1]))
            area_r = np.maximum(
                1e-6, (dets[rest, 2] - dets[rest, 0]) * (dets[rest, 3] - dets[rest, 1]))
            iou = inter / (area_i + area_r - inter)
            order = rest[iou < iou_thr]
        return dets[np.asarray(keep, dtype=np.int64)]

    def detect(self, frame: np.ndarray) -> np.ndarray:
        res = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, device=self.device,
            half=self.half, classes=[0], verbose=False)[0]  # 0 = person
        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy().reshape(-1, 1)
        out = np.hstack([xyxy, conf]).astype(np.float32)
        h = float(frame.shape[0])
        keep = []
        for row in out:
            x1, y1, x2, y2, c = [float(v) for v in row]
            bh, bw = y2 - y1, x2 - x1
            if bh < self.min_h or bw < self.min_w:
                continue
            aspect = bh / max(bw, 1.0)
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue
            # Polished-floor reflections: tall box sitting entirely in the lower
            # half with no "head" near the upper scene — common false person.
            if y1 > 0.55 * h and aspect > 3.2 and c < self.conf + 0.15:
                continue
            keep.append(row)
        if not keep:
            return np.zeros((0, 5), dtype=np.float32)
        return self._nms(np.asarray(keep, dtype=np.float32), self.nms_iou)
