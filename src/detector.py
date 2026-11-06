"""YOLOv8-face detector (Ultralytics)."""
import logging
import urllib.request
from pathlib import Path

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
    p = Path(path)
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
