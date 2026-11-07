"""InsightFace RetinaFace-family secondary detector (SCRFD / RetinaFace ONNX).

Used as a YOLO fallback when the primary face YOLO returns nothing, and as a
verifier for low-confidence YOLO boxes. Prefers a classic RetinaFace ONNX if
``model_path`` points at one; otherwise loads buffalo_l ``det_10g.onnx`` (SCRFD),
InsightFace's current RetinaFace-line detector.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_DEFAULT_SCRFD = Path.home() / ".insightface" / "models" / "buffalo_l" / "det_10g.onnx"


class RetinaFaceDetector:
    """detect(frame) -> float32 (N, 5) xyxy + score — same contract as YoloFaceDetector."""

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.conf = float(cfg.get("conf_threshold", 0.5))
        size = int(cfg.get("det_size", 640))
        self.input_size = (size, size)
        model_path = (cfg.get("model_path") or "").strip() or str(_DEFAULT_SCRFD)
        self.model_path = str(model_path)
        self._det = None
        self._load(cfg.get("device", "auto"))

    def _load(self, device: str) -> None:
        import onnxruntime as ort
        from insightface.model_zoo import get_model

        path = Path(self.model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"RetinaFace/SCRFD weights not found at {path}. "
                "Run the tracker once so InsightFace downloads buffalo_l, "
                "or set retina_face.model_path to a RetinaFace/SCRFD .onnx file."
            )
        available = ort.get_available_providers()
        use_gpu = device not in ("cpu",) and "CUDAExecutionProvider" in available
        if device == "cpu":
            use_gpu = False
        self._det = get_model(str(path))
        if self._det is None:
            raise RuntimeError(f"insightface could not load detector from {path}")
        self._det.prepare(ctx_id=0 if use_gpu else -1, input_size=self.input_size)
        log.info(
            "RetinaFace-family detector ready (%s, %s, conf>=%.2f)",
            path.name, "cuda" if use_gpu else "cpu", self.conf,
        )

    def detect(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return np.zeros((0, 5), dtype=np.float32)
        bboxes, _kpss = self._det.detect(frame, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        out = np.asarray(bboxes, dtype=np.float32).reshape(-1, 5)
        return out[out[:, 4] >= self.conf]
