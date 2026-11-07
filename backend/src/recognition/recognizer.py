"""InsightFace (ArcFace) embedding extraction.

The YOLO box is cropped (with margin) and handed to InsightFace, which re-detects
and aligns the face inside the crop before computing a 512-d ArcFace embedding.
"""
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


class FaceEmbedder:
    def __init__(self, cfg: dict):
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        available = ort.get_available_providers()
        use_gpu = "CUDAExecutionProvider" in available
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name=cfg["model_pack"],
                                allowed_modules=["detection", "recognition"],
                                providers=providers)
        size = int(cfg["det_size"])
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_thresh=0.4, det_size=(size, size))
        log.info("InsightFace %s ready (providers=%s)", cfg["model_pack"], providers)

    def embed(self, crop: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """Return (L2-normalised 512-d embedding, det_score) or None if no face is found."""
        if crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        scale = max(1.0, 160.0 / max(1, min(h, w)))
        if scale > 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        pad = int(0.2 * max(crop.shape[:2]))
        crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        faces = self.app.get(crop)
        if not faces:
            return None
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return best.normed_embedding.astype(np.float32), float(best.det_score)
