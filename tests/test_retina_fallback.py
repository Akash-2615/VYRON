"""YOLO + RetinaFace merge / verify logic (no heavy models)."""
import numpy as np

from src.detector import HybridFaceDetector, merge_yolo_retina


def test_fallback_when_yolo_empty():
    retina = np.array([[10, 10, 50, 50, 0.9]], dtype=np.float32)
    out, meta = merge_yolo_retina(np.zeros((0, 5), dtype=np.float32), retina)
    assert len(out) == 1 and meta["source"] == "retina_fallback" and meta["fallback"] == 1


def test_high_conf_yolo_kept_without_retina():
    yolo = np.array([[10, 10, 50, 50, 0.8]], dtype=np.float32)
    out, meta = merge_yolo_retina(yolo, np.zeros((0, 5), dtype=np.float32), verify_below_conf=0.55)
    assert len(out) == 1 and meta["source"] == "yolo"


def test_low_conf_yolo_verified_by_retina():
    yolo = np.array([[10, 10, 50, 50, 0.40]], dtype=np.float32)
    retina = np.array([[12, 12, 48, 48, 0.92]], dtype=np.float32)
    out, meta = merge_yolo_retina(yolo, retina, verify_below_conf=0.55, verify_iou=0.3)
    assert len(out) == 1
    assert meta["source"] == "yolo+retina_verify"
    assert meta["verified"] == 1
    # Retina geometry preferred
    assert abs(float(out[0, 4]) - 0.92) < 1e-5


def test_low_conf_yolo_dropped_without_retina_match():
    yolo = np.array([[10, 10, 50, 50, 0.40]], dtype=np.float32)
    retina = np.array([[200, 200, 240, 240, 0.9]], dtype=np.float32)
    out, meta = merge_yolo_retina(yolo, retina, verify_below_conf=0.55, verify_iou=0.3)
    assert len(out) == 0 and meta["dropped_low"] == 1


def test_hybrid_uses_injected_retina():
    class FakeYolo:
        def detect(self, frame):
            return np.zeros((0, 5), dtype=np.float32)

    class FakeRetina:
        def detect(self, frame):
            return np.array([[1, 1, 20, 20, 0.88]], dtype=np.float32)

    # Bypass real YOLO ctor
    h = HybridFaceDetector.__new__(HybridFaceDetector)
    h.yolo = FakeYolo()
    h.retina = FakeRetina()
    h.enabled = True
    h.verify_below = 0.55
    h.verify_iou = 0.3
    h.fallback_on_empty = True
    h.last_meta = {}
    out = h.detect(np.zeros((32, 32, 3), dtype=np.uint8))
    assert len(out) == 1 and h.last_meta["source"] == "retina_fallback"
