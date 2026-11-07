#!/usr/bin/env python3
"""Benchmark preprocess steps (report measured times; do not claim targets)."""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from src.detection.preprocess import (  # noqa: E402
    PreprocessSettings,
    Preprocessor,
    resize_for_detector,
    step_clahe,
    step_denoise,
    step_gamma,
    step_sharpen,
    step_white_balance,
)


def _bench(fn, n=20):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.mean(times), sorted(times)[int(0.95 * (len(times) - 1))]


def main():
    full = np.zeros((1080, 1920, 3), dtype=np.uint8)
    full[:] = (45, 50, 60)
    small, _ = resize_for_detector(full, 640)
    print("size_full", full.shape, "size_det", small.shape)
    for name, fn in [
        ("denoise", lambda: step_denoise(small, "gaussian", 0.35)),
        ("wb", lambda: step_white_balance(small, 0.6, 1.5)),
        ("gamma", lambda: step_gamma(small, 0.75)),
        ("clahe", lambda: step_clahe(small, 2.0, 8)),
        ("sharpen", lambda: step_sharpen(small, 0.3, 1.0)),
    ]:
        avg, p95 = _bench(fn)
        print(f"{name:10} avg={avg:.2f}ms p95={p95:.2f}ms @ detector_res")
    settings = PreprocessSettings(
        enabled=True, auto_gate_enabled=False, denoise_enabled=True,
        gamma_enabled=True, clahe_enabled=True, sharpen_enabled=True,
        process_at_detector_resolution=True,
    )
    pp = Preprocessor()
    avg, p95 = _bench(lambda: pp.enhance(full, settings, detector_imgsz=640))
    print(f"{'pipeline':10} avg={avg:.2f}ms p95={p95:.2f}ms (all steps, det res)")
    print("Note: main-loop added time in async mode is ~0 when enhanced is ready; measure with a live run.")


if __name__ == "__main__":
    main()
