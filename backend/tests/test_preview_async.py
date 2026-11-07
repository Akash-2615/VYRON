"""Non-blocking live preview publisher."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src.preview import PreviewPublisher, publish_preview


def test_publish_preview_sync(tmp_path):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:] = (20, 80, 180)
    path = str(tmp_path / "live.jpg")
    publish_preview(frame, path, quality=60, max_width=32)
    assert Path(path).is_file()
    assert Path(path).stat().st_size > 0


def test_preview_publisher_writes_latest(tmp_path):
    pub = PreviewPublisher()
    path = str(tmp_path / "live.jpg")
    try:
        for i in range(5):
            frame = np.full((40, 60, 3), i * 40, dtype=np.uint8)
            pub.publish(frame, path, quality=50, max_width=0)
        pub.flush(timeout=2.0)
        assert Path(path).is_file()
        assert Path(path).stat().st_size > 0
    finally:
        pub.close()
