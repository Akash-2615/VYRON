"""Frame sources: video file (sequential, lossless) and RTSP/live (threaded, latest-frame, auto-reconnect)."""
import logging
import os
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)
LIVE_PREFIXES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://")


class FrameSource:
    def __init__(self, source: str, cfg: dict):
        self.source = source
        self.is_live = source.lower().startswith(LIVE_PREFIXES) or source.isdigit()
        self.cfg = cfg
        self._cap: Optional[cv2.VideoCapture] = None
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._last_seq = 0
        self._thread: Optional[threading.Thread] = None
        self._finished = False
        if source.lower().startswith("rtsp"):
            # TCP is far more reliable than UDP for RTSP (no smeared frames)
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

    def _open(self) -> bool:
        src = int(self.source) if self.source.isdigit() else self.source
        self._cap = cv2.VideoCapture(src)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return True
        return False

    def start(self) -> "FrameSource":
        if not self._open():
            if not self.is_live:
                raise RuntimeError(f"Cannot open video source: {self.source}")
            log.warning("Initial connection to %s failed, will keep retrying", self.source)
        if self.is_live:
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()
        return self

    def _reader(self) -> None:
        retries = 0
        max_retries = int(self.cfg["max_reconnects"])
        delay = float(self.cfg["reconnect_delay_sec"])
        while not self._stop.is_set():
            ok, frame = (False, None)
            if self._cap is not None and self._cap.isOpened():
                ok, frame = self._cap.read()
            if ok:
                retries = 0
                with self._cond:
                    self._frame, self._seq = frame, self._seq + 1
                    self._cond.notify_all()
                continue
            retries += 1
            if max_retries and retries > max_retries:
                log.error("Giving up on stream after %d reconnect attempts", retries - 1)
                break
            log.warning("Stream read failed; reconnecting in %.1fs (attempt %d)", delay, retries)
            if self._cap is not None:
                self._cap.release()
            time.sleep(delay)
            self._open()
        with self._cond:
            self._finished = True
            self._cond.notify_all()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_live:
            return self._cap.read()
        timeout = float(self.cfg["read_timeout_sec"])
        with self._cond:
            end = time.time() + timeout
            while self._seq == self._last_seq and not self._finished:
                remaining = end - time.time()
                if remaining <= 0 or self._stop.is_set():
                    return False, None
                self._cond.wait(timeout=min(remaining, 0.5))
            if self._seq == self._last_seq:
                return False, None
            self._last_seq = self._seq
            return True, self._frame.copy()

    def release(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
