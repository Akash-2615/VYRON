"""Frame sources: video file (sequential), webcam (sync live), RTSP (threaded latest-frame)."""
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
    def __init__(self, source: str, cfg: dict, stop: Optional[dict] = None):
        self.source = source
        # Webcam is live for pacing/gap logic, but must NOT use a reader thread:
        # macOS AVFoundation segfaults if VideoCapture.release() runs while another
        # thread is inside cap.read(). Sync webcam reads keep open/read/release
        # on the tracker thread so dashboard Stop is safe.
        self.is_live = source.lower().startswith(LIVE_PREFIXES) or source.isdigit()
        self._threaded = source.lower().startswith(LIVE_PREFIXES)
        self.cfg = cfg
        # Shared with tracker_job / main run loop — Stop must unblock RTSP read.
        self._external_stop = stop
        self._cap: Optional[cv2.VideoCapture] = None
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._last_seq = 0
        self._thread: Optional[threading.Thread] = None
        self._finished = False
        self._cap_lock = threading.Lock()
        if source.lower().startswith("rtsp"):
            # TCP is far more reliable than UDP for RTSP (no smeared frames)
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

    def _stopping(self) -> bool:
        if self._stop.is_set():
            return True
        ext = self._external_stop
        return bool(ext is not None and ext.get("flag"))

    def _open(self) -> bool:
        if self.source.isdigit():
            src = int(self.source)
        elif self.is_live:
            src = self.source
        else:
            from src.core.paths import resolve_under_root
            src = str(resolve_under_root(self.source))
        with self._cap_lock:
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
        if self._threaded:
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()
        return self

    def interrupt(self) -> None:
        """Ask the reader / sync loop to exit. Never release VideoCapture here.

        Releasing AVFoundation/webcam (and often ffmpeg RTSP) from another thread
        while ``cap.read()`` is in progress segfaults the whole API process on macOS.
        """
        self._stop.set()
        with self._cond:
            self._finished = True
            self._cond.notify_all()

    def _release_cap(self) -> None:
        with self._cap_lock:
            cap = self._cap
            self._cap = None
        if cap is None:
            return
        try:
            cap.release()
        except Exception:  # noqa: BLE001
            pass

    def _reader(self) -> None:
        retries = 0
        max_retries = int(self.cfg["max_reconnects"])
        delay = float(self.cfg["reconnect_delay_sec"])
        self.last_gap_sec: float = 0.0
        self._gap_started: Optional[float] = None
        try:
            while not self._stopping():
                ok, frame = (False, None)
                with self._cap_lock:
                    cap = self._cap
                if cap is not None:
                    try:
                        if cap.isOpened():
                            ok, frame = cap.read()
                    except Exception:  # noqa: BLE001
                        ok, frame = False, None
                if self._stopping():
                    break
                if ok:
                    if self._gap_started is not None:
                        self.last_gap_sec = time.time() - self._gap_started
                        self._gap_started = None
                    retries = 0
                    with self._cond:
                        self._frame, self._seq = frame, self._seq + 1
                        self._cond.notify_all()
                    continue
                if self._gap_started is None:
                    self._gap_started = time.time()
                retries += 1
                if max_retries and retries > max_retries:
                    log.error("Giving up on stream after %d reconnect attempts", retries - 1)
                    break
                log.warning("Stream read failed; reconnecting in %.1fs (attempt %d)", delay, retries)
                self._release_cap()
                # Interruptible sleep — Stop during reconnect delay must not wait full delay.
                if self._stop.wait(timeout=delay) or self._stopping():
                    break
                if self._stopping():
                    break
                self._open()
        finally:
            # Capture is owned by this thread — release only here / after join.
            self._release_cap()
            with self._cond:
                self._finished = True
                self._cond.notify_all()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        # File + webcam: same-thread open/read/release (Stop-safe on macOS).
        if not self._threaded:
            if self._stopping():
                return False, None
            with self._cap_lock:
                cap = self._cap
            if cap is None:
                return False, None
            try:
                ok, frame = cap.read()
            except Exception:  # noqa: BLE001
                return False, None
            if self._stopping():
                return False, None
            return bool(ok), frame

        timeout = float(self.cfg["read_timeout_sec"])
        with self._cond:
            end = time.time() + timeout
            while self._seq == self._last_seq and not self._finished:
                if self._stopping():
                    return False, None
                remaining = end - time.time()
                if remaining <= 0:
                    return False, None
                self._cond.wait(timeout=min(remaining, 0.15))
            if self._stopping() or self._seq == self._last_seq:
                return False, None
            # Always take the newest frame (reader overwrites while we process).
            self._last_seq = self._seq
            frame = self._frame
            if frame is None:
                return False, None
            # Copy: OpenCV may reuse the capture buffer on the next grab.
            return True, frame.copy()

    def release(self) -> None:
        self.interrupt()
        t = self._thread
        self._thread = None
        if t is not None:
            t.join(timeout=3.0)
            if t.is_alive():
                # Still blocked in cap.read() — do NOT release from this thread
                # (macOS AVFoundation segfault). Daemon will release in its finally.
                log.warning(
                    "FrameSource reader still alive after stop; deferring capture release"
                )
                return
        self._release_cap()
