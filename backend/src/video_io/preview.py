"""Publish the latest annotated frame for the dashboard live view.

JPEG encode + disk write runs on a background thread so the tracker never
blocks on preview I/O. Only the newest frame is kept (older pending frames drop).
If the worker thread dies, publish falls back to a synchronous write so Live
Monitor does not stay stuck on a stale JPEG.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _atomic_replace(tmp: Path, dest: Path) -> None:
    """Replace dest with tmp; fall back to unlink+rename if replace fails (busy readers)."""
    try:
        tmp.replace(dest)
        return
    except OSError:
        pass
    try:
        dest.unlink(missing_ok=True)
        tmp.replace(dest)
    except OSError as exc:
        try:
            # Last resort: copy bytes over the destination.
            dest.write_bytes(tmp.read_bytes())
            tmp.unlink(missing_ok=True)
        except OSError as exc2:
            tmp.unlink(missing_ok=True)
            raise OSError(f"atomic replace failed for {dest}: {exc} / {exc2}") from exc2


def publish_live_stats(stats: dict, path: str = "data/runtime/live_stats.json") -> None:
    """Atomic JSON snapshot of who is currently in the live frame (for the dashboard)."""
    if not path:
        return
    from src.core.paths import resolve_under_root
    p = resolve_under_root(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(stats)
    payload["ts"] = time.time()
    tmp = p.parent / f".{p.stem}.{os.getpid()}.tmp.json"
    try:
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        _atomic_replace(tmp, p)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        log.warning("live_stats write failed path=%s err=%s", p, exc)


def _encode_preview(frame: np.ndarray, path: str, quality: int, max_width: int) -> None:
    vis = frame
    if max_width and vis.shape[1] > int(max_width):
        scale = int(max_width) / float(vis.shape[1])
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    from src.core.paths import resolve_under_root
    p = resolve_under_root(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.parent / f".{p.stem}.{os.getpid()}.tmp{p.suffix or '.jpg'}"
    q = max(1, min(100, int(quality)))
    if not cv2.imwrite(str(tmp), vis, [int(cv2.IMWRITE_JPEG_QUALITY), q]):
        tmp.unlink(missing_ok=True)
        log.warning("live preview imwrite failed path=%s", p)
        return
    try:
        _atomic_replace(tmp, p)
    except OSError as exc:
        log.warning("live preview replace failed path=%s err=%s", p, exc)


def publish_preview(frame, path: str, quality: int = 92, max_width: int = 0) -> None:
    """Synchronous JPEG write (tests / one-shot). Prefer PreviewPublisher in the hot path."""
    if not path or frame is None:
        return
    _encode_preview(frame, path, quality, max_width)


class PreviewPublisher:
    """Latest-frame preview writer. ``publish`` never blocks on JPEG/disk."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending: Optional[Tuple[Any, str, int, int]] = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="live-preview", daemon=True)
        self._thread.start()

    def _ensure_thread(self) -> bool:
        """Restart the daemon if it exited (otherwise Live Monitor stays stale forever)."""
        t = self._thread
        if t is not None and t.is_alive():
            return True
        with self._cond:
            if self._thread is not None and self._thread.is_alive():
                return True
            self._stop = False
            self._thread = threading.Thread(target=self._run, name="live-preview", daemon=True)
            self._thread.start()
            log.warning("live-preview worker restarted")
            return True

    def publish(self, frame, path: str, quality: int = 72, max_width: int = 0) -> None:
        if not path or frame is None:
            return
        item = (frame.copy(), path, int(quality), int(max_width or 0))
        if not self._ensure_thread():
            _encode_preview(item[0], item[1], item[2], item[3])
            return
        with self._cond:
            alive = self._thread is not None and self._thread.is_alive()
            if not alive:
                # Last resort: never drop the frame silently.
                pass
            else:
                self._pending = item
                self._cond.notify()
                return
        _encode_preview(item[0], item[1], item[2], item[3])

    def flush(self, timeout: float = 1.0) -> None:
        """Wait until the current pending frame is written (best-effort)."""
        deadline = time.time() + max(0.0, timeout)
        while time.time() < deadline:
            with self._lock:
                if self._pending is None:
                    return
            time.sleep(0.01)

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()
        if self._thread is not None:
            self._thread.join(timeout=1.5)

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait(timeout=0.5)
                if self._pending is None and self._stop:
                    return
                item = self._pending
                self._pending = None
            if item is None:
                continue
            frame, path, quality, max_width = item
            try:
                _encode_preview(frame, path, quality, max_width)
            except Exception:  # noqa: BLE001
                log.exception("live preview encode failed")


_default_publisher: Optional[PreviewPublisher] = None
_default_lock = threading.Lock()


def default_publisher() -> PreviewPublisher:
    global _default_publisher
    with _default_lock:
        if _default_publisher is None:
            _default_publisher = PreviewPublisher()
        elif (
            _default_publisher._thread is None
            or not _default_publisher._thread.is_alive()
        ):
            # Recreate if the previous singleton's worker died.
            try:
                _default_publisher.close()
            except Exception:  # noqa: BLE001
                pass
            _default_publisher = PreviewPublisher()
        return _default_publisher
