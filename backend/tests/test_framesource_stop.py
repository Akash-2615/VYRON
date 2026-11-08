"""Dashboard Stop must unblock live FrameSource waits."""
import threading
import time

from src.video_io.stream import FrameSource


CFG = {"reconnect_delay_sec": 0.2, "max_reconnects": 0, "read_timeout_sec": 5}


def test_external_stop_flag_marks_stopping():
    stop = {"flag": True}
    src = FrameSource("rtsp://127.0.0.1:9/nope", CFG, stop=stop)
    assert src.is_live
    assert src._threaded is True
    assert src._stopping() is True


def test_webcam_is_live_but_not_threaded():
    src = FrameSource("0", CFG, stop={"flag": False})
    assert src.is_live is True
    assert src._threaded is False


def test_webcam_read_returns_immediately_when_stopped():
    stop = {"flag": True}
    src = FrameSource("0", CFG, stop=stop)
    t0 = time.time()
    ok, frame = src.read()
    assert ok is False
    assert frame is None
    assert time.time() - t0 < 0.5


def test_live_read_returns_immediately_when_stopped():
    stop = {"flag": True}
    src = FrameSource("rtsp://127.0.0.1:9/nope", CFG, stop=stop)
    t0 = time.time()
    ok, frame = src.read()
    assert ok is False
    assert frame is None
    assert time.time() - t0 < 1.0


def test_interrupt_wakes_waiting_reader():
    stop = {"flag": False}
    src = FrameSource("rtsp://127.0.0.1:9/nope", CFG, stop=stop)
    # Live read waits for a frame; interrupt must release it without full timeout.
    result = {}

    def _reader():
        result["ok"], result["frame"] = src.read()

    th = threading.Thread(target=_reader, daemon=True)
    th.start()
    time.sleep(0.15)
    src.interrupt()
    th.join(timeout=2.0)
    assert not th.is_alive()
    assert result.get("ok") is False
