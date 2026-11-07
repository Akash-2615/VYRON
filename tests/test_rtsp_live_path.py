"""Prove RTSP/live wiring without a real camera: stream flags, wall clock, embed path."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from src.config import load_config
from src.database import Database
from src.event_logger import EventLogger
from src.pipeline import FaceTrackingPipeline
from src.stream import FrameSource, LIVE_PREFIXES
from src.tracker import Track


def test_rtsp_url_marked_live_and_tcp_hint():
    os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
    cfg = {"reconnect_delay_sec": 1, "max_reconnects": 0, "read_timeout_sec": 1}
    src = FrameSource("rtsp://user:pass@192.168.0.5:554/h264", cfg)
    assert src.is_live is True
    assert any("rtsp" in p for p in LIVE_PREFIXES)
    assert "rtsp_transport;tcp" in os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")


def test_rtsps_and_webcam_also_live():
    cfg = {"reconnect_delay_sec": 1, "max_reconnects": 0, "read_timeout_sec": 1}
    assert FrameSource("rtsps://cam/stream", cfg).is_live
    assert FrameSource("0", cfg).is_live  # webcam index
    assert FrameSource("/tmp/file.mp4", cfg).is_live is False


def test_live_timing_uses_wall_clock_and_embeddings(tmp_path):
    """Same path main/run_rtsp use: set_timing(live=True), ts=None → identify → embed → ENTRY."""
    cfg = load_config("config.json")
    cfg["away_walkers"] = {**cfg.get("away_walkers", {}), "enabled": False}
    cfg["appearance"] = {**cfg.get("appearance", {}), "enabled": False}
    cfg["detection_skip_frames"] = 0
    cfg["tracker"]["min_hits"] = 1
    cfg["recognition"]["min_embeds_to_register"] = 1
    cfg["quality"] = {"min_face_px": 5, "min_blur_var": 0.0}

    db = Database(str(tmp_path / "live.db"), reset=True)
    ev = EventLogger(str(tmp_path / "logs"), str(tmp_path / "events.log"))

    class FakeDet:
        def detect(self, frame):
            # One face box every frame
            return np.array([[20, 20, 80, 80, 0.9]], dtype=np.float32)

    class FakeEmb:
        def __init__(self):
            self.n = 0

        def embed(self, crop):
            self.n += 1
            v = np.zeros(512, dtype=np.float32)
            v[0] = 1.0
            return v, 0.99

    emb = FakeEmb()
    pipe = FaceTrackingPipeline(cfg, FakeDet(), emb, db, ev)
    pipe.set_timing(25.0, live=True)
    assert pipe.time_mode == "wall"

    # Build a non-blank crop so quality gate passes
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[20:80, 20:80] = 180

    for i in range(4):
        pipe.process_frame(frame, i, ts=None)  # live: wall/monotonic clock

    log = Path(tmp_path / "events.log").read_text()
    assert "EMBEDDING_GENERATED" in log
    assert "FACE_REGISTERED" in log or "FACE_RECOGNIZED" in log
    assert "ENTRY" in log
    assert db.unique_count() == 1
    assert emb.n >= 1
    # Embeddings persisted
    assert len(db.load_embeddings()) >= 1
    db.close()


def test_run_source_live_skips_annotated_mp4_and_sets_live(tmp_path, monkeypatch):
    """run_source must treat RTSP as live: wall clock, no MP4 writer, live preview ok."""
    import main as main_mod
    from src.stream import FrameSource as RealFS

    frames = [
        np.full((120, 160, 3), 40, dtype=np.uint8),
        np.full((120, 160, 3), 50, dtype=np.uint8),
        None,
    ]
    state = {"i": 0}

    class FakeLiveSource:
        def __init__(self, source, cfg):
            self.source = source
            self.is_live = True
            self.cfg = cfg
            self._cap = None

        def start(self):
            return self

        def read(self):
            i = state["i"]
            state["i"] += 1
            if i >= 2:
                return False, None
            return True, frames[i]

        def release(self):
            pass

    monkeypatch.setattr(main_mod, "FrameSource", FakeLiveSource)

    cfg = load_config("config.json")
    cfg["away_walkers"]["enabled"] = False
    cfg["appearance"]["enabled"] = False
    cfg["detection_skip_frames"] = 0
    cfg["tracker"]["min_hits"] = 1
    cfg["recognition"]["min_embeds_to_register"] = 1
    cfg["quality"] = {"min_face_px": 5, "min_blur_var": 0.0}
    cfg["stream"]["realtime"] = False

    db = Database(str(tmp_path / "t.db"), reset=True)
    ev = EventLogger(str(tmp_path / "logs"), str(tmp_path / "events.log"))

    class FakeDet:
        def detect(self, frame):
            return np.array([[10, 10, 60, 60, 0.95]], dtype=np.float32)

    class FakeEmb:
        def embed(self, crop):
            v = np.zeros(512, dtype=np.float32)
            v[1] = 1.0
            return v, 0.9

    pipe = FaceTrackingPipeline(cfg, FakeDet(), FakeEmb(), db, ev)
    live_jpg = str(tmp_path / "live.jpg")
    stop = {"flag": False}
    n = main_mod.run_source(
        "rtsp://camera/test", cfg, pipe, db, ev, stop,
        display=False, max_frames=0,
        out_path=str(tmp_path / "should_not_write.mp4"),
        live_path=live_jpg, live_every=1,
    )
    # Preview JPEG is written on a background thread — wait for the last frame.
    from src.preview import default_publisher
    default_publisher().flush(timeout=2.0)
    assert n == 2
    assert pipe.time_mode == "wall"
    assert not (tmp_path / "should_not_write.mp4").exists()
    assert Path(live_jpg).exists()
    log = (tmp_path / "events.log").read_text()
    assert "SOURCE_START" in log and "live=True" in log
    db.close()
