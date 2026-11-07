"""Timestamp / clock invariants — mono must never leak into wall dates."""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from src import clock as clk
from src.config import load_config
from src.database import Database
from src.event_logger import EventLogger
from src.pipeline import FaceTrackingPipeline
from tests.test_pipeline import FakeDetector, FakeEmbedder, make_frame


def test_wall_from_mono_subtracts_elapsed(monkeypatch):
    fixed_wall = datetime(2026, 11, 7, 12, 0, 0)
    monkeypatch.setattr(clk, "wall_now", lambda: fixed_wall)
    monkeypatch.setattr(clk, "mono", lambda: 100_000.0)
    got = clk.wall_from_mono(99_993.0)  # 7 s ago
    assert abs((got - (fixed_wall - timedelta(seconds=7))).total_seconds()) < 0.1


def test_mono_never_added_to_wall_for_entry_exit(tmp_path, monkeypatch):
    """Monkeypatch mono to 132000+; ENTRY/EXIT/last_seen/folder must stay near wall_now."""
    fixed_wall = datetime(2026, 11, 7, 12, 56, 38, 123000)
    mono_val = {"v": 132_000.0}

    monkeypatch.setattr(clk, "wall_now", lambda: fixed_wall)
    monkeypatch.setattr(clk, "mono", lambda: mono_val["v"])

    cfg = load_config("config.json")
    cfg["quality"].update(min_face_px=10, min_blur_var=0)
    cfg["recognition"]["min_embeds_to_register"] = 2
    cfg["away_walkers"] = {"enabled": False}
    cfg["appearance"] = {"enabled": False}
    cfg["timestamps"] = "wall"
    cfg["detection_skip_frames"] = 0
    cfg["tracker"]["max_lost_frames"] = 3
    cfg["presence"] = {
        "exit_grace_seconds": 0.2,
        "edge_exit_grace_seconds": 0.1,
        "stationary_grace_seconds": 0.2,
        "occlusion_grace_multiplier": 1.0,
        "edge_margin_px": 24,
        "stationary_speed_px_s": 15,
        "resume_spatial_iou": 0.2,
        "resume_threshold_margin": 0.05,
        "reverify_interval_seconds": 100,
        "occlusion_iou": 0.3,
        "mass_loss_ratio": 0.6,
        "max_stream_gap_seconds": 20,
        "exit_zones": [],
    }
    db = Database(str(tmp_path / "t.db"), reset=True)
    ev = EventLogger(str(tmp_path / "logs"), str(tmp_path / "logs" / "events.log"))
    pipe = FaceTrackingPipeline(cfg, FakeDetector(), FakeEmbedder(), db, ev)
    pipe.set_timing(25.0, live=True)
    assert pipe.time_mode == "wall"

    for i in range(12):
        mono_val["v"] = 132_000.0 + i * 0.04
        pipe.process_frame(make_frame(40, None), i, ts=None)
    # Disappear long enough for exit grace.
    for i in range(12, 40):
        mono_val["v"] = 132_000.0 + i * 0.04
        pipe.process_frame(make_frame(None, None), i, ts=None)
    pipe.finalize()

    rows = db.query("SELECT event_type, timestamp, image_path FROM events ORDER BY id")
    assert rows, "expected ENTRY/EXIT rows"
    for etype, ts, path in rows:
        dt = datetime.fromisoformat(ts)
        skew = abs((dt - fixed_wall).total_seconds())
        assert skew < 2.0, f"{etype} ts={ts} skew={skew}s (mono leaked into wall?)"
        assert "2026-11-07" in (path or ""), f"image folder wrong date: {path}"
        assert "2026-11-09" not in (path or "")


def test_last_seen_reflects_mono_age(tmp_path, monkeypatch):
    """Track last seen 7 s ago (by mono) → last_seen == wall_now - 7 s."""
    fixed_wall = datetime(2026, 11, 7, 15, 0, 0)
    mono_now = 50_000.0
    monkeypatch.setattr(clk, "wall_now", lambda: fixed_wall)
    monkeypatch.setattr(clk, "mono", lambda: mono_now)

    cfg = load_config("config.json")
    cfg["quality"].update(min_face_px=10, min_blur_var=0)
    cfg["recognition"]["min_embeds_to_register"] = 2
    cfg["away_walkers"] = {"enabled": False}
    cfg["appearance"] = {"enabled": False}
    cfg["timestamps"] = "wall"
    cfg["detection_skip_frames"] = 0
    cfg["tracker"]["max_lost_frames"] = 2
    cfg["presence"] = {
        "exit_grace_seconds": 0.15,
        "edge_exit_grace_seconds": 0.1,
        "stationary_grace_seconds": 0.15,
        "occlusion_grace_multiplier": 1.0,
        "edge_margin_px": 24,
        "stationary_speed_px_s": 15,
        "resume_spatial_iou": 0.2,
        "resume_threshold_margin": 0.05,
        "reverify_interval_seconds": 100,
        "occlusion_iou": 0.3,
        "mass_loss_ratio": 0.6,
        "max_stream_gap_seconds": 20,
        "exit_zones": [],
    }
    db = Database(str(tmp_path / "t.db"), reset=True)
    ev = EventLogger(str(tmp_path / "logs"), str(tmp_path / "logs" / "events.log"))
    pipe = FaceTrackingPipeline(cfg, FakeDetector(), FakeEmbedder(), db, ev)
    pipe.set_timing(25.0, live=True)

    # Present for a few frames at mono=50000.
    for i in range(8):
        pipe.process_frame(make_frame(50, None), i, ts=None)

    # Jump mono +7 s with no detections → last_seen should stay ~7 s behind wall.
    monkeypatch.setattr(clk, "mono", lambda: mono_now + 7.0)
    for i in range(8, 30):
        pipe.process_frame(make_frame(None, None), i, ts=None)
    pipe.finalize()

    exits = db.query("SELECT timestamp FROM events WHERE event_type='exit'")
    assert exits
    exit_dt = datetime.fromisoformat(exits[0][0])
    expected = fixed_wall - timedelta(seconds=7)
    assert abs((exit_dt - expected).total_seconds()) < 0.5
