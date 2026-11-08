"""ByteTrack stage-2: low-confidence dets may continue tracks, never birth them."""
import numpy as np

from src.tracking import ByteTrackLite


def _cfg(**over):
    base = {
        "high_conf": 0.5,
        "new_track_conf": 0.5,
        "low_conf": 0.1,
        "iou_threshold": 0.3,
        "low_iou_threshold": 0.25,
        "min_hits": 1,
        "max_lost_frames": 30,
    }
    base.update(over)
    return base


def test_low_conf_det_rescues_existing_track():
    tr = ByteTrackLite(_cfg(), detect_interval=0)
    tr.set_time(0.0)
    # Birth with a high-confidence detection.
    high = np.array([[10, 10, 50, 50, 0.9]], dtype=np.float32)
    active, _ = tr.update(high, (200, 200))
    assert len(active) == 1
    tid = active[0].id

    # Next frame: only a weak det near the same box — must keep the same ID.
    tr.set_time(0.04)
    low = np.array([[12, 12, 52, 52, 0.25]], dtype=np.float32)
    active, removed = tr.update(low, (200, 200))
    assert len(removed) == 0
    assert len(active) == 1
    assert active[0].id == tid
    assert active[0].matched_now is True
    assert active[0].conf == 0.25


def test_low_conf_det_does_not_birth_new_track():
    tr = ByteTrackLite(_cfg(), detect_interval=0)
    tr.set_time(0.0)
    low_only = np.array([[10, 10, 50, 50, 0.25]], dtype=np.float32)
    active, _ = tr.update(low_only, (200, 200))
    assert active == []


def test_high_conf_still_births_new_track():
    tr = ByteTrackLite(_cfg(), detect_interval=0)
    tr.set_time(0.0)
    high = np.array([[10, 10, 50, 50, 0.8]], dtype=np.float32)
    active, _ = tr.update(high, (200, 200))
    assert len(active) == 1
