"""End-to-end logic test with synthetic video: coloured squares stand in for faces.
Runs without YOLO / InsightFace so it is fast and CI-friendly."""
import os

import numpy as np
import pytest

from src.config import load_config
from src.database import Database
from src.event_logger import EventLogger
from src.pipeline import FaceTrackingPipeline
from src.tracker import iou_matrix

RED, GREEN = (0, 0, 255), (0, 255, 0)


class FakeDetector:
    """Finds exact-colour squares in the frame."""
    def detect(self, frame):
        out = []
        for colour in (RED, GREEN):
            ys, xs = np.where(np.all(frame == colour, axis=2))
            if len(xs):
                out.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 0.9])
        return np.array(out, dtype=np.float32).reshape(-1, 5)


class FakeEmbedder:
    """One-hot 'embedding' keyed on which colour dominates the crop."""
    def embed(self, crop):
        red = int(np.all(crop == RED, axis=2).sum())
        green = int(np.all(crop == GREEN, axis=2).sum())
        v = np.zeros(512, dtype=np.float32)
        v[0 if red >= green else 1] = 1.0
        return v, 0.99


def make_frame(a_x=None, b_x=None):
    f = np.zeros((240, 320, 3), dtype=np.uint8)
    if a_x is not None:
        f[60:140, a_x:a_x + 80] = RED
    if b_x is not None:
        f[100:180, b_x:b_x + 80] = GREEN
    return f


def _base_cfg():
    cfg = load_config("config.json")
    cfg["quality"].update(min_face_px=10, min_blur_var=0)
    cfg["recognition"]["min_embeds_to_register"] = 2
    cfg["away_walkers"] = {"enabled": False}
    cfg["appearance"] = {"enabled": False}
    cfg["timestamps"] = "video"
    cfg["presence"] = {
        "exit_grace_seconds": 2.0,
        "edge_exit_grace_seconds": 0.5,
        "stationary_grace_seconds": 5.0,
        "occlusion_grace_multiplier": 2.0,
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
    return cfg


def run(tmp_path, skip):
    cfg = _base_cfg()
    cfg["detection_skip_frames"] = skip
    cfg["tracker"]["max_lost_frames"] = 20
    # Gap A ~1.56 s at 25 fps — shorter than exit_grace so mid-gap must not exit;
    # but we force short grace and then a long enough absence before re-entry.
    cfg["presence"]["exit_grace_seconds"] = 0.4
    cfg["presence"]["stationary_grace_seconds"] = 0.4
    cfg["presence"]["edge_exit_grace_seconds"] = 0.2
    db = Database(str(tmp_path / "t.db"), reset=True)
    ev = EventLogger(str(tmp_path / "logs"), str(tmp_path / "logs" / "events.log"))
    pipe = FaceTrackingPipeline(cfg, FakeDetector(), FakeEmbedder(), db, ev)
    pipe.set_timing(25.0, live=False)
    for i in range(110):
        a = 20 + i if i <= 40 else (30 if 80 <= i <= 100 else None)
        b = 200 if 20 <= i <= 60 else None
        pipe.process_frame(make_frame(a if i <= 40 or 80 <= i <= 100 else None, b), i, ts=i / 25.0)
    pipe.finalize()
    return db


@pytest.mark.parametrize("skip", [0, 2])
def test_unique_count_and_entry_exit(tmp_path, skip):
    db = run(tmp_path, skip)
    assert db.unique_count() == 2
    counts = db.event_counts()
    assert counts["entry"] == 3 and counts["exit"] == 3
    per_face = db.query("SELECT face_id, event_type, COUNT(*) FROM events GROUP BY 1, 2")
    assert sorted(per_face) == [(1, "entry", 2), (1, "exit", 2), (2, "entry", 1), (2, "exit", 1)]
    assert (tmp_path / "logs" / "events.log").read_text().count("FACE_REGISTERED") == 2
    for (path,) in db.query("SELECT image_path FROM events"):
        assert path and os.path.exists(path)
    faces = db.query("SELECT image_path FROM faces")
    assert len(faces) == 2 and all(p and os.path.exists(p) for (p,) in faces)
    embs = db.query("SELECT image_path FROM embeddings")
    assert embs and all(p and os.path.exists(p) for (p,) in embs)


def test_brief_gap_does_not_flicker_entry_exit(tmp_path):
    """A few missing frames must resume the same visit — no EXIT/ENTRY flicker."""
    cfg = _base_cfg()
    cfg["detection_skip_frames"] = 0
    cfg["tracker"]["max_lost_frames"] = 5
    cfg["presence"]["exit_grace_seconds"] = 5.0
    cfg["presence"]["stationary_grace_seconds"] = 5.0
    db = Database(str(tmp_path / "flicker.db"), reset=True)
    ev = EventLogger(str(tmp_path / "logs"), str(tmp_path / "logs" / "events.log"))
    pipe = FaceTrackingPipeline(cfg, FakeDetector(), FakeEmbedder(), db, ev)
    pipe.set_timing(25.0, live=False)
    for i in range(30):
        pipe.process_frame(make_frame(30, None), i, ts=i / 25.0)
    for i in range(30, 40):
        pipe.process_frame(make_frame(None, None), i, ts=i / 25.0)
    for i in range(40, 70):
        pipe.process_frame(make_frame(30, None), i, ts=i / 25.0)
    pipe.finalize()
    counts = db.event_counts()
    assert counts.get("entry") == 1 and counts.get("exit") == 1
    log = (tmp_path / "logs" / "events.log").read_text()
    assert "VISIT_RESUMED" in log
    assert log.count("ENTRY") == 1
    db.close()


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 1 / 3, 0.0], atol=1e-6)


def test_multi_video_session_keeps_identities(tmp_path):
    """Same person in two consecutive videos (shared pipeline/DB) counts once."""
    cfg = _base_cfg()
    db = Database(str(tmp_path / "t.db"), reset=True)
    ev = EventLogger(str(tmp_path / "logs"), str(tmp_path / "logs" / "events.log"))
    pipe = FaceTrackingPipeline(cfg, FakeDetector(), FakeEmbedder(), db, ev)
    pipe.set_timing(25.0, live=False)
    for _video in range(2):
        for i in range(40):
            pipe.process_frame(make_frame(30, None), i, ts=i / 25.0)
        pipe.finalize()
    assert db.unique_count() == 1
    assert db.event_counts() == {"entry": 2, "exit": 2}
