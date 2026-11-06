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


def run(tmp_path, skip):
    cfg = load_config("config.json")
    cfg["detection_skip_frames"] = skip
    cfg["tracker"]["max_lost_frames"] = 20
    cfg["quality"].update(min_face_px=10, min_blur_var=0)
    cfg["recognition"]["min_embeds_to_register"] = 2
    db = Database(str(tmp_path / "t.db"), reset=True)
    ev = EventLogger(str(tmp_path / "logs"), str(tmp_path / "logs" / "events.log"))
    pipe = FaceTrackingPipeline(cfg, FakeDetector(), FakeEmbedder(), db, ev)
    for i in range(110):
        a = 20 + i if i <= 40 else (30 if 80 <= i <= 100 else None)   # A: 0-40, re-enters 80-100
        b = 200 if 20 <= i <= 60 else None                          # B: 20-60
        pipe.process_frame(make_frame(a if i <= 40 or 80 <= i <= 100 else None, b), i)  # returns (faces, away)
    pipe.finalize()
    return db


@pytest.mark.parametrize("skip", [0, 2])
def test_unique_count_and_entry_exit(tmp_path, skip):
    db = run(tmp_path, skip)
    assert db.unique_count() == 2                       # A re-entering must NOT add a visitor
    counts = db.event_counts()
    assert counts["entry"] == 3 and counts["exit"] == 3  # A, B, A again -> one exit each
    per_face = db.query("SELECT face_id, event_type, COUNT(*) FROM events GROUP BY 1, 2")
    assert sorted(per_face) == [(1, "entry", 2), (1, "exit", 2), (2, "entry", 1), (2, "exit", 1)]
    assert (tmp_path / "logs" / "events.log").read_text().count("FACE_REGISTERED") == 2
    for (path,) in db.query("SELECT image_path FROM events"):
        assert path and os.path.exists(path)
    faces = db.query("SELECT image_path FROM faces")
    assert len(faces) == 2 and all(p and os.path.exists(p) for (p,) in faces)       # registered close-up
    embs = db.query("SELECT image_path FROM embeddings")
    assert embs and all(p and os.path.exists(p) for (p,) in embs)                   # crop per embedding


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 1 / 3, 0.0], atol=1e-6)


def test_multi_video_session_keeps_identities(tmp_path):
    """Same person in two consecutive videos (shared pipeline/DB) counts once."""
    cfg = load_config("config.json")
    cfg["quality"].update(min_face_px=10, min_blur_var=0)
    db = Database(str(tmp_path / "t.db"), reset=True)
    ev = EventLogger(str(tmp_path / "logs"), str(tmp_path / "logs" / "events.log"))
    pipe = FaceTrackingPipeline(cfg, FakeDetector(), FakeEmbedder(), db, ev)
    for _video in range(2):
        for i in range(40):
            pipe.process_frame(make_frame(30, None), i)  # returns (faces, away)
        pipe.finalize()                                   # what main.py does between videos
    assert db.unique_count() == 1
    assert db.event_counts() == {"entry": 2, "exit": 2}
