"""Appearance helpers work without downloading OpenCLIP (enabled=False)."""
import numpy as np

from src.appearance import AppearanceAnalyzer, body_box, body_crop, person_box_for_face
from src.database import Database


def test_body_crop_and_color_fallback():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[80:200, 100:180] = (200, 40, 40)  # blue-ish BGR torso
    face = [120, 40, 160, 80]
    crop = body_crop(frame, face)
    assert crop is not None and crop.shape[0] > 40

    analyzer = AppearanceAnalyzer({"enabled": False})  # no CLIP weights
    info = analyzer.describe(frame, face)
    assert info is not None
    assert info["summary"]
    assert "dominant_colors" in info


def test_person_box_prefers_smallest_containing_det():
    face = [100, 40, 140, 80]
    dets = np.array([
        [0, 0, 300, 240, 0.9],     # huge neighbour that also contains the face
        [90, 30, 150, 200, 0.9],   # tight person box for this face
        [200, 20, 280, 220, 0.9],  # other person, does not contain face centre
    ], dtype=np.float32)
    box = person_box_for_face(face, (240, 320), dets)
    assert box == (90, 30, 150, 200)
    # Heuristic half-width is ~1.15× face width (full width ~2.3×), tighter than the old 1.8×.
    hb = body_box(face, (240, 320))
    assert hb is not None
    fw = face[2] - face[0]
    assert (hb[2] - hb[0]) <= int(2.4 * fw) + 2
    assert (hb[2] - hb[0]) < int(3.5 * fw)


def test_db_appearance_columns(tmp_path):
    db = Database(str(tmp_path / "a.db"), reset=True)
    fid = db.add_face("2026-11-06T10:00:00.000")
    assert db.get_appearance_summary(fid) is None
    db.set_appearance(fid, "blue top, jeans", '{"summary":"blue top, jeans"}',
                      image_path="logs/appearance/x.jpg")
    assert db.get_appearance_summary(fid) == "blue top, jeans"
    assert db.get_appearance_image(fid) == "logs/appearance/x.jpg"
    db.close()
