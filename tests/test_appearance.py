"""Appearance helpers work without downloading OpenCLIP (enabled=False)."""
import numpy as np

from src.appearance import AppearanceAnalyzer, body_crop
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


def test_db_appearance_columns(tmp_path):
    db = Database(str(tmp_path / "a.db"), reset=True)
    fid = db.add_face("2026-11-06T10:00:00.000")
    assert db.get_appearance_summary(fid) is None
    db.set_appearance(fid, "blue top, jeans", '{"summary":"blue top, jeans"}')
    assert db.get_appearance_summary(fid) == "blue top, jeans"
    db.close()
