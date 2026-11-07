"""Ghost-box overlay: unmatched / coasting tracks must not be painted."""
import numpy as np

from src.tracker import Track
from src.visualize import draw_overlay


def test_overlay_hides_stale_tracks():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    live = Track(np.array([40, 40, 80, 90], dtype=np.float64), 0.9)
    live.face_id = 1
    live.owns = True
    live.time_since_update = 0

    ghost = Track(np.array([200, 100, 240, 150], dtype=np.float64), 0.7)
    ghost.face_id = 99
    ghost.owns = True
    ghost.time_since_update = 20  # lost — empty floor

    away_ghost = Track(np.array([10, 10, 50, 180], dtype=np.float64), 0.6)
    away_ghost.away_id = 7
    away_ghost.time_since_update = 30

    out = draw_overlay(frame.copy(), [live, ghost], away_tracks=[away_ghost],
                       max_box_age=3)
    # Live face paints a white stroke on its box edge.
    assert out[40, 40].sum() > 0
    # Ghost face region must stay black (no white/green stroke).
    assert out[100, 200].sum() == 0
    assert out[125, 220].sum() == 0
    # Away ghost undrawn (sample mid-box, clear of the live body expansion).
    assert out[160, 30].sum() == 0


def test_predict_freezes_after_skip_coast():
    t = Track(np.array([10, 10, 50, 50], dtype=np.float64), 0.9)
    t.vel = np.array([5.0, 5.0, 5.0, 5.0])
    t.speed_px_s = 20.0  # moving — allow coast during skip
    t.predict(detect_interval=2)  # tsu=1 — coasts
    t.predict(detect_interval=2)  # tsu=2 — coasts
    assert t.box[0] == 20.0
    frozen = t.box.copy()
    t.predict(detect_interval=2)  # tsu=3 — freeze
    assert np.allclose(t.box, frozen)
    t.predict(detect_interval=2)
    assert np.allclose(t.box, frozen)


def test_overlay_nms_drops_trailing_duplicate():
    """Walker leaves a stale box behind — only the fresh match is painted."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    fresh = Track(np.array([180, 80, 220, 130], dtype=np.float64), 0.95)
    fresh.face_id = 4
    fresh.owns = True
    fresh.time_since_update = 0
    fresh.matched_now = True

    trail = Track(np.array([160, 70, 200, 120], dtype=np.float64), 0.7)
    trail.face_id = 13
    trail.owns = False
    trail.time_since_update = 1
    trail.matched_now = False

    out = draw_overlay(frame.copy(), [fresh, trail], max_box_age=2)
    # Fresh face edge painted.
    assert out[80, 180].sum() > 0
    # Trailing ghost near old centre should not get its own stroke if NMS kept fresh.
    # Sample a pixel on trail's top-left corner that is outside the fresh box.
    assert out[70, 160].sum() == 0


def test_overlay_draws_outer_person_box():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    face = Track(np.array([140, 40, 180, 90], dtype=np.float64), 0.9)
    face.face_id = 7
    face.owns = True
    face.time_since_update = 1  # coasting — person box must still show
    out = draw_overlay(frame.copy(), [face], max_box_age=3)
    # body_box is a stroke only — sample its left edge below the face.
    assert out[120, 88].sum() > 0
