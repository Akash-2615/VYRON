"""Overlay drawing for the live window / annotated video / dashboard preview."""
import cv2
import numpy as np

from src.recognition.appearance import body_box


def _sharp_label(frame, text, org, fg, scale=0.65, thickness=2):
    """High-contrast outlined label (no soft filled banner that blurs under JPEG)."""
    x, y = int(org[0]), int(org[1])
    y = max(18, y)
    # Dark outline for crisp edges on any background
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv2.LINE_AA)


def _box(frame, x1, y1, x2, y2, colour, thickness=2):
    """Pixel-aligned rectangle so edges stay sharp."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness, lineType=cv2.LINE_8)


def _is_visible(t, max_age: int) -> bool:
    """Hide coasting / lost tracks so empty-floor ghost boxes are not drawn."""
    return int(getattr(t, "time_since_update", 0) or 0) <= max_age


def _center(box):
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _boxes_conflict(a, b, iou_thr: float = 0.22, center_frac: float = 0.55) -> bool:
    """True when two boxes are the same person (overlap or centres too close)."""
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    iou = inter / max(area_a + area_b - inter, 1e-9)
    if iou >= iou_thr:
        return True
    acx, acy = _center(a)
    bcx, bcy = _center(b)
    diag = max(np.hypot(ax2 - ax1, ay2 - ay1), np.hypot(bx2 - bx1, by2 - by1), 1.0)
    return np.hypot(acx - bcx, acy - bcy) <= center_frac * diag


def _is_stable(t) -> bool:
    """Prefer confirmed / identified tracks — single-frame flashes still clutter crowds."""
    if getattr(t, "face_id", None) is not None:
        return True
    if getattr(t, "away_id", None) is not None:
        return True
    # Confirmed-ish (2+ hits): show cyan "…" while embedding so users see progress to green.
    return int(getattr(t, "hits", 0) or 0) >= 2


def _nms_tracks(tracks, max_age: int, *, require_stable: bool = False):
    """Keep the freshest box per person — drops trailing 'ghost' duplicates."""
    cand = [t for t in tracks if _is_visible(t, max_age)]
    if require_stable:
        cand = [t for t in cand if _is_stable(t)]
    ranked = sorted(
        cand,
        key=lambda t: (
            0 if _is_stable(t) else 1,
            int(getattr(t, "time_since_update", 0) or 0),
            0 if getattr(t, "owns", False) else 1,
            0 if getattr(t, "matched_now", False) else 1,
            -float(getattr(t, "conf", 0) or 0),
        ),
    )
    keep = []
    for t in ranked:
        if any(_boxes_conflict(t.box, k.box) for k in keep):
            continue
        keep.append(t)
    return keep


def draw_overlay(frame, tracks, unique_count: int = 0, fps: float = 0.0,
                 away_tracks=None, away_unique: int = 0, max_box_age: int = 3,
                 away_max_box_age: int | None = None):
    """Draw crisp boxes + ID/confidence only. No Unique/Away/FPS HUD on the frame.

    max_box_age: face tracks matched within this many frames (detection-skip coast).
    away_max_box_age: person/no-face tracks (defaults to max_box_age; pass a larger
    value when person YOLO runs less often than face YOLO).
    Overlapping stale tracks are NMS'd so walkers don't leave trail ghosts.
    Tentative (unconfirmed) flashes are hidden in crowded scenes.
    """
    age = max(0, int(max_box_age))
    away_age = age if away_max_box_age is None else max(0, int(away_max_box_age))
    face_draw = _nms_tracks(list(tracks or []), age, require_stable=True)
    away_draw = _nms_tracks(list(away_tracks or []), away_age, require_stable=True)

    # Drop no-face boxes that sit on a face visitor (same body).
    away_clean = []
    for at in away_draw:
        if any(_boxes_conflict(at.box, ft.box, iou_thr=0.12, center_frac=0.65)
               for ft in face_draw):
            continue
        away_clean.append(at)

    # No-face people — magenta body box + conf
    for t in away_clean:
        x1, y1, x2, y2 = [int(v) for v in t.box]
        colour = (255, 0, 200)
        _box(frame, x1, y1, x2, y2, colour, thickness=2)
        aid = getattr(t, "away_id", None)
        conf = getattr(t, "conf", None)
        parts = ["NO-FACE"]
        if aid is not None:
            parts.append(str(aid))
        if conf is not None:
            parts.append(f"{conf:.2f}")
        _sharp_label(frame, " ".join(parts), (x1 + 2, y1 - 8), colour)

    for t in face_draw:
        x1, y1, x2, y2 = [int(v) for v in t.box]
        conf = getattr(t, "conf", None)
        if t.face_id is not None:
            colour = (0, 220, 0) if t.owns else (0, 180, 255)
            label = f"ID {t.face_id}"
        else:
            colour, label = (0, 210, 255), "…"
        if conf is not None:
            label = f"{label}  {conf:.2f}"

        # Body box only for identified visitors — tentative face flashes used to
        # spray huge cyan person boxes across crowded hallways.
        if t.face_id is not None:
            person = body_box(t.box, frame.shape)
            if person is not None:
                px1, py1, px2, py2 = person
                _box(frame, px1, py1, px2, py2, colour, thickness=2)

        # Face box — white, sharp
        _box(frame, x1, y1, x2, y2, (255, 255, 255), thickness=2)
        _sharp_label(frame, label, (x1 + 2, y1 - 8), colour)

    # unique_count / away_unique / fps intentionally not drawn on the video
    return frame
