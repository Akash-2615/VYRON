"""Overlay drawing for the live window / annotated video / dashboard preview."""
import cv2

from .appearance import body_box


def _label_bg(frame, text, org, colour, scale=0.55, thickness=2):
    x, y = org
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    ty = max(th + 4, y)
    cv2.rectangle(frame, (x, ty - th - 6), (x + tw + 6, ty + 2), colour, -1)
    cv2.putText(frame, text, (x + 3, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness)


def draw_overlay(frame, tracks, unique_count: int, fps: float = 0.0,
                 away_tracks=None, away_unique: int = 0):
    # People facing away (backs) — magenta full-body boxes
    for t in away_tracks or []:
        x1, y1, x2, y2 = [int(v) for v in t.box]
        colour = (255, 0, 200)  # magenta-ish BGR
        aid = getattr(t, "away_id", None)
        label = f"AWAY {aid}" if aid is not None else "AWAY ..."
        conf = getattr(t, "conf", None)
        if conf is not None:
            label += f" {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
        appearance = getattr(t, "appearance_summary", None)
        tag = appearance or "walking away"
        short = tag if len(tag) <= 42 else tag[:39] + "..."
        _label_bg(frame, label, (x1, y1), colour)
        _label_bg(frame, short, (x1, min(y2 - 4, frame.shape[0] - 4)), colour, scale=0.5)

    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.box]
        conf = getattr(t, "conf", None)
        conf_s = f" {conf:.2f}" if conf is not None else ""
        if t.face_id is not None:
            colour = (0, 200, 0) if t.owns else (0, 160, 255)
            label = f"ID {t.face_id}{conf_s}"
        else:
            colour, label = (0, 200, 255), f"...{conf_s}"

        person = body_box(t.box, frame.shape)
        if person is not None:
            px1, py1, px2, py2 = person
            cv2.rectangle(frame, (px1, py1), (px2, py2), colour, 3)
            appearance = getattr(t, "appearance_summary", None)
            tag = appearance or "person"
            short = tag if len(tag) <= 42 else tag[:39] + "..."
            _label_bg(frame, short, (px1, max(py1 + 18, 18)), colour, scale=0.5)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        _label_bg(frame, label, (x1, y1), colour)

    cv2.putText(frame,
                f"Unique visitors: {unique_count}   Away (backs): {away_unique}   FPS: {fps:.1f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame
