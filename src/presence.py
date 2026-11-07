"""Time-based visit / presence state machine.

A visit is PRESENT while the face is detected, HIDDEN while the owning track is
lost but within a grace deadline, BODY_ONLY when only a person-box keep-alive is
active, and EXITED once grace expires (or the stream ends).

Grace is chosen from *where* and *how* the person disappeared (edge / mid-frame
stationary / default), measured in seconds so it is stable across FPS and lag.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


STATE_PRESENT = "PRESENT"
STATE_HIDDEN = "HIDDEN"
STATE_BODY_ONLY = "BODY_ONLY"
STATE_EXITED = "EXITED"


@dataclass
class Visit:
    face_id: int
    entry_ts: float
    last_seen_ts: float
    state: str = STATE_PRESENT
    visit_id: Optional[int] = None
    track_id: Optional[int] = None
    hidden_since: Optional[float] = None
    deadline_ts: Optional[float] = None
    last_box: Optional[np.ndarray] = None
    last_crop: Optional[np.ndarray] = None
    last_emb: Optional[np.ndarray] = None
    visible_seconds: float = 0.0
    occluded_seconds: float = 0.0
    resume_count: int = 0
    exit_reason: Optional[str] = None
    grace_seconds: float = 10.0
    _last_tick_ts: Optional[float] = None
    _prev_box: Optional[np.ndarray] = None
    _prev_ts: Optional[float] = None

    def speed_px_s(self) -> float:
        if self.last_box is None or self._prev_box is None or self._prev_ts is None:
            return 0.0
        dt = max(1e-3, self.last_seen_ts - self._prev_ts)
        acx = (float(self.last_box[0]) + float(self.last_box[2])) * 0.5
        acy = (float(self.last_box[1]) + float(self.last_box[3])) * 0.5
        bcx = (float(self._prev_box[0]) + float(self._prev_box[2])) * 0.5
        bcy = (float(self._prev_box[1]) + float(self._prev_box[3])) * 0.5
        return float(np.hypot(acx - bcx, acy - bcy) / dt)

    def touch(self, ts: float, box=None, crop=None, emb=None, track_id: Optional[int] = None) -> None:
        if self._last_tick_ts is not None and self.state == STATE_PRESENT:
            self.visible_seconds += max(0.0, ts - self._last_tick_ts)
        if box is not None:
            self._prev_box = None if self.last_box is None else np.asarray(self.last_box[:4], dtype=np.float64).copy()
            self._prev_ts = self.last_seen_ts
            self.last_box = np.asarray(box[:4], dtype=np.float64).copy()
        if crop is not None:
            self.last_crop = crop
        if emb is not None:
            self.last_emb = np.asarray(emb, dtype=np.float32).copy()
        if track_id is not None:
            self.track_id = int(track_id)
        self.last_seen_ts = float(ts)
        self._last_tick_ts = float(ts)


def _centre(box) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _point_in_poly(x: float, y: float, poly) -> bool:
    # Ray casting; poly = [[x,y], ...]
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / max(yj - yi, 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def near_edge(box, frame_size: Tuple[int, int], margin: float) -> bool:
    w, h = frame_size
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return (x1 <= margin or y1 <= margin or x2 >= w - margin or y2 >= h - margin)


def moving_outward(box, prev_box, frame_size: Tuple[int, int]) -> bool:
    if prev_box is None:
        return False
    w, h = frame_size
    cx, cy = _centre(box)
    px, py = _centre(prev_box)
    dx, dy = cx - px, cy - py
    # Moving toward the nearest border.
    dist_left, dist_right = cx, w - cx
    dist_top, dist_bot = cy, h - cy
    nearest = min(dist_left, dist_right, dist_top, dist_bot)
    if nearest == dist_left:
        return dx < -0.5
    if nearest == dist_right:
        return dx > 0.5
    if nearest == dist_top:
        return dy < -0.5
    return dy > 0.5


def in_exit_zone(box, zones: list) -> bool:
    if not zones:
        return False
    cx, cy = _centre(box)
    for z in zones:
        poly = z.get("polygon") if isinstance(z, dict) else z
        if poly and _point_in_poly(cx, cy, poly):
            return True
    return False


class PresenceManager:
    """Owns open visits and decides hide / resume / exit using second-based grace."""

    def __init__(self, cfg: dict):
        p = cfg.get("presence", {}) or {}
        self.exit_grace = float(p.get("exit_grace_seconds", 10))
        self.edge_grace = float(p.get("edge_exit_grace_seconds", 2))
        self.stationary_grace = float(p.get("stationary_grace_seconds", 45))
        self.occ_mult = float(p.get("occlusion_grace_multiplier", 2.0))
        self.edge_margin = float(p.get("edge_margin_px", 24))
        self.stationary_speed = float(p.get("stationary_speed_px_s", 15))
        self.resume_iou = float(p.get("resume_spatial_iou", 0.2))
        self.resume_margin = float(p.get("resume_threshold_margin", 0.05))
        self.reverify_interval = float(p.get("reverify_interval_seconds", 5))
        self.occlusion_iou = float(p.get("occlusion_iou", 0.3))
        self.mass_loss_ratio = float(p.get("mass_loss_ratio", 0.6))
        self.max_stream_gap = float(p.get("max_stream_gap_seconds", 20))
        self.exit_zones = list(p.get("exit_zones") or [])
        self.visits: Dict[int, Visit] = {}  # face_id -> Visit
        self.disturbance_until: Optional[float] = None
        self.disturbance_reason: Optional[str] = None
        self._frozen_deadlines: Dict[int, float] = {}  # face_id -> remaining grace at freeze

    def open_visits(self) -> List[Visit]:
        return [v for v in self.visits.values() if v.state != STATE_EXITED]

    def hidden_count(self) -> int:
        return sum(1 for v in self.visits.values() if v.state in (STATE_HIDDEN, STATE_BODY_ONLY))

    def inside_count(self) -> int:
        return sum(1 for v in self.visits.values() if v.state != STATE_EXITED)

    def choose_grace(self, visit: Visit, frame_size: Tuple[int, int],
                     occluded_by: bool = False) -> Tuple[float, str]:
        box = visit.last_box
        if box is None:
            return self.exit_grace, "default"
        if in_exit_zone(box, self.exit_zones) or (
                near_edge(box, frame_size, self.edge_margin)
                and moving_outward(box, visit._prev_box, frame_size)):
            g, reason = self.edge_grace, "edge_exit"
        elif (not near_edge(box, frame_size, self.edge_margin * 2)
              and visit.speed_px_s() <= self.stationary_speed):
            g, reason = self.stationary_grace, "stationary"
        else:
            g, reason = self.exit_grace, "default"
        if occluded_by:
            g *= self.occ_mult
            reason = f"{reason}+occlusion"
        return float(g), reason

    def start_visit(self, face_id: int, ts: float, track_id: int, box, crop=None,
                    emb=None, visit_id: Optional[int] = None) -> Visit:
        v = Visit(
            face_id=int(face_id), entry_ts=float(ts), last_seen_ts=float(ts),
            state=STATE_PRESENT, visit_id=visit_id, track_id=int(track_id),
        )
        v.touch(ts, box=box, crop=crop, emb=emb, track_id=track_id)
        self.visits[int(face_id)] = v
        return v

    def mark_present(self, face_id: int, ts: float, track_id: int, box, crop=None, emb=None) -> Optional[Visit]:
        v = self.visits.get(int(face_id))
        if v is None or v.state == STATE_EXITED:
            return None
        was_hidden = v.state in (STATE_HIDDEN, STATE_BODY_ONLY)
        gap = 0.0
        if was_hidden and v.hidden_since is not None:
            gap = max(0.0, float(ts) - float(v.hidden_since))
            v.occluded_seconds += gap
            v.resume_count += 1
        v.state = STATE_PRESENT
        v.hidden_since = None
        v.deadline_ts = None
        v.touch(ts, box=box, crop=crop, emb=emb, track_id=track_id)
        return v if was_hidden else None  # return visit when resumed

    def hide(self, face_id: int, ts: float, frame_size: Tuple[int, int],
             occluded_by: bool = False, reason: str = "lost") -> Optional[Visit]:
        v = self.visits.get(int(face_id))
        if v is None or v.state == STATE_EXITED:
            return None
        if v.state in (STATE_HIDDEN, STATE_BODY_ONLY):
            return v
        grace, g_reason = self.choose_grace(v, frame_size, occluded_by=occluded_by)
        v.state = STATE_HIDDEN
        v.hidden_since = float(ts)
        v.grace_seconds = grace
        v.deadline_ts = float(ts) + grace
        v.exit_reason = reason if reason != "lost" else (
            "edge_exit" if g_reason.startswith("edge") else "hidden_timeout")
        v.track_id = None
        return v

    def body_only(self, face_id: int, ts: float, box) -> Optional[Visit]:
        v = self.visits.get(int(face_id))
        if v is None or v.state == STATE_EXITED:
            return None
        if v.state in (STATE_PRESENT, STATE_HIDDEN):
            if v.hidden_since is None:
                v.hidden_since = float(ts)
            v.state = STATE_BODY_ONLY
        # Extend deadline while body is visible.
        grace = max(v.grace_seconds, self.stationary_grace)
        v.deadline_ts = float(ts) + grace
        v.last_box = np.asarray(box[:4], dtype=np.float64).copy()
        v._last_tick_ts = float(ts)
        return v

    def freeze_deadlines(self, ts: float, reason: str) -> None:
        self.disturbance_until = float(ts)
        self.disturbance_reason = reason
        self._frozen_deadlines = {}
        for fid, v in self.visits.items():
            if v.state in (STATE_HIDDEN, STATE_BODY_ONLY) and v.deadline_ts is not None:
                self._frozen_deadlines[fid] = max(0.0, float(v.deadline_ts) - float(ts))

    def unfreeze_deadlines(self, ts: float) -> None:
        for fid, remaining in self._frozen_deadlines.items():
            v = self.visits.get(fid)
            if v is not None and v.state in (STATE_HIDDEN, STATE_BODY_ONLY):
                v.deadline_ts = float(ts) + remaining
        self._frozen_deadlines = {}
        self.disturbance_until = None
        self.disturbance_reason = None

    def in_disturbance(self) -> bool:
        return self.disturbance_until is not None

    def due_exits(self, ts: float) -> List[Visit]:
        if self.in_disturbance():
            return []
        due = []
        for v in list(self.visits.values()):
            if v.state not in (STATE_HIDDEN, STATE_BODY_ONLY):
                continue
            if v.deadline_ts is not None and float(ts) >= float(v.deadline_ts):
                due.append(v)
        return due

    def close(self, face_id: int, reason: str) -> Optional[Visit]:
        v = self.visits.get(int(face_id))
        if v is None or v.state == STATE_EXITED:
            return None
        v.state = STATE_EXITED
        v.exit_reason = reason or v.exit_reason or "hidden_timeout"
        v.deadline_ts = None
        return v

    def pop_exited(self, face_id: int) -> Optional[Visit]:
        v = self.visits.pop(int(face_id), None)
        return v
