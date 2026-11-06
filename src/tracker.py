"""ByteTrack-style multi-object tracker (IoU + constant-velocity prediction).

* Two-stage association: high-confidence detections first, then low-confidence
  detections against the still-unmatched tracks (the core idea of ByteTrack).
* Between detection cycles (detection_skip_frames) tracks coast using their
  estimated velocity, so IDs persist while YOLO is skipped.
* Tracks are dropped after `max_lost_frames` unmatched frames, or immediately
  once the predicted centre leaves the frame (face walked out).
"""
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


class Track:
    _next_id = 1

    def __init__(self, box: np.ndarray, conf: float):
        self.id = Track._next_id
        Track._next_id += 1
        self.box = box.astype(np.float64).copy()
        self.vel = np.zeros(4)
        self.conf = conf
        self.hits = 1
        self.time_since_update = 0
        self.matched_now = True
        # --- identity / pipeline state (managed by pipeline.py) ---
        self.face_id: Optional[int] = None
        self.owns = False                 # True => this track emits the entry/exit for face_id
        self.announced = False
        self.rejected = False
        self.embed_attempts = 0
        self.cycles_since_attempt = 10 ** 6
        self.pending: list = []
        self.last_crop = None
        self.appearance_summary: Optional[str] = None  # clothing / person tags for overlay
        self.kind: str = "face"                       # "face" | "away"
        self.away_id: Optional[int] = None

    def predict(self, detect_interval: int) -> None:
        self.box = self.box + self.vel
        self.time_since_update += 1
        if self.time_since_update > detect_interval:
            self.vel *= 0.8  # coasting for too long: stop trusting the velocity

    def update(self, box: np.ndarray, conf: float) -> None:
        frames = max(1, self.time_since_update)
        new_vel = (box - self.box) / frames + self.vel  # measured displacement beyond prediction
        self.vel = 0.5 * self.vel + 0.5 * new_vel
        self.box = box.astype(np.float64).copy()
        self.conf = conf
        self.hits += 1
        self.time_since_update = 0
        self.matched_now = True


class ByteTrackLite:
    def __init__(self, cfg: dict, detect_interval: int):
        self.high_conf = cfg["high_conf"]
        self.new_conf = cfg["new_track_conf"]
        self.iou_thr = cfg["iou_threshold"]
        self.min_hits = cfg["min_hits"]
        self.max_lost = cfg["max_lost_frames"]
        self.interval = detect_interval + 1
        self.tracks: List[Track] = []

    def get(self, track_id: int) -> Optional[Track]:
        return next((t for t in self.tracks if t.id == track_id), None)

    def is_confirmed(self, t: Track) -> bool:
        return t.hits >= self.min_hits

    def _match(self, tracks: List[Track], dets: np.ndarray, det_idx: List[int]):
        if not tracks or not det_idx:
            return [], list(range(len(tracks))), list(det_idx)
        iou = iou_matrix(np.array([t.box for t in tracks]), dets[det_idx, :4])
        rows, cols = linear_sum_assignment(1.0 - iou)
        matches, m_t, m_d = [], set(), set()
        for r, c in zip(rows, cols):
            if iou[r, c] >= self.iou_thr:
                matches.append((r, det_idx[c]))
                m_t.add(r)
                m_d.add(c)
        un_t = [i for i in range(len(tracks)) if i not in m_t]
        un_d = [det_idx[c] for c in range(len(det_idx)) if c not in m_d]
        return matches, un_t, un_d

    def update(self, dets: Optional[np.ndarray], frame_size: Tuple[int, int]):
        """dets: (N,5) array on detection frames, None on skipped frames.
        Returns (active_tracks, removed_tracks)."""
        for t in self.tracks:
            t.matched_now = False
            t.predict(self.interval - 1)

        if dets is not None:
            high = [i for i in range(len(dets)) if dets[i, 4] >= self.high_conf]
            low = [i for i in range(len(dets)) if dets[i, 4] < self.high_conf]

            matches, un_t, un_high = self._match(self.tracks, dets, high)
            for ti, di in matches:
                self.tracks[ti].update(dets[di, :4], float(dets[di, 4]))

            # stage 2: low-confidence detections rescue still-unmatched tracks
            rest = [self.tracks[i] for i in un_t]
            m2, _, _ = self._match(rest, dets, low)
            for ti, di in m2:
                rest[ti].update(dets[di, :4], float(dets[di, 4]))

            for di in un_high:
                if dets[di, 4] >= self.new_conf:
                    self.tracks.append(Track(dets[di, :4], float(dets[di, 4])))

        w, h = frame_size
        keep, removed = [], []
        for t in self.tracks:
            cx, cy = (t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2
            limit = self.max_lost if self.is_confirmed(t) else self.interval
            out_of_frame = t.time_since_update > 0 and not (0 <= cx <= w and 0 <= cy <= h)
            if t.time_since_update > limit or out_of_frame:
                removed.append(t)
            else:
                keep.append(t)
        self.tracks = keep
        return list(self.tracks), removed
