"""ByteTrack-style multi-object tracker (IoU + constant-velocity prediction).

* Two-stage association: high-confidence detections first, then low-confidence
  detections against the still-unmatched tracks (the core idea of ByteTrack).
* Between detection cycles (detection_skip_frames) tracks coast using their
  estimated velocity, so IDs persist while YOLO is skipped.
* Lost confirmed tracks enter a lost-pool and can be re-matched by IoU / embedding
  until the pipeline visit deadline expires.
* Stationary prior freezes velocity; overlapping tracks freeze ID swaps.
"""
from typing import Dict, List, Optional, Tuple

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


def _centre(box) -> Tuple[float, float]:
    return (float(box[0]) + float(box[2])) * 0.5, (float(box[1]) + float(box[3])) * 0.5


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


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
        self.last_seen_ts: Optional[float] = None
        self.embedding: Optional[np.ndarray] = None
        self.speed_px_s: float = 0.0
        self.frozen_vel: bool = False
        # --- identity / pipeline state (managed by pipeline.py) ---
        self.face_id: Optional[int] = None
        self.owns = False                 # True => this track emits the entry/exit for face_id
        self.announced = False
        self.rejected = False
        self.embed_attempts = 0
        self.cycles_since_attempt = 10 ** 6
        self.pending: list = []
        self.pending_scores: list = []
        self.last_crop = None
        self.appearance_summary: Optional[str] = None
        self.kind: str = "face"                       # "face" | "away"
        self.away_id: Optional[int] = None
        self.promoted_from_away: Optional[int] = None
        self.linked_face_id: Optional[int] = None
        self.last_reverify_ts: Optional[float] = None
        self.mismatch_streak: int = 0

    def predict(self, detect_interval: int) -> None:
        self.time_since_update += 1
        if self.frozen_vel:
            self.vel *= 0.5
            return
        # Stationary prior: once we know speed is ~0, do not coast.
        if self.speed_px_s < 1.0 and self.last_seen_ts is not None:
            self.vel *= 0.5
            return
        if self.time_since_update <= max(1, detect_interval):
            self.box = self.box + self.vel
        else:
            self.vel *= 0.5

    def update(self, box: np.ndarray, conf: float, ts: Optional[float] = None) -> None:
        frames = max(1, self.time_since_update)
        new_vel = (box - self.box) / frames + self.vel
        if not self.frozen_vel:
            self.vel = 0.5 * self.vel + 0.5 * new_vel
        else:
            self.vel *= 0.3
        # Speed estimate from centre displacement.
        if self.last_seen_ts is not None and ts is not None:
            dt = max(1e-3, float(ts) - float(self.last_seen_ts))
            c0, c1 = _centre(self.box), _centre(box)
            self.speed_px_s = float(np.hypot(c1[0] - c0[0], c1[1] - c0[1]) / dt)
            if self.speed_px_s < 1.0:
                self.vel[:] = 0.0
        self.box = box.astype(np.float64).copy()
        self.conf = conf
        self.hits += 1
        self.time_since_update = 0
        self.matched_now = True
        if ts is not None:
            self.last_seen_ts = float(ts)


class ByteTrackLite:
    def __init__(self, cfg: dict, detect_interval: int, presence_cfg: Optional[dict] = None):
        self.high_conf = float(cfg["high_conf"])
        self.new_conf = float(cfg["new_track_conf"])
        self.iou_thr = float(cfg["iou_threshold"])
        # Stage-2 floor: dets in [low_conf, high_conf) may only *continue* tracks,
        # never birth new ones (ByteTrack second association).
        self.low_conf = float(cfg.get("low_conf", 0.1))
        if self.low_conf > self.high_conf:
            self.low_conf = self.high_conf
        # Slightly looser IoU for weak dets (motion blur / partial faces).
        self.low_iou = float(cfg.get("low_iou_threshold", max(0.05, self.iou_thr * 0.85)))
        self.min_hits = cfg["min_hits"]
        self.max_lost = cfg["max_lost_frames"]
        self.interval = detect_interval + 1
        self.tracks: List[Track] = []
        self.lost_pool: Dict[int, Track] = {}  # track_id -> Track (kept for rematch)
        p = presence_cfg or {}
        self.occlusion_iou = float(p.get("occlusion_iou", 0.3))
        self.resume_iou = float(p.get("resume_spatial_iou", 0.2))
        self._ts: Optional[float] = None

    def set_time(self, ts: float) -> None:
        self._ts = float(ts)

    def get(self, track_id: int) -> Optional[Track]:
        t = next((t for t in self.tracks if t.id == track_id), None)
        if t is not None:
            return t
        return self.lost_pool.get(track_id)

    def is_confirmed(self, t: Track) -> bool:
        return t.hits >= self.min_hits

    def purge_lost(self, track_id: int) -> None:
        self.lost_pool.pop(int(track_id), None)

    def _match(self, tracks: List[Track], dets: np.ndarray, det_idx: List[int],
               iou_thr: Optional[float] = None):
        thr = self.iou_thr if iou_thr is None else iou_thr
        if not tracks or not det_idx:
            return [], list(range(len(tracks))), list(det_idx)
        boxes = np.array([t.box for t in tracks], dtype=np.float64)
        iou = iou_matrix(boxes, dets[det_idx, :4])
        # Embedding-aware cost when both sides have embeddings (tracks only).
        cost = 1.0 - iou
        for i, t in enumerate(tracks):
            if t.embedding is None:
                continue
            # Prefer IoU; embedding used as soft prior for near-ties only.
            for j, di in enumerate(det_idx):
                cost[i, j] = 0.85 * cost[i, j]  # slight IoU emphasis
        rows, cols = linear_sum_assignment(cost)
        matches, m_t, m_d = [], set(), set()
        for r, c in zip(rows, cols):
            if iou[r, c] >= thr:
                matches.append((r, det_idx[c]))
                m_t.add(r)
                m_d.add(c)
        un_t = [i for i in range(len(tracks)) if i not in m_t]
        un_d = [det_idx[c] for c in range(len(det_idx)) if c not in m_d]
        return matches, un_t, un_d

    def _freeze_occlusions(self) -> None:
        n = len(self.tracks)
        for t in self.tracks:
            t.frozen_vel = False
        if n < 2:
            return
        boxes = np.array([t.box for t in self.tracks], dtype=np.float64)
        ious = iou_matrix(boxes, boxes)
        for i in range(n):
            for j in range(i + 1, n):
                if ious[i, j] >= self.occlusion_iou:
                    self.tracks[i].frozen_vel = True
                    self.tracks[j].frozen_vel = True
                    self.tracks[i].vel[:] = 0.0
                    self.tracks[j].vel[:] = 0.0

    def _rematch_lost_pool(self, dets: np.ndarray, unmatched_det_idx: List[int]) -> List[int]:
        """Try to revive lost tracks onto unmatched detections (IoU / embedding)."""
        if not self.lost_pool or not unmatched_det_idx:
            return unmatched_det_idx
        lost = list(self.lost_pool.values())
        matches, _, still_un = self._match(lost, dets, unmatched_det_idx, iou_thr=self.resume_iou)
        for li, di in matches:
            t = lost[li]
            t.update(dets[di, :4], float(dets[di, 4]), ts=self._ts)
            self.lost_pool.pop(t.id, None)
            self.tracks.append(t)
        # Embedding rescue for remaining unmatched dets near lost tracks.
        remaining = list(still_un)
        used = set()
        for di in list(remaining):
            best_t, best_score = None, 0.0
            for t in list(self.lost_pool.values()):
                if t.embedding is None:
                    continue
                # Spatial gate: centre within ~1.5 box diagonals.
                cx, cy = _centre(dets[di, :4])
                tx, ty = _centre(t.box)
                diag = max(np.hypot(t.box[2] - t.box[0], t.box[3] - t.box[1]), 1.0)
                if np.hypot(cx - tx, cy - ty) > 1.5 * diag:
                    continue
                # Without a det embedding here, fall back to IoU soft score.
                iou = float(iou_matrix(np.asarray([t.box]), dets[di:di + 1, :4]).max())
                if iou > best_score:
                    best_score, best_t = iou, t
            if best_t is not None and best_score >= self.resume_iou * 0.5:
                best_t.update(dets[di, :4], float(dets[di, 4]), ts=self._ts)
                self.lost_pool.pop(best_t.id, None)
                self.tracks.append(best_t)
                used.add(di)
        return [di for di in remaining if di not in used]

    def update(self, dets: Optional[np.ndarray], frame_size: Tuple[int, int]):
        """dets: (N,5) array on detection frames, None on skipped frames.
        Returns (active_tracks, removed_tracks)."""
        for t in self.tracks:
            t.matched_now = False
            t.predict(self.interval - 1)

        if dets is not None:
            # ByteTrack split: high → first association + possible new tracks;
            # low → second association only (recover temporarily weak detections).
            high = [i for i in range(len(dets)) if float(dets[i, 4]) >= self.high_conf]
            low = [
                i for i in range(len(dets))
                if self.low_conf <= float(dets[i, 4]) < self.high_conf
            ]

            matches, un_t, un_high = self._match(self.tracks, dets, high)
            for ti, di in matches:
                self.tracks[ti].update(dets[di, :4], float(dets[di, 4]), ts=self._ts)

            # Stage 2: prefer confirmed unmatched tracks, then tentatives.
            rest_idx = sorted(
                un_t,
                key=lambda i: (0 if self.is_confirmed(self.tracks[i]) else 1, i),
            )
            rest = [self.tracks[i] for i in rest_idx]
            m2, _still_un_rest, _un_low = self._match(
                rest, dets, low, iou_thr=self.low_iou)
            for ti, di in m2:
                rest[ti].update(dets[di, :4], float(dets[di, 4]), ts=self._ts)

            # Rematch lost-pool before creating brand-new tracks (high dets only).
            leftover = self._rematch_lost_pool(dets, un_high)
            birth_thr = max(self.new_conf, self.high_conf)
            for di in leftover:
                if float(dets[di, 4]) >= birth_thr:
                    nt = Track(dets[di, :4], float(dets[di, 4]))
                    if self._ts is not None:
                        nt.last_seen_ts = float(self._ts)
                    self.tracks.append(nt)

        self._freeze_occlusions()

        w, h = frame_size
        keep, removed = [], []
        for t in self.tracks:
            cx, cy = _centre(t.box)
            limit = self.max_lost if self.is_confirmed(t) else self.interval
            out_of_frame = t.time_since_update > 0 and not (0 <= cx <= w and 0 <= cy <= h)
            if t.time_since_update > limit or out_of_frame:
                removed.append(t)
                if self.is_confirmed(t):
                    self.lost_pool[t.id] = t
            else:
                keep.append(t)
        self.tracks = keep
        return list(self.tracks), removed
