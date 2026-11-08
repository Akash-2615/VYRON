"""Core pipeline: detect -> track -> identify (embed/match/register) -> entry/exit events.

Also tracks people facing *away* from the camera (no usable face) via a person
detector, stores their clothing appearance, and keeps that KPI separate from
the unique face count.

Internal layout (single class, sectioned for review — no extra call overhead):
  1. Construction / timing
  2. Per-frame orchestration (``process_frame``)
  3. Away-walker / person-YOLO helpers
  4. Face identify / register / appearance
  5. Visit entry-exit / presence / reverify
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core import clock as _clock
from src.event_logging import EventLogger
from src.presence import (
    STATE_BODY_ONLY, STATE_HIDDEN, STATE_PRESENT, PresenceManager,
)
from src.recognition import Gallery, normalize
from src.storage import Database
from src.tracking import ByteTrackLite, Track, iou_matrix


def crop_box(frame: np.ndarray, box, margin: float) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    mw, mh = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = max(0, int(x1 - mw)), max(0, int(y1 - mh))
    x2, y2 = min(w, int(x2 + mw)), min(h, int(y2 + mh))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return frame[y1:y2, x1:x2].copy()


def _center_in_box(inner, outer, pad: float = 0.05) -> bool:
    ix1, iy1, ix2, iy2 = [float(v) for v in inner[:4]]
    ox1, oy1, ox2, oy2 = [float(v) for v in outer[:4]]
    cx, cy = (ix1 + ix2) / 2.0, (iy1 + iy2) / 2.0
    pw, ph = (ox2 - ox1) * pad, (oy2 - oy1) * pad
    return (ox1 - pw) <= cx <= (ox2 + pw) and (oy1 - ph) <= cy <= (oy2 + ph)


class FaceTrackingPipeline:
    # --- 1. Construction / timing -------------------------------------------------
    def __init__(self, cfg: dict, detector, embedder, db: Database, events: EventLogger,
                 appearance=None, person_detector=None):
        self.skip = cfg["detection_skip_frames"]
        self.rec = cfg["recognition"]
        self.quality = cfg["quality"]
        self.detector, self.embedder, self.db, self.events = detector, embedder, db, events
        self.appearance = appearance
        self.person_detector = person_detector
        self.away_cfg = cfg.get("away_walkers", {})
        self.away_enabled = bool(self.away_cfg.get("enabled", False)) and person_detector is not None
        self.appearance_cfg = cfg.get("appearance", {}) or {}
        tr_cfg = cfg.get("tracker", {}) or {}
        presence_cfg = cfg.get("presence", {}) or {}
        self.presence = PresenceManager(cfg)
        self.tracker = ByteTrackLite(tr_cfg, self.skip, presence_cfg)
        # Person YOLO may run every N face-detect cycles (detect_every_n). Unconfirmed
        # tracks die after ``interval`` frames — that interval must cover one full
        # person-detect period or AWAY never confirms / never writes to the DB.
        away_every = max(1, int(self.away_cfg.get("detect_every_n", 1)))
        self._away_detect_period = (self.skip + 1) * away_every
        self.away_tracker = ByteTrackLite(
            cfg.get("away_tracker", tr_cfg),
            detect_interval=max(0, self._away_detect_period - 1),
            presence_cfg=presence_cfg,
        )
        # Legacy soft-exit knobs kept for away walkers / backward-compat tests.
        self.reentry_grace_frames = int(tr_cfg.get("reentry_grace_frames", 45))
        self.reentry_grace_sec = float(tr_cfg.get("reentry_grace_sec",
                                                   presence_cfg.get("exit_grace_seconds", 10)))
        self.gallery = Gallery(self.rec["max_embeddings_per_face"])
        for fid, emb in db.load_embeddings():
            self.gallery.add(fid, emb)
        self.present: Dict[int, int] = {}              # face_id -> track_id (None-ish via missing = hidden)
        self.away_present: Dict[int, int] = {}
        self.pending_exit: Dict[int, dict] = {}        # legacy alias unused; kept for tests that poke it
        self.pending_away_exit: Dict[int, dict] = {}
        self._last_person_dets = None                  # cached YOLO person boxes for appearance
        self.face_as_away: Dict[int, int] = {}
        self.away_grace_frames = int(self.away_cfg.get(
            "reentry_grace_frames", tr_cfg.get("reentry_grace_frames", 75)))
        self.away_grace_sec = float(self.away_cfg.get(
            "reentry_grace_sec", presence_cfg.get("stationary_grace_seconds", 15.0)))
        self.frame_idx = -1
        self._clock: float = 0.0                       # mono/synthetic seconds for presence
        self._ts: datetime = _clock.wall_now()         # wall datetime for crop filenames
        self.fps: float = 25.0
        # Production default is wall-clock processing time for files and live.
        # Tests may set timestamps="video" and pass synthetic ts= for deterministic grace.
        self.time_mode: str = str(cfg.get("timestamps", "auto"))
        self._anchor_wall: Optional[datetime] = None   # wall time at clock anchor
        self._anchor_clock: float = 0.0                # mono/synthetic value at anchor
        self._synthetic_clock: bool = False            # True when ts= injected / video mode
        self._last_frame_wall: Optional[float] = None
        self._brightness_ema: Optional[float] = None
        self._active_before_loss: int = 0
        self._disturbance_cooldown_until: float = 0.0
        self._brightness_jump_thr: float = float(
            (cfg.get("presence") or {}).get("brightness_jump_threshold", 70.0))
        self._disturbance_min_sec: float = float(
            (cfg.get("presence") or {}).get("disturbance_min_seconds", 0.6))
        self._disturbance_cooldown_sec: float = float(
            (cfg.get("presence") or {}).get("disturbance_cooldown_seconds", 3.0))

    def set_timing(self, fps: float, live: bool = False) -> None:
        self.fps = max(1e-3, float(fps or 25.0))
        mode = self.time_mode
        if mode == "auto":
            # Wall-clock processing time for both file and live (no video-time mode).
            mode = "wall"
        self.time_mode = mode

    def _ensure_anchor(self, clock_val: float, *, synthetic: bool) -> None:
        if self._anchor_wall is None:
            self._anchor_wall = _clock.wall_now()
            self._anchor_clock = float(clock_val)
            self._synthetic_clock = synthetic

    def _wall_dt(self, clock: Optional[float] = None) -> datetime:
        """Convert an internal clock instant to wall datetime. Never adds mono to wall."""
        c = self._clock if clock is None else float(clock)
        if self._synthetic_clock or self.time_mode == "video":
            if self._anchor_wall is None:
                self._ensure_anchor(c, synthetic=True)
            return self._anchor_wall + timedelta(seconds=(c - self._anchor_clock))
        # Real monotonic → wall via (wall_now - (mono - instant)).
        return _clock.wall_from_mono(c)

    def _iso(self, clock: Optional[float] = None) -> str:
        return _clock.iso_ms(self._safe_wall(clock))

    def _safe_wall(self, clock: Optional[float] = None) -> datetime:
        """Wall datetime for storage; clamp anomalies >60s from wall_now()."""
        dt = self._wall_dt(clock)
        if _clock.is_anomalous(dt, max_skew_sec=60.0):
            fixed = _clock.wall_now()
            self.events.log(
                "TIMESTAMP_ANOMALY", logging.WARNING,
                raw=_clock.iso_ms(dt), used=_clock.iso_ms(fixed),
                skew_sec=f"{(dt - fixed).total_seconds():.3f}",
            )
            return fixed
        return dt

    def _advance_clock(self, frame_idx: int, ts: Optional[float]) -> float:
        if ts is not None:
            # Injected / test clock (may be synthetic video seconds).
            self._clock = float(ts)
            self._ensure_anchor(self._clock, synthetic=True)
        elif self.time_mode == "video":
            self._clock = float(frame_idx) / self.fps
            self._ensure_anchor(self._clock, synthetic=True)
        else:
            self._clock = _clock.mono()
            self._ensure_anchor(self._clock, synthetic=False)
        self._ts = self._wall_dt()
        return self._clock

    def notify_stream_gap(self, gap_seconds: float) -> None:
        """Called by the frame source after an RTSP reconnect / read stall."""
        gap = float(gap_seconds)
        if gap <= 0:
            return
        self.events.log("STREAM_GAP", gap_sec=f"{gap:.2f}")
        if gap >= self.presence.max_stream_gap:
            for v in list(self.presence.open_visits()):
                self._commit_exit(v.face_id, reason="stream_gap")
        else:
            self.presence.freeze_deadlines(self._clock, reason="stream_gap")
            self.events.log("DISTURBANCE_START", ts=self._iso(), reason="stream_gap",
                            gap_sec=f"{gap:.2f}")

    # ------------------------------------------------------------------ main step
    # --- 2. Per-frame orchestration ----------------------------------------------
    def process_frame(self, frame: np.ndarray, frame_idx: int,
                      ts: Optional[float] = None,
                      detect_frame: Optional[np.ndarray] = None,
                      detect_scale: float = 1.0) -> Tuple[List[Track], List[Track]]:
        """Process one frame.

        ``frame`` is always the original (crops, preview, away, disturbance).
        Optional ``detect_frame`` / ``detect_scale`` feed only the face detector;
        boxes are mapped back to original coordinates.
        """
        self.frame_idx = frame_idx
        clock = self._advance_clock(frame_idx, ts)
        self.tracker.set_time(clock)
        self.away_tracker.set_time(clock)
        run_det = frame_idx % (self.skip + 1) == 0
        det_in = detect_frame if detect_frame is not None else frame
        dets = self.detector.detect(det_in) if run_det else None
        if run_det and dets is not None and detect_scale and detect_scale != 1.0:
            from src.detection.preprocess import map_boxes_to_original
            dets = map_boxes_to_original(dets, float(detect_scale))
        if run_det and dets is not None:
            meta = getattr(self.detector, "last_meta", None)
            if isinstance(meta, dict) and meta.get("source") not in (None, "yolo"):
                self.events.log(
                    "FACE_RETINA_ASSIST",
                    frame=frame_idx,
                    source=meta.get("source"),
                    fallback=meta.get("fallback", 0),
                    verified=meta.get("verified", 0),
                    dropped_low=meta.get("dropped_low", 0),
                    n=len(dets),
                )
        h, w = frame.shape[:2]
        frame_size = (w, h)

        # Distant / seated faces: person YOLO finds the body, then re-scan the head ROI.
        person_dets_raw = None
        if run_det and self.away_enabled:
            every = max(1, int(self.away_cfg.get("detect_every_n", 1)))
            if self.frame_idx % ((self.skip + 1) * every) == 0:
                person_dets_raw = self.person_detector.detect(frame)
                if person_dets_raw is not None and len(person_dets_raw):
                    self._last_person_dets = person_dets_raw
                    roi_faces = self._faces_from_person_heads(frame, person_dets_raw, dets)
                    if roi_faces is not None and len(roi_faces):
                        dets = (np.vstack([dets, roi_faces]) if dets is not None and len(dets)
                                else roi_faces)
                        self.events.log("FACE_ROI_REFINE", frame=frame_idx, added=len(roi_faces))

        n_before = len([t for t in self.tracker.tracks if self.tracker.is_confirmed(t)])
        active, removed = self.tracker.update(dets, frame_size)
        self._detect_disturbance(frame, n_before, removed, clock)

        # Crowded frames: embed strongest unidentified faces. <=0 = no cap.
        max_id = int(self.rec.get("max_identify_per_frame", 6) or 0)
        id_candidates: List[Track] = []
        for t in active:
            t.kind = "face"
            if t.matched_now:
                t.last_crop = crop_box(frame, t.box, 0.10)
                if t.last_seen_ts is None:
                    t.last_seen_ts = clock
            if self.tracker.is_confirmed(t):
                if not t.announced:
                    t.announced = True
                    self.events.log("TRACK_STARTED", track=t.id, frame=frame_idx, conf=f"{t.conf:.2f}")
                # Lost-pool rematch: same track_id / face_id returns — reclaim the visit.
                if t.face_id is not None and not t.owns and t.matched_now:
                    self._reclaim_visit(t, frame)
                if t.face_id is None and t.matched_now and not t.rejected:
                    id_candidates.append(t)
                elif t.face_id is not None and t.owns and t.matched_now:
                    self._touch_visit(t, frame)
                    self._maybe_reverify(t, frame)
        # Prefer faces already mid-registration, then highest detector confidence.
        id_candidates.sort(
            key=lambda x: (len(getattr(x, "pending", []) or []), float(getattr(x, "conf", 0) or 0)),
            reverse=True,
        )
        batch = id_candidates if max_id <= 0 else id_candidates[:max_id]
        for t in batch:
            self._try_identify(t, frame)

        removed_list = list(removed)
        for t in removed_list:
            if not (t.owns and t.face_id is not None):
                self._on_track_removed(t, reason="lost", frame_size=frame_size)

        away_active: List[Track] = []
        if self.away_enabled:
            away_active = self._process_away(frame, run_det, active, person_dets_raw)
            for t in removed_list:
                if t.owns and t.face_id is not None:
                    if not self._continue_face_as_away(t, away_active, frame):
                        self._on_track_removed(t, reason="lost", frame_size=frame_size)
            away_active = self._promote_turned_faces(active, away_active, frame)
            self._body_only_keepalive(person_dets_raw, away_active, clock)
        else:
            for t in removed_list:
                if t.owns and t.face_id is not None:
                    self._on_track_removed(t, reason="lost", frame_size=frame_size)

        if self.presence.in_disturbance():
            started = self.presence.disturbance_until
            held = (float(clock) - float(started)) if started is not None else 0.0
            # Webcam auto-gain causes flicker — hold disturbance briefly or we log
            # START/END every frame and starve the API (Stop + Live Events hang).
            if held >= self._disturbance_min_sec and self._disturbance_cleared(active):
                self.presence.unfreeze_deadlines(clock)
                self._disturbance_cooldown_until = float(clock) + self._disturbance_cooldown_sec
                self.events.log(
                    "DISTURBANCE_END",
                    ts=self._iso(clock),
                    duration_sec=f"{held:.3f}",
                )

        self._flush_due_exits(force=False)
        self._flush_pending_away_exits(force=False)
        return active, away_active

    def in_frame_counts(self, face_tracks: List[Track], away_tracks: List[Track]) -> dict:
        """Real-time bodies currently visible (face + no-face, no double-count).

        Counts confirmed / identified tracks only — tentative YOLO flashes in
        crowded hallways were inflating the dashboard to 40+ "persons".
        """
        max_age = self.skip + 1
        face_n = 0
        face_confirmed = 0
        face_ids = set()
        for t in face_tracks:
            if int(getattr(t, "time_since_update", 0) or 0) > max_age:
                continue
            confirmed = self.tracker.is_confirmed(t)
            if not confirmed and t.face_id is None:
                continue
            face_n += 1
            if confirmed:
                face_confirmed += 1
            if t.face_id is not None:
                face_ids.add(int(t.face_id))
        away_n = 0
        for t in away_tracks:
            if int(getattr(t, "time_since_update", 0) or 0) > max_age:
                continue
            if not self.away_tracker.is_confirmed(t) and getattr(t, "away_id", None) is None:
                continue
            linked = getattr(t, "linked_face_id", None)
            if linked is not None and int(linked) in face_ids:
                continue  # same person already counted as face
            if linked is not None and int(linked) in self.face_as_away:
                away_n += 1
                continue
            away_n += 1
        return {
            "faces": face_n,
            "faces_confirmed": face_confirmed,
            "no_face": away_n,
            "persons": face_n + away_n,
            "faces_turned_away": len(self.face_as_away),
        }

    def _faces_from_person_heads(self, frame: np.ndarray, person_dets: np.ndarray,
                                existing: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Run face YOLO on upscaled head crops of people who have no face yet."""
        face_boxes = []
        if existing is not None and len(existing):
            face_boxes = [row[:4] for row in existing]
        extras = []
        h, w = frame.shape[:2]
        min_side = int(self.away_cfg.get("head_roi_min_px", 160))
        # Cap ROI passes — each is a full face-YOLO call; crowds otherwise stall the live UI.
        max_rois = max(0, int(self.away_cfg.get("max_head_roi_per_cycle", 4) or 4))
        ranked = sorted(
            [d for d in person_dets
             if not any(_center_in_box(fb, d, pad=0.08) for fb in face_boxes)],
            key=lambda d: float(d[4]) if len(d) > 4 else 0.0,
            reverse=True,
        )[:max_rois]
        for det in ranked:
            x1, y1, x2, y2 = [int(v) for v in det[:4]]
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            # Upper ~42% of the person box ≈ head/shoulders
            hy1 = max(0, y1 - int(0.05 * bh))
            hy2 = min(h, y1 + int(0.42 * bh))
            hx1 = max(0, x1 - int(0.08 * bw))
            hx2 = min(w, x2 + int(0.08 * bw))
            if hy2 - hy1 < 12 or hx2 - hx1 < 12:
                continue
            crop = frame[hy1:hy2, hx1:hx2]
            ch, cw = crop.shape[:2]
            scale = 1.0
            if max(ch, cw) < min_side:
                scale = float(min_side) / float(max(ch, cw))
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            try:
                local = self.detector.detect(crop)
            except Exception:  # noqa: BLE001
                continue
            if local is None or not len(local):
                continue
            for d in local:
                extras.append([
                    d[0] / scale + hx1, d[1] / scale + hy1,
                    d[2] / scale + hx1, d[3] / scale + hy1,
                    float(d[4]),
                ])
        if not extras:
            return None
        return np.asarray(extras, dtype=np.float32).reshape(-1, 5)

    def finalize(self) -> None:
        """Stream ended: close every still-open visit so each entry has exactly one exit."""
        for t in list(self.tracker.tracks):
            if t.owns and t.face_id is not None:
                self.present.pop(int(t.face_id), None)
            t.owns = False
        self.tracker.tracks = []
        self.tracker.lost_pool.clear()
        for fid in list(self.presence.visits.keys()):
            self._commit_exit(fid, reason="stream_end")
        for t in list(self.away_tracker.tracks):
            self._on_away_removed(t, reason="stream_end")
        self.away_tracker.tracks = []
        self._flush_pending_away_exits(force=True)

    # ------------------------------------------------------------------ away walkers
    # --- 3. Away-walker / person-YOLO helpers ------------------------------------
    def _process_away(self, frame: np.ndarray, run_det: bool, face_tracks: List[Track],
                      person_dets: Optional[np.ndarray] = None) -> List[Track]:
        h, w = frame.shape[:2]
        every = max(1, int(self.away_cfg.get("detect_every_n", 1)))
        run_person = run_det and (self.frame_idx % ((self.skip + 1) * every) == 0)
        if person_dets is None and run_person:
            person_dets = self.person_detector.detect(frame)
        elif not run_person:
            person_dets = None
        if person_dets is not None and len(person_dets):
            person_dets = self._filter_person_dets(person_dets, face_tracks, frame)
        active, removed = self.away_tracker.update(person_dets, (w, h))
        for t in active:
            t.kind = "away"
            if t.matched_now:
                t.last_crop = crop_box(frame, t.box, 0.05)
            if self.away_tracker.is_confirmed(t):
                if not t.announced:
                    t.announced = True
                    self.events.log("AWAY_TRACK_STARTED", track=t.id, frame=self.frame_idx,
                                    conf=f"{t.conf:.2f}")
                # Register once confirmed (do not wait for another matched frame — sparse person dets).
                if getattr(t, "away_id", None) is None and not t.rejected:
                    if t.last_crop is None:
                        t.last_crop = crop_box(frame, t.box, 0.05)
                    self._register_away(t, frame)
        for t in removed:
            self._on_away_removed(t, reason="lost")
        return active

    def _filter_person_dets(self, dets: np.ndarray, face_tracks: List[Track],
                            frame: np.ndarray) -> np.ndarray:
        """Keep person boxes that do not contain a face (crowd-safe: centre-in-box only)."""
        if not len(dets):
            return dets
        keep = []
        # Only confirmed faces — tentative face flashes should not wipe no-face people.
        face_boxes = [t.box for t in face_tracks if self.tracker.is_confirmed(t)]
        require_clip = bool(self.away_cfg.get("require_facing_away", False))
        for det in dets:
            # A face whose centre sits inside this person box → that person is a face visitor.
            if any(_center_in_box(fb, det, pad=0.05) for fb in face_boxes):
                continue
            if require_clip and self.appearance is not None:
                crop = crop_box(frame, det[:4], 0.02)
                if crop is not None:
                    try:
                        is_away, score = self.appearance.facing_away(crop)
                    except Exception:  # noqa: BLE001
                        is_away, score = True, 0.5
                    min_score = float(self.away_cfg.get("away_score_min", 0.35))
                    if not is_away or score < min_score:
                        continue
            keep.append(det)
        return np.asarray(keep, dtype=np.float32).reshape(-1, 5) if keep else np.zeros((0, 5), dtype=np.float32)

    def _iou_box(self, a, b) -> float:
        m = iou_matrix(np.asarray([a[:4]], dtype=np.float64),
                       np.asarray([b[:4]], dtype=np.float64))
        return float(m.max()) if m.size else 0.0

    def _same_body_region(self, a, b) -> float:
        """Similarity of two person boxes (IoU, or centre proximity for seated jitter)."""
        iou = self._iou_box(a, b)
        if iou >= 0.15:
            return max(iou, 0.5)
        ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
        bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
        acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
        bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        diag = max(np.hypot(ax2 - ax1, ay2 - ay1), np.hypot(bx2 - bx1, by2 - by1), 1.0)
        dist = np.hypot(acx - bcx, acy - bcy)
        if dist <= 0.35 * diag:
            return 0.45
        if dist <= 0.55 * diag:
            return 0.25
        return iou

    def _pending_away_resume_score(self, box, info: dict) -> float:
        """Resume score for a soft-exited away walker (walkers need looser spatial match)."""
        last = info.get("box")
        if last is None:
            return 0.0
        score = self._same_body_region(box, last)
        vel = info.get("vel")
        if vel is not None:
            try:
                dt = max(0, int(self.frame_idx) - int(info.get("frame_idx", self.frame_idx)))
                pred = np.asarray(last[:4], dtype=np.float64) + np.asarray(vel[:4], dtype=np.float64) * dt
                score = max(score, self._same_body_region(box, pred))
            except Exception:  # noqa: BLE001
                pass
        # Walking people often reappear ~1 body-length ahead after a face flash.
        ax1, ay1, ax2, ay2 = [float(v) for v in box[:4]]
        bx1, by1, bx2, by2 = [float(v) for v in last[:4]]
        acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
        bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        diag = max(np.hypot(ax2 - ax1, ay2 - ay1), np.hypot(bx2 - bx1, by2 - by1), 1.0)
        dist = float(np.hypot(acx - bcx, acy - bcy))
        if dist <= 1.25 * diag:
            score = max(score, 0.28)
        if dist <= 2.0 * diag:
            score = max(score, 0.22)
        return float(score)

    def _register_away(self, t: Track, frame: np.ndarray) -> None:
        # Another owned track already covering this body → same person still inside.
        for other in self.away_tracker.tracks:
            if other is t or not getattr(other, "owns", False):
                continue
            oid = getattr(other, "away_id", None)
            if oid is None:
                continue
            if self._same_body_region(t.box, other.box) >= 0.25:
                t.away_id = int(oid)
                t.owns = False
                t.appearance_summary = getattr(other, "appearance_summary", None)
                self.events.log("AWAY_DUPLICATE_TRACK", logging.WARNING,
                                away_id=int(oid), track=t.id, owner=other.id)
                return

        # Same seat / person after a brief drop → resume visit (still inside the frame).
        best_id, best_score = None, 0.0
        for aid, info in list(self.pending_away_exit.items()):
            score = self._pending_away_resume_score(t.box, info)
            if score > best_score:
                best_score, best_id = score, int(aid)
        if best_id is not None and best_score >= 0.22:
            info = self.pending_away_exit.pop(best_id)
            t.away_id = best_id
            t.owns = True
            self.away_present[best_id] = t.id
            t.appearance_summary = info.get("summary")
            linked = info.get("linked_face_id")
            if linked is not None:
                t.linked_face_id = int(linked)
            self.events.log("AWAY_RESUMED", away_id=best_id, track=t.id,
                            score=f"{best_score:.2f}", note="still_inside")
            return

        ts = self._ts.isoformat(timespec="milliseconds")
        away_id = self.db.add_away_visitor(ts)
        t.away_id = away_id
        t.owns = True
        self.away_present[away_id] = t.id
        crop = crop_box(frame, t.box, 0.05)
        path = self.events.save_crop("away", away_id, self._ts, crop, id_prefix="away")
        summary = None
        info = {"evidence_image": path} if path else {}
        want_tags = bool(self.away_cfg.get("describe_away", True))
        if self.appearance is not None and want_tags:
            try:
                tagged = self.appearance.describe(frame, person_box=t.box)
            except Exception:  # noqa: BLE001
                logging.exception("Away appearance failed for %s", away_id)
                tagged = None
            if tagged:
                info.update(tagged)
                if path:
                    info["evidence_image"] = path
                summary = tagged.get("summary")
                self.events.log("AWAY_APPEARANCE", away_id=away_id, summary=summary, image=path)
        if path or summary:
            self.db.set_away_appearance(away_id, summary or "", json.dumps(info) if info else "", path)
            t.appearance_summary = summary
        path_e = self.events.save_crop("away_entry", away_id, self._ts, crop, id_prefix="away")
        self.db.log_away_event(away_id, "entry", ts, path_e or path, t.id, self.frame_idx, note="no_face")
        self.events.log("AWAY_ENTRY", away_id=away_id, track=t.id, frame=self.frame_idx,
                        note="no_face", summary=summary, image=path_e or path)

    def _linked_face_for_away(self, away_id: int, track: Optional[Track] = None) -> Optional[int]:
        if track is not None:
            linked = getattr(track, "linked_face_id", None)
            if linked is not None:
                return int(linked)
        for fid, aid in self.face_as_away.items():
            if int(aid) == int(away_id):
                return int(fid)
        try:
            row = self.db.query(
                "SELECT linked_face_id FROM away_visitors WHERE away_id=?", (int(away_id),))
            if row and row[0][0] is not None:
                return int(row[0][0])
        except Exception:  # noqa: BLE001
            pass
        return None

    def _close_face_continued_as_away(self, face_id: int, away_id: int, crop, note: str,
                                      track_id: Optional[int] = None,
                                      frame_idx: Optional[int] = None,
                                      ts: Optional[datetime] = None) -> None:
        """Away body finally left the frame → close the original face visit too."""
        self.face_as_away.pop(int(face_id), None)
        self.present.pop(int(face_id), None)
        if int(face_id) in self.presence.visits:
            self._commit_exit(int(face_id), reason=note or f"left_as_away:{away_id}")
            return
        when = ts or self._ts
        fidx = self.frame_idx if frame_idx is None else frame_idx
        path = self.events.save_crop("exit", face_id, when, crop)
        self.db.log_event(
            face_id, "exit", when.isoformat(timespec="milliseconds"),
            path, track_id, fidx, note or f"left_as_away:{away_id}")
        self.events.log("EXIT", face_id=face_id, track=track_id, frame=fidx,
                        note=note or f"left_as_away:{away_id}", image=path,
                        away_id=away_id)

    def _on_away_removed(self, t: Track, reason: str,
                         promoted_by_face_track: Optional[int] = None) -> None:
        away_id = getattr(t, "away_id", None)
        self.events.log("AWAY_TRACK_ENDED", track=t.id, away_id=away_id, reason=reason)
        if t.owns and away_id is not None:
            self.away_present.pop(away_id, None)
            linked = self._linked_face_for_away(int(away_id), t)
            if reason in ("stream_end", "turned_to_face"):
                path = self.events.save_crop("away_exit", away_id, self._ts, t.last_crop, id_prefix="away")
                self.db.log_away_event(away_id, "exit", self._ts.isoformat(timespec="milliseconds"),
                                       path, t.id, self.frame_idx, note=reason)
                self.events.log("AWAY_EXIT", away_id=away_id, track=t.id, frame=self.frame_idx,
                                note=reason, image=path)
                self.pending_away_exit.pop(int(away_id), None)
                # Stream ended while following a turned face → close that face visit once.
                if reason == "stream_end" and linked is not None and linked in self.face_as_away:
                    self._close_face_continued_as_away(
                        linked, int(away_id), t.last_crop, "stream_end", t.id)
            else:
                # Defer exit — sitting / briefly unmatched / face-flash people stay resumable.
                vel = getattr(t, "vel", None)
                self.pending_away_exit[int(away_id)] = {
                    "away_id": int(away_id),
                    "box": np.asarray(t.box[:4], dtype=np.float64).copy(),
                    "vel": (np.asarray(vel, dtype=np.float64).copy()
                            if vel is not None else np.zeros(4, dtype=np.float64)),
                    "crop": t.last_crop,
                    "track_id": t.id,
                    "frame_idx": self.frame_idx,
                    "ts": self._ts,
                    "note": reason,
                    "summary": getattr(t, "appearance_summary", None),
                    "linked_face_id": linked,
                    # Only the face track that triggered awaiting_face_id may claim later.
                    "promoted_by_face_track": (
                        int(promoted_by_face_track)
                        if promoted_by_face_track is not None else None),
                }
                self.events.log("AWAY_EXIT_PENDING", away_id=away_id, track=t.id,
                                grace_frames=self.away_grace_frames, reason=reason)
            t.owns = False

    def _flush_pending_away_exits(self, force: bool = False) -> None:
        if not self.pending_away_exit:
            return
        done = []
        for away_id, info in list(self.pending_away_exit.items()):
            age_frames = self.frame_idx - int(info["frame_idx"])
            age_sec = (self._ts - info["ts"]).total_seconds()
            # Both frame + time grace (frame-only was marking seated people Left too soon).
            if not force and (age_frames < self.away_grace_frames
                              or age_sec < self.away_grace_sec):
                continue
            when = self._ts
            path = self.events.save_crop(
                "away_exit", away_id, when, info.get("crop"), id_prefix="away")
            note = info.get("note") or "lost"
            self.db.log_away_event(
                away_id, "exit", when.isoformat(timespec="milliseconds"),
                path, info.get("track_id"), self.frame_idx, note)
            self.events.log("AWAY_EXIT", away_id=away_id, track=info.get("track_id"),
                            frame=self.frame_idx, note=note, image=path)
            linked = info.get("linked_face_id")
            if linked is None:
                linked = self._linked_face_for_away(int(away_id))
            if linked is not None and int(linked) in self.face_as_away:
                self._close_face_continued_as_away(
                    int(linked), int(away_id), info.get("crop"),
                    f"left_as_away:{away_id}", info.get("track_id"),
                    info.get("frame_idx"), info.get("ts"))
            done.append(away_id)
        for away_id in done:
            self.pending_away_exit.pop(away_id, None)

    def _face_matches_away(self, face: Track, away: Track, frame: np.ndarray) -> float:
        """Score >0 only when the face centre sits in this person's *head* region.

        Crowds previously false-linked random nearby face IDs via loose IoU.
        Require the face centre inside the upper 55% of the person box.
        """
        ax1, ay1, ax2, ay2 = [float(v) for v in away.box[:4]]
        fx1, fy1, fx2, fy2 = [float(v) for v in face.box[:4]]
        fcx, fcy = (fx1 + fx2) * 0.5, (fy1 + fy2) * 0.5
        pw = (ax2 - ax1) * 0.08
        ph = (ay2 - ay1) * 0.05
        head_bottom = ay1 + 0.55 * max(ay2 - ay1, 1.0)
        if not ((ax1 - pw) <= fcx <= (ax2 + pw) and (ay1 - ph) <= fcy <= (head_bottom + ph)):
            return 0.0
        # Tight IoU of face vs person head band as confidence.
        hx1, hy1, hx2, hy2 = ax1, ay1, ax2, head_bottom
        ix1, iy1 = max(fx1, hx1), max(fy1, hy1)
        ix2, iy2 = min(fx2, hx2), min(fy2, hy2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_f = max(1.0, (fx2 - fx1) * (fy2 - fy1))
        # Face mostly inside head band → high score
        frac = inter / area_f
        return float(max(0.35, min(1.0, frac)))

    def _continue_face_as_away(self, face_t: Track, away_tracks: List[Track],
                               frame: np.ndarray) -> bool:
        """Face lost but body still visible → keep same visit, follow as no-face."""
        if face_t.face_id is None or not face_t.owns:
            return False
        from src.recognition.appearance import body_box
        body = body_box(face_t.box, frame.shape) or face_t.box
        best: Optional[Track] = None
        best_score = 0.0
        second = 0.0
        for at in away_tracks:
            # Face centre must still sit on this body (not a neighbour).
            if not _center_in_box(face_t.box, at.box, pad=0.12):
                continue
            score = self._same_body_region(body, at.box)
            if score > best_score:
                second = best_score
                best_score, best = score, at
            elif score > second:
                second = score
        # Strong unique match — avoid handing face ID 1 to a random passer-by.
        if best is None or best_score < 0.35 or (best_score - second) < 0.08:
            return False
        if getattr(best, "away_id", None) is None:
            self._register_away(best, frame)
        away_id = getattr(best, "away_id", None)
        if away_id is None:
            return False
        fid = int(face_t.face_id)
        self._link_away_face(int(away_id), fid)
        best.linked_face_id = fid
        self.face_as_away[fid] = int(away_id)
        self.present.pop(fid, None)
        # Keep the face visit alive as BODY_ONLY while they face away.
        if fid in self.presence.visits:
            self.presence.body_only(fid, self._clock, best.box)
        face_t.owns = False
        self.events.log("FACE_CONTINUED_AS_AWAY", face_id=fid, away_id=int(away_id),
                        score=f"{best_score:.2f}", note="turned_still_inside")
        return True

    def _promote_turned_faces(self, face_tracks: List[Track], away_tracks: List[Track],
                              frame: np.ndarray) -> List[Track]:
        """When a no-face walker turns and a face is detected on them, hand off to face monitoring."""
        if not face_tracks or not away_tracks:
            return away_tracks
        remaining: List[Track] = []
        for at in away_tracks:
            away_id = getattr(at, "away_id", None)
            if away_id is None:
                remaining.append(at)
                continue
            ranked: List[Tuple[float, Track]] = []
            for ft in face_tracks:
                score = self._face_matches_away(ft, at, frame)
                if score > 0:
                    ranked.append((score, ft))
            ranked.sort(key=lambda x: x[0], reverse=True)
            if not ranked or ranked[0][0] < 0.45:
                remaining.append(at)
                continue
            best_score, best_ft = ranked[0]
            second = ranked[1][0] if len(ranked) > 1 else 0.0
            if (best_score - second) < 0.12:
                # Ambiguous in a crowd — do not guess an identity.
                remaining.append(at)
                continue

            prior_linked = self._linked_face_for_away(int(away_id), at)
            ft_fid = int(best_ft.face_id) if best_ft.face_id is not None else None
            # Only reuse an existing face_id if this away was already that person
            # (turned away earlier) — never steal a neighbour's registered ID.
            can_reuse = False
            if ft_fid is not None:
                if prior_linked is not None and int(prior_linked) == ft_fid:
                    can_reuse = True
                elif self.face_as_away.get(ft_fid) == int(away_id):
                    can_reuse = True
                else:
                    # Known face on this body but never linked → likely wrong person nearby.
                    remaining.append(at)
                    continue
            elif prior_linked is not None:
                # Unidentified face track on the same body → restore known ID.
                best_ft.face_id = int(prior_linked)
                ft_fid = int(prior_linked)
                can_reuse = True

            best_ft.promoted_from_away = int(away_id)
            if ft_fid is not None and can_reuse:
                # Real handoff to a known face visit — permanent away exit.
                self.events.log(
                    "AWAY_PROMOTED_TO_FACE",
                    away_id=int(away_id),
                    face_track=best_ft.id,
                    face_id=best_ft.face_id,
                    note="turned_to_face",
                    score=f"{best_score:.2f}",
                )
                self._link_away_face(int(away_id), int(ft_fid))
                self.face_as_away.pop(int(ft_fid), None)
                self._on_away_removed(at, reason="turned_to_face")
                if at in self.away_tracker.tracks:
                    self.away_tracker.tracks.remove(at)
                fid = int(best_ft.face_id)
                best_ft.owns = True
                self.present[fid] = best_ft.id
                resumed = self.presence.mark_present(
                    fid, self._clock, best_ft.id, best_ft.box, crop=best_ft.last_crop)
                if resumed is not None:
                    self.events.log("VISIT_RESUMED", face_id=fid, track=best_ft.id,
                                    away_id=int(away_id), note="away_promoted_to_face",
                                    occluded_sec=f"{resumed.occluded_seconds:.2f}")
                elif fid not in self.presence.visits:
                    self.events.log("VISIT_RESUMED", face_id=fid, track=best_ft.id,
                                    away_id=int(away_id), note="away_promoted_to_face")
            else:
                # Unidentified face flash must NOT mint a new away_id when the body
                # reappears — soft-park so `_register_away` can AWAY_RESUME.
                # Remember which face track owns this promote so a neighbour who
                # already has a face_id cannot steal the link later.
                self.events.log(
                    "AWAY_PROMOTED_TO_FACE",
                    away_id=int(away_id),
                    face_track=best_ft.id,
                    face_id=best_ft.face_id,
                    note="awaiting_face_id",
                    score=f"{best_score:.2f}",
                )
                self._on_away_removed(
                    at, reason="awaiting_face_id",
                    promoted_by_face_track=int(best_ft.id))
                if at in self.away_tracker.tracks:
                    self.away_tracker.tracks.remove(at)
        return remaining

    def _link_away_face(self, away_id: int, face_id: int) -> None:
        self.db.set_away_linked_face(away_id, face_id)
        self.events.log("AWAY_FACE_LINK", away_id=away_id, face_id=face_id, note="turned_to_face")

    def _finalize_away_after_face_id(self, away_id: int, face_id: int, face_t: Track) -> None:
        """Face ID landed after a soft-parked promote — link and close the away visit once."""
        self._link_away_face(int(away_id), int(face_id))
        self.face_as_away.pop(int(face_id), None)
        info = self.pending_away_exit.pop(int(away_id), None)
        if info is None:
            # Body may have AWAY_RESUMEd meanwhile — close the live away track.
            for at in list(self.away_tracker.tracks):
                if int(getattr(at, "away_id", -1) or -1) != int(away_id):
                    continue
                if not getattr(at, "owns", False):
                    continue
                self._on_away_removed(at, reason="turned_to_face")
                if at in self.away_tracker.tracks:
                    self.away_tracker.tracks.remove(at)
                return
            return
        crop = info.get("crop")
        if crop is None:
            crop = face_t.last_crop
        path = self.events.save_crop(
            "away_exit", away_id, self._ts, crop, id_prefix="away")
        self.db.log_away_event(
            away_id, "exit", self._ts.isoformat(timespec="milliseconds"),
            path, info.get("track_id"), self.frame_idx, note="turned_to_face")
        self.events.log("AWAY_EXIT", away_id=away_id, track=info.get("track_id"),
                        frame=self.frame_idx, note="turned_to_face", image=path,
                        face_id=face_id)

    def _claim_pending_away_for_face(self, face_t: Track, face_id: int,
                                     frame_shape) -> None:
        """Link a soft-parked away ONLY when this face is clearly that same body.

        Crowds previously false-linked a neighbour's registered face_id onto an
        unrelated no-face walker via a loose spatial score (≥0.28 within ~1 body).
        Rules now:
          1. If the soft-park recorded ``promoted_by_face_track``, only that face
             track may claim (the one that actually sat in the head region).
          2. Otherwise require the face centre in the away head band AND a strong
             score — never claim merely because someone else is nearby.
          3. Never overwrite an existing linked_face_id with a different face.
        """
        if not self.pending_away_exit:
            return
        # Preferred path: this track already owns the soft-park from promote.
        owned = [
            (int(aid), info) for aid, info in self.pending_away_exit.items()
            if info.get("promoted_by_face_track") is not None
            and int(info["promoted_by_face_track"]) == int(face_t.id)
        ]
        if owned:
            aid, info = owned[0]
            prior = info.get("linked_face_id")
            if prior is not None and int(prior) != int(face_id):
                return
            face_t.promoted_from_away = aid
            self._finalize_away_after_face_id(aid, face_id, face_t)
            return

        best_id, best_score = None, 0.0
        for aid, info in list(self.pending_away_exit.items()):
            # Reserved for another face track — do not steal.
            owner = info.get("promoted_by_face_track")
            if owner is not None and int(owner) != int(face_t.id):
                continue
            prior = info.get("linked_face_id")
            if prior is not None and int(prior) != int(face_id):
                continue
            away_box = info.get("box")
            if away_box is None:
                continue
            # Reuse the same head-band gate as promote (not loose body proximity).
            fake_away = Track(np.asarray(away_box[:4], dtype=np.float64), 0.9)
            score = self._face_matches_away(face_t, fake_away, np.zeros((1, 1, 3), dtype=np.uint8))
            if score > best_score:
                best_score, best_id = score, int(aid)
        # Strong head-band overlap only — 0.55 ≈ face mostly inside head region.
        if best_id is not None and best_score >= 0.55:
            face_t.promoted_from_away = best_id
            self._finalize_away_after_face_id(best_id, face_id, face_t)

    # ------------------------------------------------------------------ identification
    # --- 4. Face identify / register / appearance --------------------------------
    def _quality_ok(self, box, crop) -> Optional[str]:
        x1, y1, x2, y2 = box
        if min(x2 - x1, y2 - y1) < self.quality["min_face_px"]:
            return "too_small"
        if crop is None:
            return "empty_crop"
        gray = cv2.cvtColor(cv2.resize(crop, (112, 112)), cv2.COLOR_BGR2GRAY)
        # Optional contrast assist for the blur gate only (never changes embed crop).
        pp = getattr(self, "_preprocess_settings", None)
        if pp is not None and getattr(pp, "quality_assist", False):
            try:
                from src.detection.preprocess import quality_assist_gray
                gray = quality_assist_gray(gray)
            except Exception:  # noqa: BLE001
                pass
        if cv2.Laplacian(gray, cv2.CV_64F).var() < self.quality["min_blur_var"]:
            return "blurry"
        return None

    def _try_identify(self, t: Track, frame: np.ndarray) -> None:
        if t.embed_attempts >= self.rec["max_embed_attempts"]:
            t.rejected = True
            self.events.log("TRACK_REJECTED", logging.WARNING, track=t.id, reason="no_usable_embedding")
            return
        if t.cycles_since_attempt < self.rec["embed_retry_interval"]:
            t.cycles_since_attempt += 1
            return
        crop = crop_box(frame, t.box, 0.25)
        reason = self._quality_ok(t.box, crop)
        if reason:
            t.cycles_since_attempt = self.rec["embed_retry_interval"]
            return
        t.embed_attempts += 1
        t.cycles_since_attempt = 0
        res = self.embedder.embed(crop)
        if res is None:
            self.events.log("EMBED_NO_FACE", track=t.id, attempt=t.embed_attempts)
            return
        emb, score = res
        self.events.log("EMBEDDING_GENERATED", track=t.id, det_score=f"{score:.2f}", attempt=t.embed_attempts)

        thr = float(self.rec["match_threshold"])
        soft = float(self.rec.get("soft_match_margin", 0.10))
        min_gap = float(self.rec.get("min_match_gap", 0.05))
        min_reg = float(self.rec.get("register_min_det_score", 0.40))

        anti_split = float(self.rec.get("anti_split_threshold", max(0.45, thr - 0.08)))

        def _claim_existing(emb_vec: np.ndarray, note: str = "") -> bool:
            """Prefer rematch over minting twins (RTSP lighting often dips under thr)."""
            fid, sim, gap = self.gallery.match_with_gap(emb_vec)
            if fid is None:
                return False
            # Hard match: strong sim + clear margin vs other gallery faces.
            if sim >= thr and gap >= min_gap:
                self._identified(t, fid, sim, emb_vec, frame, is_new=False)
                return True
            # Soft match: near-threshold with enough gap.
            if soft > 0 and sim >= thr - soft and gap >= min_gap:
                self.events.log(
                    "FACE_SOFT_MATCH", face_id=fid, track=t.id,
                    similarity=f"{sim:.3f}", threshold=f"{thr:.2f}",
                    gap=f"{gap:.3f}", note=note or "",
                )
                self._identified(t, fid, sim, emb_vec, frame, is_new=False)
                return True
            # Strong sim but two gallery faces look similar — still attach to best
            # rather than minting a third twin ID.
            if sim >= thr and gap < min_gap:
                self.events.log(
                    "FACE_MATCH_AMBIGUOUS", face_id=fid, track=t.id,
                    similarity=f"{sim:.3f}", gap=f"{gap:.3f}", min_gap=f"{min_gap:.2f}",
                    note=note or "claim_best",
                )
                self._identified(t, fid, sim, emb_vec, frame, is_new=False)
                return True
            return False

        if _claim_existing(emb):
            return

        t.pending.append(emb)
        t.pending_scores.append(float(score))
        if len(t.pending) < self.rec["min_embeds_to_register"]:
            return
        mean = normalize(np.mean(t.pending, axis=0))
        if _claim_existing(mean, note="mean"):
            return
        # Weak InsightFace det scores (blur / profile) mint junk identities — wait.
        best_score = max(t.pending_scores) if t.pending_scores else float(score)
        if best_score < min_reg:
            max_pending = max(int(self.rec["min_embeds_to_register"]) * 3, 6)
            if len(t.pending) > max_pending:
                keep = int(self.rec["min_embeds_to_register"])
                t.pending = t.pending[-keep:]
                t.pending_scores = t.pending_scores[-keep:]
            return
        # Anti-split: RTSP / angle change often lands just under thr against an
        # existing face (e.g. V0005). Never mint a twin when gallery is that close.
        fid_near, sim_near, gap_near = self.gallery.match_with_gap(mean)
        if fid_near is not None and sim_near >= anti_split:
            self.events.log(
                "FACE_ANTI_SPLIT", face_id=fid_near, track=t.id,
                similarity=f"{sim_near:.3f}", gap=f"{gap_near:.3f}",
                anti_split=f"{anti_split:.2f}", threshold=f"{thr:.2f}",
            )
            self._identified(t, int(fid_near), float(sim_near), mean, frame, is_new=False)
            return
        new_id = self.db.add_face(self._ts.isoformat(timespec="milliseconds"))
        self.gallery.add(new_id, mean)
        crop_path = self._save_embedding(new_id, mean, frame, t, primary=True)
        self.events.log("FACE_REGISTERED", face_id=new_id, track=t.id,
                        unique_total=self.db.unique_count(), closeup=crop_path)
        self._identified(t, new_id, 1.0, mean, frame, is_new=True)

    def _identified(self, t: Track, face_id: int, sim: float, emb: np.ndarray,
                    frame: np.ndarray, is_new: bool) -> None:
        t.face_id = face_id
        t.pending = []
        t.pending_scores = []
        if not is_new:
            self.events.log("FACE_RECOGNIZED", face_id=face_id, track=t.id, similarity=f"{sim:.3f}")
            if sim < 0.75 and self.gallery.add(face_id, emb):
                self._save_embedding(face_id, emb, frame, t)

        existing = self.db.get_appearance_summary(face_id)
        want = bool(self.appearance_cfg.get("describe_on_register", True))
        # Fill once when empty (new face or older row missing tags). Never overwrite
        # an existing summary — re-id can grab a neighbour's body crop.
        if want and not existing:
            t.appearance_summary = self._describe_appearance(face_id, frame, t.box)
        else:
            t.appearance_summary = existing

        # No-face walker who turned: link away → face. Never let claim errors skip ENTRY.
        note_extra = None
        try:
            promoted = getattr(t, "promoted_from_away", None)
            if promoted is not None:
                self._finalize_away_after_face_id(int(promoted), int(face_id), t)
                note_extra = "from_away_turn"
            else:
                self._claim_pending_away_for_face(t, int(face_id), frame.shape)
                if getattr(t, "promoted_from_away", None) is not None:
                    note_extra = "from_away_turn"
        except Exception:  # noqa: BLE001
            logging.exception(
                "Away→face finalize failed for face_id=%s track=%s", face_id, t.id)

        t.embedding = np.asarray(emb, dtype=np.float32)

        # Resume paths only apply to faces that already had a visit — a brand-new
        # FACE_REGISTERED must always get an ENTRY (unique and entries stay in sync).
        if not is_new and face_id in self.presence.visits and self.presence.visits[face_id].state in (
                STATE_HIDDEN, STATE_BODY_ONLY):
            t.owns = True
            self.present[face_id] = t.id
            resumed = self.presence.mark_present(
                face_id, self._clock, t.id, t.box, crop=crop_box(frame, t.box, 0.10), emb=emb)
            if resumed is not None:
                self.events.log("VISIT_RESUMED", face_id=face_id, track=t.id,
                                note="gap_under_grace",
                                occluded_sec=f"{resumed.occluded_seconds:.2f}",
                                resumes=resumed.resume_count)
                if resumed.visit_id is not None:
                    self.db.update_visit_progress(
                        resumed.visit_id, resumed.visible_seconds,
                        resumed.occluded_seconds, resumed.resume_count)
            return

        # Was followed as no-face after turning; face visible again — same visit.
        if not is_new and face_id in self.face_as_away:
            away_id = self.face_as_away.pop(face_id)
            self.pending_away_exit.pop(int(away_id), None)
            t.owns = True
            self.present[face_id] = t.id
            if face_id in self.presence.visits:
                self.presence.mark_present(
                    face_id, self._clock, t.id, t.box, crop=t.last_crop, emb=emb)
            self.events.log("VISIT_RESUMED", face_id=face_id, track=t.id,
                            away_id=away_id, note="turned_back_to_face")
            return

        # New identity must not inherit a stale face_as_away / present slot.
        if is_new:
            self.face_as_away.pop(int(face_id), None)

        owner_id = self.present.get(face_id)
        owner = self.tracker.get(owner_id) if owner_id is not None else None
        open_visit = self.presence.visits.get(face_id)
        if owner is None and (open_visit is None or open_visit.state != STATE_PRESENT):
            t.owns = True
            self.present[face_id] = t.id
            if is_new:
                entry_note = "new" if not note_extra else f"new;{note_extra}"
            else:
                entry_note = "re-entry" if not note_extra else f"re-entry;{note_extra}"
            self._emit_entry(t, crop_box(frame, t.box, 0.10), note=entry_note, emb=emb)
        elif owner is not None and owner.id != t.id and owner.time_since_update > 0:
            owner.owns = False
            t.owns = True
            self.present[face_id] = t.id
            if open_visit is not None:
                open_visit.track_id = t.id
            self.events.log("TRACK_HANDOVER", face_id=face_id, old_track=owner.id, new_track=t.id)
        elif owner is not None and owner.id == t.id:
            t.owns = True
            self._touch_visit(t, frame)
        elif open_visit is not None and open_visit.state == STATE_PRESENT and owner is None:
            # Owner track id stale after rematch — reclaim.
            t.owns = True
            self.present[face_id] = t.id
            self.presence.mark_present(face_id, self._clock, t.id, t.box, emb=emb)
            self.events.log("TRACK_HANDOVER", face_id=face_id, old_track=-1, new_track=t.id)
        else:
            t.owns = False
            self.events.log("DUPLICATE_TRACK", logging.WARNING, face_id=face_id, track=t.id,
                            owner=owner.id if owner else None)

    def _describe_appearance(self, face_id: int, frame: np.ndarray, box) -> Optional[str]:
        """Save a full-person evidence crop; tag clothing when OpenCLIP is loaded."""
        from src.recognition.appearance import body_crop, person_box_for_face
        person_dets = self._last_person_dets
        pbox = person_box_for_face(box, frame.shape, person_dets=person_dets)
        crop = body_crop(frame, box, person_dets=person_dets)
        path = self.events.save_crop("appearance", face_id, self._ts, crop) if crop is not None else None
        info: Optional[dict] = None
        if self.appearance is not None:
            try:
                info = self.appearance.describe(frame, face_box=box, person_box=pbox)
            except Exception:  # noqa: BLE001
                logging.exception("Appearance describe failed for face %s", face_id)
                info = None
        if info is None and path:
            info = {"evidence_image": path, "summary": "person (body crop)"}
        elif info is not None:
            if path:
                info["evidence_image"] = path
        if not info:
            return None
        summary = info.get("summary") or "person"
        self.db.set_appearance(face_id, summary, json.dumps(info), path)
        self.events.log("APPEARANCE_DESCRIBED", face_id=face_id, summary=summary, image=path)
        return summary

    def _save_embedding(self, face_id: int, emb: np.ndarray, frame: np.ndarray, t: Track,
                        primary: bool = False) -> Optional[str]:
        path = self.events.save_crop("face", face_id, self._ts, crop_box(frame, t.box, 0.05))
        self.db.add_embedding(face_id, emb, self._ts.isoformat(timespec="milliseconds"), path)
        if primary and path:
            self.db.set_face_image(face_id, path)
        return path

    # --- 5. Visit entry-exit / presence / reverify -------------------------------
    def _touch_visit(self, t: Track, frame: np.ndarray) -> None:
        if t.face_id is None:
            return
        v = self.presence.visits.get(int(t.face_id))
        if v is None:
            return
        crop = t.last_crop if t.last_crop is not None else crop_box(frame, t.box, 0.10)
        self.presence.mark_present(
            int(t.face_id), self._clock, t.id, t.box, crop=crop, emb=t.embedding)

    def _reclaim_visit(self, t: Track, frame: np.ndarray) -> None:
        """Owning track rematched from the lost-pool — resume the same visit."""
        fid = int(t.face_id)
        v = self.presence.visits.get(fid)
        if v is None or v.state == "EXITED":
            t.face_id = None  # force re-identify as a fresh visit later
            return
        t.owns = True
        self.present[fid] = t.id
        crop = t.last_crop if t.last_crop is not None else crop_box(frame, t.box, 0.10)
        resumed = self.presence.mark_present(fid, self._clock, t.id, t.box, crop=crop, emb=t.embedding)
        if resumed is not None:
            self.events.log("VISIT_RESUMED", face_id=fid, track=t.id,
                            note="lost_pool_rematch",
                            occluded_sec=f"{resumed.occluded_seconds:.2f}",
                            resumes=resumed.resume_count)
            if resumed.visit_id is not None:
                self.db.update_visit_progress(
                    resumed.visit_id, resumed.visible_seconds,
                    resumed.occluded_seconds, resumed.resume_count)

    def _emit_entry(self, t: Track, crop: Optional[np.ndarray], note: str = "",
                    emb=None) -> None:
        entry_clock = self._clock
        when = self._safe_wall(entry_clock)
        iso = _clock.iso_ms(when)
        visit_id = self.db.start_visit(int(t.face_id), iso)
        path = self.events.save_crop("entry", t.face_id, when, crop)
        self.db.log_event(t.face_id, "entry", iso, path, t.id, self.frame_idx, note,
                          visit_id=visit_id)
        # First visit for this face_id → new; later visits after a closed EXIT → re-entry.
        note_s = (note or "").strip()
        entry_kind = "re_entry" if note_s.startswith("re-entry") else "new"
        self.events.log("ENTRY", ts=iso, visit_id=visit_id, face_id=t.face_id,
                        track=t.id, image=path, frame=self.frame_idx,
                        entry_kind=entry_kind, note=note_s or None)
        # Explicit companion lines for the mandatory “new entry / re-entry” audit trail.
        kind_event = "RE_ENTRY" if entry_kind == "re_entry" else "NEW_ENTRY"
        self.events.log(kind_event, ts=iso, visit_id=visit_id, face_id=t.face_id,
                        track=t.id, image=path, frame=self.frame_idx, entry_kind=entry_kind)
        self.presence.start_visit(
            int(t.face_id), entry_clock, t.id, t.box, crop=crop, emb=emb, visit_id=visit_id)

    def _on_track_removed(self, t: Track, reason: str,
                          frame_size: Tuple[int, int] = (640, 480)) -> None:
        self.events.log("TRACK_ENDED", track=t.id, face_id=t.face_id, reason=reason)
        self.events.log("TRACK_LOST", track=t.id, face_id=t.face_id, reason=reason)
        if t.owns and t.face_id is not None:
            fid = int(t.face_id)
            self.present.pop(fid, None)
            if reason == "stream_end":
                self._commit_exit(fid, reason="stream_end")
            else:
                occluded = self._box_occluded_by_others(t.box, exclude_track=t.id)
                v = self.presence.hide(fid, self._clock, frame_size,
                                       occluded_by=occluded, reason=reason)
                if v is not None:
                    if v.last_crop is None:
                        v.last_crop = t.last_crop
                    if v.last_box is None:
                        v.last_box = np.asarray(t.box[:4], dtype=np.float64).copy()
                    deadline_in = (
                        max(0.0, float(v.deadline_ts) - float(self._clock))
                        if v.deadline_ts is not None else None)
                    self.events.log(
                        "VISIT_HIDDEN", visit_id=v.visit_id, face_id=fid, track=t.id,
                        grace_sec=f"{v.grace_seconds:.1f}",
                        deadline_in_sec=f"{deadline_in:.1f}" if deadline_in is not None else None,
                        deadline_ts=self._iso(v.deadline_ts) if v.deadline_ts is not None else None,
                        reason=v.exit_reason)
            t.owns = False

    def _box_occluded_by_others(self, box, exclude_track: Optional[int] = None) -> bool:
        for other in self.tracker.tracks:
            if exclude_track is not None and other.id == exclude_track:
                continue
            if self._iou_box(box, other.box) >= self.presence.occlusion_iou:
                return True
        return False

    def _body_only_keepalive(self, person_dets, away_tracks: List[Track], clock: float) -> None:
        """Face lost but person box still overlaps last position → BODY_ONLY keep-alive."""
        if not self.away_enabled:
            return
        boxes = []
        if person_dets is not None and len(person_dets):
            boxes.extend([np.asarray(d[:4], dtype=np.float64) for d in person_dets])
        for t in away_tracks:
            if self.away_tracker.is_confirmed(t):
                boxes.append(np.asarray(t.box[:4], dtype=np.float64))
        if not boxes:
            return
        for fid, v in list(self.presence.visits.items()):
            if v.state not in (STATE_HIDDEN, STATE_BODY_ONLY) or v.last_box is None:
                continue
            best = max((self._same_body_region(v.last_box, b) for b in boxes), default=0.0)
            if best >= 0.18:
                for b in boxes:
                    if self._same_body_region(v.last_box, b) >= 0.18:
                        self.presence.body_only(fid, clock, b)
                        break

    def _commit_exit(self, face_id: int, reason: str = "hidden_timeout") -> None:
        v = self.presence.visits.get(int(face_id))
        if v is None:
            return
        self.presence.close(int(face_id), reason)
        exit_clock = float(v.last_seen_ts)  # EXIT stamped at last real detection
        when = self._safe_wall(exit_clock)
        iso = _clock.iso_ms(when)
        path = self.events.save_crop("exit", face_id, when, v.last_crop)
        note = reason or v.exit_reason or "hidden_timeout"
        duration_sec = max(0.0, float(v.last_seen_ts) - float(v.entry_ts))
        self.db.log_event(face_id, "exit", iso, path, v.track_id, self.frame_idx, note,
                          visit_id=v.visit_id)
        if v.visit_id is not None:
            # Final visible tick up to last_seen.
            self.db.close_visit(
                v.visit_id, iso, v.visible_seconds, v.occluded_seconds,
                v.resume_count, note)
        self.events.log(
            "EXIT", ts=iso, visit_id=v.visit_id, face_id=face_id, track=v.track_id,
            image=path, duration_sec=f"{duration_sec:.3f}",
            visible_sec=f"{v.visible_seconds:.3f}",
            occluded_sec=f"{v.occluded_seconds:.3f}",
            reason=note, last_seen=iso, frame=self.frame_idx)
        self.present.pop(int(face_id), None)
        if v.track_id is not None:
            self.tracker.purge_lost(int(v.track_id))
        self.presence.pop_exited(int(face_id))

    def _flush_due_exits(self, force: bool = False) -> None:
        if force:
            for v in list(self.presence.open_visits()):
                self._commit_exit(v.face_id, reason=v.exit_reason or "stream_end")
            return
        for v in self.presence.due_exits(self._clock):
            self._commit_exit(v.face_id, reason=v.exit_reason or "hidden_timeout")

    def _flush_pending_exits(self, force: bool = False) -> None:
        """Back-compat alias used by older tests / call sites."""
        self._flush_due_exits(force=force)

    def _detect_disturbance(self, frame: np.ndarray, n_before: int,
                            removed: List[Track], clock: float) -> None:
        if n_before >= 2 and len(removed) / max(n_before, 1) >= self.presence.mass_loss_ratio:
            if not self.presence.in_disturbance():
                self._active_before_loss = n_before
                self.presence.freeze_deadlines(clock, reason="mass_loss")
                self.events.log("DISTURBANCE_START", ts=self._iso(clock), reason="mass_loss",
                                lost=len(removed), active_before=n_before)
        # Brightness collapse / flicker (debounced — webcam AE otherwise floods logs).
        try:
            mean = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
        except Exception:  # noqa: BLE001
            return
        if self._brightness_ema is None:
            self._brightness_ema = mean
            return
        jump = abs(mean - self._brightness_ema)
        self._brightness_ema = 0.9 * self._brightness_ema + 0.1 * mean
        if (
            jump > self._brightness_jump_thr
            and n_before >= 1
            and not self.presence.in_disturbance()
            and float(clock) >= float(self._disturbance_cooldown_until)
        ):
            self._active_before_loss = max(int(n_before), 1)
            self.presence.freeze_deadlines(clock, reason="brightness_jump")
            self.events.log(
                "DISTURBANCE_START", ts=self._iso(clock),
                reason="brightness_jump", delta=f"{jump:.1f}",
            )

    def _disturbance_cleared(self, active: List[Track]) -> bool:
        if not self.presence.in_disturbance():
            return False
        confirmed = sum(1 for t in active if self.tracker.is_confirmed(t))
        return confirmed >= max(1, self._active_before_loss // 2)

    def _maybe_reverify(self, t: Track, frame: np.ndarray) -> None:
        if t.face_id is None or not t.owns:
            return
        interval = self.presence.reverify_interval
        if t.last_reverify_ts is not None and (self._clock - t.last_reverify_ts) < interval:
            return
        crop = crop_box(frame, t.box, 0.25)
        if self._quality_ok(t.box, crop):
            return
        res = self.embedder.embed(crop)
        if res is None:
            return
        emb, _ = res
        t.last_reverify_ts = self._clock
        t.embedding = np.asarray(emb, dtype=np.float32)
        fid, sim = self.gallery.match(emb)
        thr = float(self.rec["match_threshold"])
        soft = float(self.rec.get("soft_match_margin", 0.10))
        mismatch_margin = float(self.rec.get("mismatch_margin", 0.08))
        if fid is not None and fid == t.face_id and sim >= thr - self.presence.resume_margin:
            t.mismatch_streak = 0
            return
        if fid is not None and fid != t.face_id and sim >= thr:
            cur_sim = self.gallery.similarity_to(int(t.face_id), emb)
            # Same person under pose often beats a sibling ID by a tiny margin —
            # keep the current ID and absorb the embedding instead of flipping.
            if cur_sim >= 0.0 and (sim - cur_sim) < mismatch_margin:
                if cur_sim >= thr - soft:
                    t.mismatch_streak = 0
                    if cur_sim < 0.75 and self.gallery.add(int(t.face_id), emb):
                        self._save_embedding(int(t.face_id), emb, frame, t)
                return
            t.mismatch_streak += 1
            self.events.log("IDENTITY_MISMATCH", face_id=t.face_id, new_face=fid,
                            track=t.id, similarity=f"{sim:.3f}", streak=t.mismatch_streak,
                            current_sim=f"{cur_sim:.3f}")
            if t.mismatch_streak >= 2:
                # End wrong visit, re-attach as the matched identity.
                old = int(t.face_id)
                self._commit_exit(old, reason="identity_mismatch")
                t.face_id = None
                t.owns = False
                t.pending = []
                t.pending_scores = []
                self._identified(t, int(fid), sim, emb, frame, is_new=False)
        else:
            t.mismatch_streak += 1

    def _emit(self, kind: str, t: Track, crop: Optional[np.ndarray], note: str = "") -> None:
        """Legacy helper — prefer _emit_entry / _commit_exit."""
        if kind == "entry":
            self._emit_entry(t, crop, note=note)
            return
        if kind == "exit" and t.face_id is not None:
            self._commit_exit(int(t.face_id), reason=note or "stream_end")
            return
        path = self.events.save_crop(kind, t.face_id, self._ts, crop)
        self.db.log_event(t.face_id, kind, self._iso(), path, t.id, self.frame_idx, note)
        self.events.log(kind.upper(), face_id=t.face_id, track=t.id, frame=self.frame_idx,
                        note=note, image=path)
