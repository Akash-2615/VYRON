"""Core pipeline: detect -> track -> identify (embed/match/register) -> entry/exit events.

Also tracks people facing *away* from the camera (no usable face) via a person
detector, stores their clothing appearance, and keeps that KPI separate from
the unique face count.
"""
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .database import Database
from .event_logger import EventLogger
from .gallery import Gallery, normalize
from .tracker import ByteTrackLite, Track, iou_matrix


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
    def __init__(self, cfg: dict, detector, embedder, db: Database, events: EventLogger,
                 appearance=None, person_detector=None):
        self.skip = cfg["detection_skip_frames"]
        self.rec = cfg["recognition"]
        self.quality = cfg["quality"]
        self.detector, self.embedder, self.db, self.events = detector, embedder, db, events
        self.appearance = appearance
        self.person_detector = person_detector
        self.away_cfg = cfg.get("away_walkers", {})
        self.away_enabled = bool(self.away_cfg.get("enabled", True)) and person_detector is not None
        self.tracker = ByteTrackLite(cfg["tracker"], self.skip)
        self.away_tracker = ByteTrackLite(cfg.get("away_tracker", cfg["tracker"]), self.skip)
        self.gallery = Gallery(self.rec["max_embeddings_per_face"])
        for fid, emb in db.load_embeddings():
            self.gallery.add(fid, emb)
        self.present: Dict[int, int] = {}              # face_id -> track_id
        self.away_present: Dict[int, int] = {}         # away_id -> track_id
        self.frame_idx = -1
        self._ts: datetime = datetime.now()

    # ------------------------------------------------------------------ main step
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[List[Track], List[Track]]:
        self.frame_idx, self._ts = frame_idx, datetime.now()
        run_det = frame_idx % (self.skip + 1) == 0
        dets = self.detector.detect(frame) if run_det else None
        h, w = frame.shape[:2]
        active, removed = self.tracker.update(dets, (w, h))

        for t in active:
            t.kind = "face"
            if t.matched_now:
                t.last_crop = crop_box(frame, t.box, 0.10)
            if self.tracker.is_confirmed(t):
                if not t.announced:
                    t.announced = True
                    self.events.log("TRACK_STARTED", track=t.id, frame=frame_idx, conf=f"{t.conf:.2f}")
                if t.face_id is None and t.matched_now and not t.rejected:
                    self._try_identify(t, frame)
        for t in removed:
            self._on_track_removed(t, reason="lost")

        away_active: List[Track] = []
        if self.away_enabled:
            away_active = self._process_away(frame, run_det, active)
        return active, away_active

    def finalize(self) -> None:
        """Stream ended: close every still-open visit so each entry has exactly one exit."""
        for t in list(self.tracker.tracks):
            self._on_track_removed(t, reason="stream_end")
        self.tracker.tracks = []
        for t in list(self.away_tracker.tracks):
            self._on_away_removed(t, reason="stream_end")
        self.away_tracker.tracks = []

    # ------------------------------------------------------------------ away walkers
    def _process_away(self, frame: np.ndarray, run_det: bool, face_tracks: List[Track]) -> List[Track]:
        h, w = frame.shape[:2]
        person_dets = self.person_detector.detect(frame) if run_det else None
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
                if getattr(t, "away_id", None) is None and t.matched_now and not t.rejected:
                    self._register_away(t, frame)
        for t in removed:
            self._on_away_removed(t, reason="lost")
        return active

    def _filter_person_dets(self, dets: np.ndarray, face_tracks: List[Track],
                            frame: np.ndarray) -> np.ndarray:
        """Keep person boxes that look like backs / faces-away (no overlapping face)."""
        if not len(dets):
            return dets
        keep = []
        face_boxes = [t.box for t in face_tracks if self.tracker.is_confirmed(t) or t.hits >= 1]
        require_clip = bool(self.away_cfg.get("require_facing_away", True))
        for det in dets:
            covered = any(_center_in_box(fb, det) for fb in face_boxes)
            if covered:
                continue
            if face_boxes:
                ious = iou_matrix(np.asarray([det[:4]], dtype=np.float64),
                                  np.asarray(face_boxes, dtype=np.float64))
                if ious.size and float(ious.max()) > 0.15:
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

    def _register_away(self, t: Track, frame: np.ndarray) -> None:
        ts = self._ts.isoformat(timespec="milliseconds")
        away_id = self.db.add_away_visitor(ts)
        t.away_id = away_id
        t.owns = True
        self.away_present[away_id] = t.id
        crop = crop_box(frame, t.box, 0.05)
        path = self.events.save_crop("away", away_id, self._ts, crop, id_prefix="away")
        summary = None
        if self.appearance is not None:
            try:
                info = self.appearance.describe(frame, person_box=t.box)
            except Exception:  # noqa: BLE001
                logging.exception("Away appearance failed for %s", away_id)
                info = None
            if info and info.get("summary"):
                summary = info["summary"]
                self.db.set_away_appearance(away_id, summary, json.dumps(info), path)
                self.events.log("AWAY_APPEARANCE", away_id=away_id, summary=summary)
                t.appearance_summary = summary
        path_e = self.events.save_crop("away_entry", away_id, self._ts, crop, id_prefix="away")
        self.db.log_away_event(away_id, "entry", ts, path_e or path, t.id, self.frame_idx, note="away")
        self.events.log("AWAY_ENTRY", away_id=away_id, track=t.id, frame=self.frame_idx,
                        summary=summary, image=path_e or path)

    def _on_away_removed(self, t: Track, reason: str) -> None:
        away_id = getattr(t, "away_id", None)
        self.events.log("AWAY_TRACK_ENDED", track=t.id, away_id=away_id, reason=reason)
        if t.owns and away_id is not None:
            self.away_present.pop(away_id, None)
            path = self.events.save_crop("away_exit", away_id, self._ts, t.last_crop, id_prefix="away")
            self.db.log_away_event(away_id, "exit", self._ts.isoformat(timespec="milliseconds"),
                                   path, t.id, self.frame_idx, note=reason)
            self.events.log("AWAY_EXIT", away_id=away_id, track=t.id, frame=self.frame_idx,
                            note=reason, image=path)
            t.owns = False

    # ------------------------------------------------------------------ identification
    def _quality_ok(self, box, crop) -> Optional[str]:
        x1, y1, x2, y2 = box
        if min(x2 - x1, y2 - y1) < self.quality["min_face_px"]:
            return "too_small"
        if crop is None:
            return "empty_crop"
        gray = cv2.cvtColor(cv2.resize(crop, (112, 112)), cv2.COLOR_BGR2GRAY)
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

        thr = self.rec["match_threshold"]
        fid, sim = self.gallery.match(emb)
        if fid is not None and sim >= thr:
            self._identified(t, fid, sim, emb, frame, is_new=False)
            return

        t.pending.append(emb)
        if len(t.pending) < self.rec["min_embeds_to_register"]:
            return
        mean = normalize(np.mean(t.pending, axis=0))
        fid, sim = self.gallery.match(mean)
        if fid is not None and sim >= thr:
            self._identified(t, fid, sim, mean, frame, is_new=False)
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
        if not is_new:
            self.events.log("FACE_RECOGNIZED", face_id=face_id, track=t.id, similarity=f"{sim:.3f}")
            if sim < 0.75 and self.gallery.add(face_id, emb):
                self._save_embedding(face_id, emb, frame, t)

        existing = self.db.get_appearance_summary(face_id)
        if is_new or not existing:
            t.appearance_summary = self._describe_appearance(face_id, frame, t.box)
        else:
            t.appearance_summary = existing

        owner_id = self.present.get(face_id)
        owner = self.tracker.get(owner_id) if owner_id is not None else None
        if owner is None:
            t.owns = True
            self.present[face_id] = t.id
            self._emit("entry", t, crop_box(frame, t.box, 0.10), note="new" if is_new else "re-entry")
        elif owner.time_since_update > 0:
            owner.owns = False
            t.owns = True
            self.present[face_id] = t.id
            self.events.log("TRACK_HANDOVER", face_id=face_id, old_track=owner.id, new_track=t.id)
        else:
            t.owns = False
            self.events.log("DUPLICATE_TRACK", logging.WARNING, face_id=face_id, track=t.id, owner=owner.id)

    def _describe_appearance(self, face_id: int, frame: np.ndarray, box) -> Optional[str]:
        if self.appearance is None:
            return None
        try:
            info = self.appearance.describe(frame, face_box=box)
        except Exception:  # noqa: BLE001
            logging.exception("Appearance describe failed for face %s", face_id)
            return None
        if not info or not info.get("summary"):
            return None
        self.db.set_appearance(face_id, info["summary"], json.dumps(info))
        self.events.log("APPEARANCE_DESCRIBED", face_id=face_id, summary=info["summary"])
        return info["summary"]

    def _save_embedding(self, face_id: int, emb: np.ndarray, frame: np.ndarray, t: Track,
                        primary: bool = False) -> Optional[str]:
        path = self.events.save_crop("face", face_id, self._ts, crop_box(frame, t.box, 0.05))
        self.db.add_embedding(face_id, emb, self._ts.isoformat(timespec="milliseconds"), path)
        if primary and path:
            self.db.set_face_image(face_id, path)
        return path

    def _on_track_removed(self, t: Track, reason: str) -> None:
        self.events.log("TRACK_ENDED", track=t.id, face_id=t.face_id, reason=reason)
        if t.owns and t.face_id is not None:
            self.present.pop(t.face_id, None)
            self._emit("exit", t, t.last_crop, note=reason)
            t.owns = False

    def _emit(self, kind: str, t: Track, crop: Optional[np.ndarray], note: str = "") -> None:
        ts = self._ts
        path = self.events.save_crop(kind, t.face_id, ts, crop)
        self.db.log_event(t.face_id, kind, ts.isoformat(timespec="milliseconds"),
                          path, t.id, self.frame_idx, note)
        self.events.log(kind.upper(), face_id=t.face_id, track=t.id, frame=self.frame_idx,
                        note=note, image=path)
