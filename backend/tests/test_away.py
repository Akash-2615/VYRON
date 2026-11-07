"""Away-walker DB + filtering helpers (no YOLO/CLIP required)."""
import numpy as np

from src.database import Database
from src.event_logger import EventLogger
from src.pipeline import FaceTrackingPipeline
from src.tracker import Track


def test_away_registers_with_sparse_person_detect(tmp_path):
    """detect_every_n>1 must still confirm + store away visitors (interval covers period)."""
    db = Database(str(tmp_path / "sparse.db"), reset=True)
    ev = EventLogger(str(tmp_path / "logs"), str(tmp_path / "logs" / "events.log"))
    skip = 1
    every = 2
    cfg = {
        "detection_skip_frames": skip,
        "recognition": {"match_threshold": 0.45, "max_embeddings_per_face": 5,
                        "min_embeds_to_register": 2, "embed_retry_interval": 1,
                        "max_embed_attempts": 5},
        "quality": {"min_face_px": 10, "min_blur_var": 0},
        "tracker": {"high_conf": 0.5, "new_track_conf": 0.5, "iou_threshold": 0.25,
                    "min_hits": 2, "max_lost_frames": 10},
        "away_tracker": {"high_conf": 0.3, "new_track_conf": 0.3, "iou_threshold": 0.2,
                         "min_hits": 2, "max_lost_frames": 30},
        "away_walkers": {"enabled": True, "require_facing_away": False,
                         "describe_away": False, "detect_every_n": every},
        "appearance": {"enabled": False},
    }
    person_box = np.array([[40, 40, 160, 220, 0.9]], dtype=np.float32)

    class FaceDet:
        def detect(self, frame):
            return np.zeros((0, 5), dtype=np.float32)

    class Emb:
        def embed(self, crop):
            return None

    class PersonDet:
        def detect(self, frame):
            return person_box.copy()

    pipe = FaceTrackingPipeline(
        cfg, FaceDet(), Emb(), db, ev, person_detector=PersonDet())
    assert pipe.away_tracker.interval == (skip + 1) * every
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    # Two person-detect cycles → hits>=2 → AWAY_ENTRY in DB.
    for i in range((skip + 1) * every * 2 + 1):
        pipe.process_frame(frame, i, ts=float(i) * 0.04)
    assert db.away_stats()["away_unique"] >= 1
    assert db.away_stats()["away_entries"] >= 1
    away_log = (tmp_path / "logs" / "away_events.log").read_text()
    assert "AWAY_ENTRY" in away_log
    assert "AWAY_TRACK_STARTED" in away_log
    db.close()


def test_away_visitor_stats(tmp_path):
    db = Database(str(tmp_path / "a.db"), reset=True)
    aid = db.add_away_visitor("2026-11-06T10:00:00.000")
    db.set_away_appearance(aid, "black jacket, jeans", '{"summary":"black jacket, jeans"}', None)
    db.log_away_event(aid, "entry", "2026-11-06T10:00:01.000", None, 1, 0, "away")
    db.log_away_event(aid, "exit", "2026-11-06T10:00:20.000", None, 1, 10, "lost")
    s = db.away_stats()
    assert s["away_unique"] == 1
    assert s["away_entries"] == 1 and s["away_exits"] == 1
    assert s["away_currently"] == 0
    assert s["away_with_clothes"] == 1
    db.close()


def test_filter_person_dets_centre_only(tmp_path):
    """A face beside a person must not remove them; only a face centred in the body does."""
    from src.pipeline import FaceTrackingPipeline, _center_in_box

    db = Database(str(tmp_path / "f.db"), reset=True)
    ev = EventLogger(str(tmp_path / "logs"), str(tmp_path / "logs" / "events.log"))
    cfg = {
        "detection_skip_frames": 0,
        "recognition": {"match_threshold": 0.45, "max_embeddings_per_face": 5,
                        "min_embeds_to_register": 2, "embed_retry_interval": 1,
                        "max_embed_attempts": 5},
        "quality": {"min_face_px": 10, "min_blur_var": 0},
        "tracker": {"high_conf": 0.5, "new_track_conf": 0.5, "iou_threshold": 0.25,
                    "min_hits": 1, "max_lost_frames": 10},
        "away_walkers": {"enabled": True, "require_facing_away": False},
        "appearance": {"enabled": False},
    }

    class Fake:
        def detect(self, frame):
            return np.zeros((0, 5), dtype=np.float32)

        def embed(self, crop):
            return None

    pipe = FaceTrackingPipeline(cfg, Fake(), Fake(), db, ev, person_detector=Fake())
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    person = np.array([[40, 40, 160, 220, 0.9]], dtype=np.float32)
    # Face far to the right — overlaps box in IoU terms if expanded, but centre not inside
    near = Track(np.array([150, 30, 190, 70], dtype=np.float64), 0.9)
    near.hits = 5
    kept = pipe._filter_person_dets(person, [near], frame)
    assert len(kept) == 1
    # Face centred in the torso → drop
    inside = Track(np.array([80, 50, 120, 100], dtype=np.float64), 0.9)
    inside.hits = 5
    assert _center_in_box(inside.box, person[0], pad=0.05)
    kept2 = pipe._filter_person_dets(person, [inside], frame)
    assert len(kept2) == 0
    db.close()


def test_away_resume_same_person_still_inside(tmp_path):
    """Brief loss of a sitting person must not create a new away_id every second."""
    db = Database(str(tmp_path / "c.db"), reset=True)
    ev = EventLogger(str(tmp_path / "logs"), str(tmp_path / "logs" / "events.log"))
    cfg = {
        "detection_skip_frames": 0,
        "recognition": {"match_threshold": 0.45, "max_embeddings_per_face": 5,
                        "min_embeds_to_register": 2, "embed_retry_interval": 1,
                        "max_embed_attempts": 5},
        "quality": {"min_face_px": 10, "min_blur_var": 0},
        "tracker": {"high_conf": 0.5, "new_track_conf": 0.5, "iou_threshold": 0.25,
                    "min_hits": 1, "max_lost_frames": 5,
                    "reentry_grace_frames": 5, "reentry_grace_sec": 0.01},
        "away_tracker": {"high_conf": 0.3, "new_track_conf": 0.3, "iou_threshold": 0.2,
                         "min_hits": 1, "max_lost_frames": 3},
        "away_walkers": {"enabled": True, "require_facing_away": False,
                         "describe_away": False, "detect_every_n": 1,
                         "reentry_grace_frames": 40, "reentry_grace_sec": 30.0},
        "appearance": {"enabled": False},
    }

    class Fake:
        def detect(self, frame):
            return np.zeros((0, 5), dtype=np.float32)

        def embed(self, crop):
            return None

    pipe = FaceTrackingPipeline(cfg, Fake(), Fake(), db, ev, person_detector=Fake())
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    t1 = Track(np.array([40, 40, 160, 220], dtype=np.float64), 0.9)
    pipe._register_away(t1, frame)
    aid = t1.away_id
    assert aid is not None
    pipe.away_tracker.tracks.append(t1)
    pipe._on_away_removed(t1, reason="lost")
    assert aid in pipe.pending_away_exit
    t2 = Track(np.array([50, 50, 170, 230], dtype=np.float64), 0.9)
    pipe.away_tracker.tracks.append(t2)
    pipe._register_away(t2, frame)
    assert t2.away_id == aid
    assert aid not in pipe.pending_away_exit
    assert db.away_stats()["away_unique"] == 1
    assert db.away_stats()["away_entries"] == 1
    away_log = (tmp_path / "logs" / "away_events.log").read_text()
    assert "AWAY_RESUMED" in away_log
    assert "AWAY_RESUMED" not in (tmp_path / "logs" / "events.log").read_text()
    db.close()


def test_away_linked_face_and_promote(tmp_path):
    db = Database(str(tmp_path / "b.db"), reset=True)
    aid = db.add_away_visitor("2026-11-06T10:00:00.000")
    fid = db.add_face("2026-11-06T10:00:05.000")
    db.set_away_linked_face(aid, fid)
    row = db.query("SELECT linked_face_id FROM away_visitors WHERE away_id=?", (aid,))
    assert row[0][0] == fid

    ev = EventLogger(str(tmp_path / "logs"), str(tmp_path / "logs" / "events.log"))
    cfg = {
        "detection_skip_frames": 0,
        "recognition": {"match_threshold": 0.45, "max_embeddings_per_face": 5,
                        "min_embeds_to_register": 2, "embed_retry_interval": 1,
                        "max_embed_attempts": 5},
        "quality": {"min_face_px": 10, "min_blur_var": 0},
        "tracker": {"high_conf": 0.5, "new_track_conf": 0.5, "iou_threshold": 0.25,
                    "min_hits": 1, "max_lost_frames": 10},
        "away_walkers": {"enabled": True, "require_facing_away": False},
        "appearance": {"enabled": False, "describe_on_register": False},
    }

    class FakeDet:
        def detect(self, frame):
            return np.zeros((0, 5), dtype=np.float32)

    class FakeEmb:
        def embed(self, crop):
            return None

    pipe = FaceTrackingPipeline(cfg, FakeDet(), FakeEmb(), db, ev, person_detector=FakeDet())
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    away = Track(np.array([40, 40, 160, 220], dtype=np.float64), 0.9)
    away.away_id = aid
    away.owns = True
    pipe.away_present[aid] = away.id
    pipe.away_tracker.tracks.append(away)
    face = Track(np.array([70, 50, 110, 100], dtype=np.float64), 0.9)
    out = pipe._promote_turned_faces([face], [away], frame)
    assert out == []
    assert face.promoted_from_away == aid
    assert aid not in pipe.away_present
    db.close()


def _pipe(tmp_path):
    db = Database(str(tmp_path / "c.db"), reset=True)
    ev = EventLogger(str(tmp_path / "logs"), str(tmp_path / "logs" / "events.log"))
    cfg = {
        "detection_skip_frames": 0,
        "recognition": {"match_threshold": 0.45, "max_embeddings_per_face": 5,
                        "min_embeds_to_register": 2, "embed_retry_interval": 1,
                        "max_embed_attempts": 5},
        "quality": {"min_face_px": 10, "min_blur_var": 0},
        "tracker": {"high_conf": 0.5, "new_track_conf": 0.5, "iou_threshold": 0.25,
                    "min_hits": 1, "max_lost_frames": 10,
                    "reentry_grace_frames": 5, "reentry_grace_sec": 0.5},
        "away_walkers": {"enabled": True, "require_facing_away": False,
                         "reentry_grace_frames": 3, "reentry_grace_sec": 0.2},
        "appearance": {"enabled": False, "describe_on_register": False},
    }

    class FakeDet:
        def detect(self, frame):
            return np.zeros((0, 5), dtype=np.float32)

    class FakeEmb:
        def embed(self, crop):
            return None

    pipe = FaceTrackingPipeline(cfg, FakeDet(), FakeEmb(), db, ev, person_detector=FakeDet())
    return pipe, db


def test_face_continues_as_away_when_turned(tmp_path):
    """Identified face turns around but body stays → same visit, followed as no-face."""
    pipe, db = _pipe(tmp_path)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    fid = db.add_face("2026-11-06T10:00:00.000")
    db.log_event(fid, "entry", "2026-11-06T10:00:01.000", None, 1, 0, "new")

    face = Track(np.array([80, 40, 120, 90], dtype=np.float64), 0.9)
    face.face_id = fid
    face.owns = True
    face.hits = 3
    face.age = 10
    pipe.present[fid] = face.id
    pipe.tracker.tracks.append(face)

    away = Track(np.array([40, 30, 160, 220], dtype=np.float64), 0.9)
    away.hits = 3
    away.age = 5
    pipe.away_tracker.tracks.append(away)

    ok = pipe._continue_face_as_away(face, [away], frame)
    assert ok is True
    assert fid in pipe.face_as_away
    assert away.linked_face_id == fid
    assert fid not in pipe.present
    assert face.owns is False
    assert getattr(away, "away_id", None) is not None

    counts = pipe.in_frame_counts([], [away])
    assert counts["persons"] == 1
    assert counts["faces"] == 0
    assert counts["no_face"] == 1
    assert counts["faces_turned_away"] == 1

    # Turn back to face → resume same visit, no new ENTRY.
    pipe.frame_idx = 20
    face2 = Track(np.array([85, 45, 115, 95], dtype=np.float64), 0.9)
    out = pipe._promote_turned_faces([face2], [away], frame)
    assert out == []
    assert face2.face_id == fid
    assert face2.owns is True
    assert fid in pipe.present
    assert fid not in pipe.face_as_away
    main = (tmp_path / "logs" / "events.log").read_text()
    away = (tmp_path / "logs" / "away_events.log").read_text()
    assert "FACE_CONTINUED_AS_AWAY" in away
    assert "FACE_CONTINUED_AS_AWAY" not in main
    assert "VISIT_RESUMED" in main
    db.close()


def test_in_frame_counts_no_double(tmp_path):
    pipe, db = _pipe(tmp_path)
    face = Track(np.array([80, 40, 120, 90], dtype=np.float64), 0.9)
    face.hits = 3
    face.face_id = 1
    away = Track(np.array([200, 40, 280, 200], dtype=np.float64), 0.9)
    away.hits = 3
    away.away_id = 9
    counts = pipe.in_frame_counts([face], [away])
    assert counts["persons"] == 2
    assert counts["faces"] == 1
    assert counts["no_face"] == 1
    db.close()


def test_hidden_visit_held_while_body_still_in_frame(tmp_path):
    """Seated person: face track dies but body still visible → BODY_ONLY, no EXIT."""
    pipe, db = _pipe(tmp_path)
    fid = db.add_face("2026-11-06T10:00:00.000")
    pipe.set_timing(25.0, live=False)
    pipe._clock = 10.0
    v = pipe.presence.start_visit(fid, 0.0, track_id=1,
                                  box=np.array([40, 30, 160, 220], dtype=np.float64),
                                  visit_id=db.start_visit(fid, "2026-11-06T10:00:00.000"))
    pipe.presence.hide(fid, 10.0, (320, 240), occluded_by=False)
    assert pipe.presence.visits[fid].state in ("HIDDEN", "BODY_ONLY")
    person = np.array([[45, 35, 155, 215, 0.9]], dtype=np.float32)
    pipe._body_only_keepalive(person, [], 10.5)
    assert pipe.presence.visits[fid].state == "BODY_ONLY"
    pipe._clock = 11.0
    pipe._flush_due_exits(force=False)
    assert fid in pipe.presence.visits  # must NOT commit EXIT yet
    assert db.event_counts().get("exit", 0) == 0
    db.close()


def test_time_grace_not_frame_count(tmp_path):
    """Many frames with little wall/video time must not force EXIT."""
    pipe, db = _pipe(tmp_path)
    fid = db.add_face("2026-11-06T10:00:00.000")
    pipe.set_timing(25.0, live=False)
    pipe.presence.exit_grace = 30.0
    pipe.presence.stationary_grace = 30.0
    pipe.presence.start_visit(fid, 0.0, track_id=1,
                              box=np.array([100, 80, 140, 130], dtype=np.float64),
                              visit_id=db.start_visit(fid, "2026-11-06T10:00:00.000"))
    pipe.presence.hide(fid, 1.0, (320, 240))
    pipe._clock = 2.0  # only 1s hidden; grace is 30s
    pipe.frame_idx = 100
    pipe._flush_due_exits(force=False)
    assert fid in pipe.presence.visits
    db.close()
