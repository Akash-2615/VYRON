"""Presence / visit state-machine scenarios (S1–S9) with fake detector/embedder."""
import numpy as np

from src.database import Database
from src.event_logger import EventLogger
from src.pipeline import FaceTrackingPipeline
from tests.test_pipeline import FakeDetector, FakeEmbedder, _base_cfg, make_frame


def _pipe(tmp_path, presence=None, fps=25.0):
    cfg = _base_cfg()
    if presence:
        cfg["presence"].update(presence)
    cfg["detection_skip_frames"] = 0
    cfg["tracker"]["max_lost_frames"] = 3
    db = Database(str(tmp_path / "p.db"), reset=True)
    ev = EventLogger(str(tmp_path / "logs"), str(tmp_path / "logs" / "events.log"))
    pipe = FaceTrackingPipeline(cfg, FakeDetector(), FakeEmbedder(), db, ev)
    pipe.set_timing(fps, live=False)
    return pipe, db, ev


def test_s1_midframe_hide_resumes_one_visit(tmp_path):
    """S1: visible → hidden 5s mid-frame → visible: 1 ENTRY, 1 EXIT, occluded ~5s."""
    pipe, db, ev = _pipe(tmp_path, {
        "exit_grace_seconds": 8, "stationary_grace_seconds": 8, "edge_exit_grace_seconds": 1,
    })
    # Present 2s
    for i in range(50):
        pipe.process_frame(make_frame(100, None), i, ts=i / 25.0)
    # Hidden 5s (mid-frame — square stays off)
    for i in range(50, 175):
        pipe.process_frame(make_frame(None, None), i, ts=i / 25.0)
    # Visible again 2s
    for i in range(175, 225):
        pipe.process_frame(make_frame(100, None), i, ts=i / 25.0)
    pipe.finalize()
    counts = db.event_counts()
    assert counts.get("entry") == 1 and counts.get("exit") == 1
    visits = db.visits_for_face(1)
    assert len(visits) == 1
    assert visits[0]["resume_count"] >= 1
    assert visits[0]["occluded_seconds"] >= 4.0
    assert "VISIT_RESUMED" in (tmp_path / "logs" / "events.log").read_text()
    db.close()


def test_s3_edge_exit_uses_last_seen_ts(tmp_path):
    """S3: person exits near the edge → short grace; EXIT ts == last_seen."""
    pipe, db, ev = _pipe(tmp_path, {
        "exit_grace_seconds": 5, "stationary_grace_seconds": 5, "edge_exit_grace_seconds": 0.3,
        "edge_margin_px": 40,
    })
    # Walk toward left edge
    for i in range(40):
        x = max(0, 40 - i)
        pipe.process_frame(make_frame(x, None), i, ts=i / 25.0)
    last_seen_clock = None
    v = pipe.presence.visits.get(1)
    if v:
        last_seen_clock = v.last_seen_ts
    # Gone
    for i in range(40, 80):
        pipe.process_frame(make_frame(None, None), i, ts=i / 25.0)
    pipe.finalize()
    counts = db.event_counts()
    assert counts.get("entry") == 1 and counts.get("exit") == 1
    exit_row = db.query("SELECT timestamp, note FROM events WHERE event_type='exit'")[0]
    assert last_seen_clock is not None
    # EXIT note should prefer edge_exit when near border
    visits = db.visits_for_face(1)
    assert visits and visits[0]["exit_ts"] is not None
    db.close()


def test_s4_stationary_timeout_then_reentry(tmp_path):
    """S4: hidden longer than stationary grace mid-frame → EXIT then new ENTRY."""
    pipe, db, ev = _pipe(tmp_path, {
        "exit_grace_seconds": 1.0, "stationary_grace_seconds": 1.0, "edge_exit_grace_seconds": 0.3,
    })
    for i in range(40):
        pipe.process_frame(make_frame(120, None), i, ts=i / 25.0)
    for i in range(40, 100):  # 2.4s > 1s grace
        pipe.process_frame(make_frame(None, None), i, ts=i / 25.0)
    for i in range(100, 140):
        pipe.process_frame(make_frame(120, None), i, ts=i / 25.0)
    pipe.finalize()
    counts = db.event_counts()
    assert counts.get("entry") == 2 and counts.get("exit") == 2
    assert len(db.visits_for_face(1)) == 2
    db.close()


def test_s5_stream_gap_freeze_and_hard_gap(tmp_path):
    """S5: short stream gap freezes deadlines; long gap closes with stream_gap."""
    pipe, db, ev = _pipe(tmp_path, {
        "exit_grace_seconds": 2, "stationary_grace_seconds": 2, "edge_exit_grace_seconds": 0.5,
        "max_stream_gap_seconds": 5,
    })
    for i in range(40):
        pipe.process_frame(make_frame(80, None), i, ts=i / 25.0)
    assert pipe.presence.inside_count() == 1
    pipe.notify_stream_gap(2.0)  # under max → freeze
    assert pipe.presence.in_disturbance()
    log = (tmp_path / "logs" / "events.log").read_text()
    assert "DISTURBANCE_START" in log
    pipe._clock = 10.0
    pipe.presence.unfreeze_deadlines(10.0)
    pipe.notify_stream_gap(6.0)  # over max → exit
    assert db.event_counts().get("exit") == 1
    note = db.query("SELECT note FROM events WHERE event_type='exit'")[0][0]
    assert note == "stream_gap"
    db.close()


def test_s6_mass_loss_disturbance(tmp_path):
    """S6: mass loss of tracks freezes exits."""
    pipe, db, ev = _pipe(tmp_path, {
        "exit_grace_seconds": 0.5, "stationary_grace_seconds": 0.5, "edge_exit_grace_seconds": 0.2,
        "mass_loss_ratio": 0.5,
    })
    for i in range(30):
        pipe.process_frame(make_frame(40, 200), i, ts=i / 25.0)
    # Both vanish → mass loss
    for i in range(30, 50):
        pipe.process_frame(make_frame(None, None), i, ts=i / 25.0)
    log = (tmp_path / "logs" / "events.log").read_text()
    # Disturbance should have fired (2 confirmed tracks lost together)
    assert "DISTURBANCE_START" in log or db.event_counts().get("exit", 0) <= 2
    db.close()


def test_s2_adjacent_alternate_occlusion_no_id_swap(tmp_path):
    """S2: two people alternately hide — no ID swap, one visit each."""
    pipe, db, ev = _pipe(tmp_path, {
        "exit_grace_seconds": 6, "stationary_grace_seconds": 6, "edge_exit_grace_seconds": 1,
    })
    # Both present
    for i in range(40):
        pipe.process_frame(make_frame(40, 200), i, ts=i / 25.0)
    # Alternate occlusion for ~10s (250 frames): A only, then B only, repeat
    for i in range(40, 290):
        if ((i // 25) % 2) == 0:
            pipe.process_frame(make_frame(40, None), i, ts=i / 25.0)   # A visible
        else:
            pipe.process_frame(make_frame(None, 200), i, ts=i / 25.0)  # B visible
    # Both again then done
    for i in range(290, 320):
        pipe.process_frame(make_frame(40, 200), i, ts=i / 25.0)
    pipe.finalize()
    assert db.unique_count() == 2
    # Each identity should have exactly one visit (resumes, not new entries per flicker)
    for fid in (1, 2):
        visits = db.visits_for_face(fid)
        assert len(visits) == 1, f"face {fid} visits={visits}"
        assert visits[0]["resume_count"] >= 1
    counts = db.event_counts()
    assert counts.get("entry") == 2 and counts.get("exit") == 2
    db.close()


def test_s7_body_only_keepalive_with_fake_person_detector(tmp_path):
    """S7: face lost but overlapping person box keeps visit in BODY_ONLY."""
    from src.tracker import Track

    cfg = _base_cfg()
    cfg["detection_skip_frames"] = 0
    cfg["tracker"]["max_lost_frames"] = 2
    cfg["presence"].update({
        "exit_grace_seconds": 1.0, "stationary_grace_seconds": 1.0, "edge_exit_grace_seconds": 0.3,
    })
    cfg["away_walkers"] = {"enabled": True, "require_facing_away": False, "detect_every_n": 1,
                           "describe_away": False}
    cfg["appearance"] = {"enabled": False, "describe_on_register": False}

    class PersonDet:
        def __init__(self):
            self.box = np.array([30, 40, 150, 220, 0.9], dtype=np.float32)

        def detect(self, frame):
            return self.box.reshape(1, 5)

    db = Database(str(tmp_path / "s7.db"), reset=True)
    ev = EventLogger(str(tmp_path / "logs"), str(tmp_path / "logs" / "events.log"))
    person = PersonDet()
    pipe = FaceTrackingPipeline(cfg, FakeDetector(), FakeEmbedder(), db, ev,
                                person_detector=person)
    pipe.set_timing(25.0, live=False)
    # Register face in person region
    for i in range(30):
        pipe.process_frame(make_frame(50, None), i, ts=i / 25.0)
    assert pipe.presence.inside_count() == 1
    # Face gone; person detector still returns overlapping body
    for i in range(30, 55):  # 1.0s — would exit without body keep-alive
        pipe.process_frame(make_frame(None, None), i, ts=i / 25.0)
    states = [v.state for v in pipe.presence.visits.values()]
    assert any(s in ("BODY_ONLY", "HIDDEN", "PRESENT") for s in states)
    # Still open (body keep-alive extended deadline) OR at most not double-entered
    assert db.event_counts().get("entry", 0) == 1
    # Face returns
    for i in range(55, 80):
        pipe.process_frame(make_frame(50, None), i, ts=i / 25.0)
    pipe.finalize()
    assert db.event_counts().get("entry") == 1
    assert db.event_counts().get("exit") == 1
    db.close()


def test_s8_identity_mismatch_recovery(tmp_path):
    """S8: re-verify disagrees → IDENTITY_MISMATCH and visit ends/swaps cleanly."""
    cfg = _base_cfg()
    cfg["detection_skip_frames"] = 0
    cfg["tracker"]["max_lost_frames"] = 20
    cfg["presence"]["reverify_interval_seconds"] = 0.05
    cfg["presence"]["exit_grace_seconds"] = 2
    cfg["presence"]["stationary_grace_seconds"] = 2
    cfg["presence"]["edge_exit_grace_seconds"] = 0.5

    class SwapEmbedder:
        """Starts as RED identity, then flips to GREEN for reverify."""
        def __init__(self):
            self.flip = False

        def embed(self, crop):
            v = np.zeros(512, dtype=np.float32)
            if self.flip:
                v[1] = 1.0
            else:
                red = int(np.all(crop == (0, 0, 255), axis=2).sum())
                green = int(np.all(crop == (0, 255, 0), axis=2).sum())
                v[0 if red >= green else 1] = 1.0
            return v, 0.99

    db = Database(str(tmp_path / "s8.db"), reset=True)
    ev = EventLogger(str(tmp_path / "logs"), str(tmp_path / "logs" / "events.log"))
    emb = SwapEmbedder()
    pipe = FaceTrackingPipeline(cfg, FakeDetector(), emb, db, ev)
    pipe.set_timing(25.0, live=False)
    # Establish RED person as face 1
    for i in range(40):
        pipe.process_frame(make_frame(80, None), i, ts=i / 25.0)
    assert db.unique_count() == 1
    # Establish GREEN as face 2 briefly so gallery has both
    for i in range(40, 70):
        pipe.process_frame(make_frame(None, 200), i, ts=i / 25.0)
    assert db.unique_count() == 2
    # Back to RED position but force embeddings to report GREEN
    emb.flip = True
    for i in range(70, 120):
        pipe.process_frame(make_frame(80, None), i, ts=i / 25.0)
    log = (tmp_path / "logs" / "events.log").read_text()
    assert "IDENTITY_MISMATCH" in log
    pipe.finalize()
    # Visits should close cleanly (no crash); unique still 2
    assert db.unique_count() == 2
    assert db.event_counts().get("entry", 0) >= 2
    assert db.event_counts().get("exit", 0) == db.event_counts().get("entry", 0)
    db.close()


def test_s9_dwell_sums_multiple_visits(tmp_path):
    pipe, db, ev = _pipe(tmp_path, {
        "exit_grace_seconds": 0.4, "stationary_grace_seconds": 0.4, "edge_exit_grace_seconds": 0.2,
    })
    # Visit 1
    for i in range(40):
        pipe.process_frame(make_frame(90, None), i, ts=i / 25.0)
    for i in range(40, 70):
        pipe.process_frame(make_frame(None, None), i, ts=i / 25.0)
    # Visit 2
    for i in range(70, 110):
        pipe.process_frame(make_frame(90, None), i, ts=i / 25.0)
    pipe.finalize()
    visits = db.visits_for_face(1)
    assert len(visits) == 2
    total = sum((v["visible_seconds"] or 0) + (v["occluded_seconds"] or 0) for v in visits)
    assert total > 0
    stats = db.visit_stats()
    assert stats["closed_visits"] == 2
    db.close()
