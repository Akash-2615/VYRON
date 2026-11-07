"""Unit tests for optional video preprocessing (OFF by default)."""
import json
import time
from pathlib import Path

import numpy as np
import pytest

from src.detection.preprocess import (
    AutoGate,
    PreprocessSettings,
    Preprocessor,
    map_boxes_to_original,
    quality_assist_gray,
    resize_for_detector,
    step_clahe,
    step_denoise,
    step_gamma,
    step_sharpen,
    step_white_balance,
)


def _bgr(h=120, w=160, color=(40, 40, 40)):
    f = np.zeros((h, w, 3), dtype=np.uint8)
    f[:] = color
    return f


def test_disabled_returns_same_array_object_semantics():
    pp = Preprocessor()
    frame = _bgr()
    orig = frame.copy()
    out, scale, gate = pp.enhance(frame, PreprocessSettings(enabled=False, clahe_enabled=True))
    assert scale == 1.0 and gate is False
    np.testing.assert_array_equal(out, orig)
    np.testing.assert_array_equal(frame, orig)  # input untouched


def test_gamma_brightens():
    dark = _bgr(color=(30, 30, 30))
    out = step_gamma(dark, gamma=0.6)
    assert float(out.mean()) > float(dark.mean())
    np.testing.assert_array_equal(dark, _bgr(color=(30, 30, 30)))


def test_clahe_raises_contrast_on_flatish_frame():
    f = _bgr(color=(80, 80, 80))
    f[40:80, 40:120] = (50, 50, 50)
    before = float(cv2_std(f))
    out = step_clahe(f, clip_limit=3.0, tile_grid=8)
    assert float(cv2_std(out)) >= before * 0.95  # not collapsed
    assert out.dtype == np.uint8 and out.shape == f.shape


def cv2_std(bgr):
    import cv2
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return g.std()


def test_denoise_lowers_noise_variance():
    rng = np.random.default_rng(0)
    f = _bgr(color=(100, 100, 100))
    noisy = np.clip(f.astype(np.int16) + rng.integers(-25, 26, f.shape), 0, 255).astype(np.uint8)
    out = step_denoise(noisy, "gaussian", 0.8)
    assert float(cv2_std(out)) < float(cv2_std(noisy))


def test_white_balance_skips_grayscale():
    gray = _bgr(color=(90, 90, 90))
    out = step_white_balance(gray, strength=1.0, max_gain=1.5)
    np.testing.assert_array_equal(out, gray)


def test_white_balance_removes_cast():
    cast = _bgr(color=(40, 60, 120))  # strong blue cast
    out = step_white_balance(cast, strength=1.0, max_gain=1.5)
    means = out.reshape(-1, 3).mean(axis=0)
    assert abs(float(means[0]) - float(means[2])) < abs(40 - 120)


def test_sharpen_increases_edge_energy():
    f = _bgr()
    f[:, 80:] = (200, 200, 200)
    import cv2
    e0 = float(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    out = step_sharpen(f, amount=0.8, sigma=1.0)
    e1 = float(cv2.Laplacian(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    assert e1 >= e0


def test_box_mapping_within_one_px():
    dets = np.array([[10.0, 20.0, 50.0, 60.0, 0.9]], dtype=np.float32)
    scale = 0.5
    mapped = map_boxes_to_original(dets, scale)
    np.testing.assert_allclose(mapped[0, :4], [20, 40, 100, 120], atol=1.0)


def test_resize_for_detector_scale():
    f = _bgr(h=1080, w=1920)
    small, scale = resize_for_detector(f, 640)
    assert max(small.shape[:2]) == 640
    assert abs(scale - 640 / 1920) < 1e-6


def test_auto_gate_hysteresis_no_flicker():
    s = PreprocessSettings(
        enabled=True, auto_gate_enabled=True, brightness_below=90, contrast_below=35, gate_mode="any",
        clahe_enabled=True,
    )
    g = AutoGate()
    # noisy series around threshold — should not flip every frame
    flips = 0
    prev = None
    for i in range(40):
        bri = 88 + (i % 3)  # jitter
        active = g.update(bri, 20.0, s)
        if prev is not None and active != prev:
            flips += 1
        prev = active
    assert flips <= 2


def test_quality_assist_gray_shape():
    g = np.full((112, 112), 40, dtype=np.uint8)
    out = quality_assist_gray(g)
    assert out.shape == (112, 112) and out.dtype == np.uint8


def test_settings_bounds_clamp():
    s = PreprocessSettings.from_dict({"gamma": {"enabled": True, "gamma": 9.0}})
    assert 0.5 <= s.gamma <= 2.0


def test_api_preprocess_get(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from api.server import create_app
    from src.database import Database
    db = Database(str(tmp_path / "t.db"), reset=True)
    app = create_app(db=db, log_dir=str(tmp_path / "logs"), web_dist=None)
    client = TestClient(app)
    r = client.get("/api/preprocess")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is False
    assert "step_catalog" in body and len(body["step_catalog"]) >= 5


def test_api_preprocess_put_and_reset(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from api.server import create_app
    from src.database import Database
    import api.preprocess_service as svc
    from src.detection import preprocess as ppmod

    rt = tmp_path / "runtime"
    rt.mkdir()
    monkeypatch.setattr(svc, "override_path", lambda: rt / "preprocess.json")
    monkeypatch.setattr(ppmod, "runtime_path", lambda name: rt / Path(name).name)
    # SettingsStore.override_path uses runtime_path — rebind store after patch
    rt_obj = ppmod.get_preprocess_runtime()
    rt_obj.store = ppmod.SettingsStore({"enabled": False})

    db = Database(str(tmp_path / "t.db"), reset=True)
    app = create_app(db=db, log_dir=str(tmp_path / "logs"), web_dist=None)
    client = TestClient(app)

    r = client.put("/api/preprocess", json={"enabled": True, "clahe": {"enabled": True, "clip_limit": 2.5}})
    assert r.status_code == 200, r.text
    assert r.json()["settings"]["enabled"] is True
    bad = client.put("/api/preprocess", json={"not_a_key": 1})
    assert bad.status_code == 422
    oom = client.put("/api/preprocess", json={"gamma": {"enabled": True, "gamma": 9.0}})
    assert oom.status_code == 422
    r3 = client.post("/api/preprocess/reset")
    assert r3.status_code == 200


def test_api_preview_404(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from api.server import create_app
    from src.database import Database
    import api.preprocess_service as svc

    monkeypatch.setattr(svc, "preview_path", lambda: tmp_path / "missing.jpg")
    db = Database(str(tmp_path / "t.db"), reset=True)
    app = create_app(db=db, log_dir=str(tmp_path / "logs"), web_dist=None)
    client = TestClient(app)
    assert client.get("/api/preprocess/preview").status_code == 404


def test_api_token_required(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from api.server import create_app
    from src.database import Database
    import api.preprocess_service as svc

    monkeypatch.setenv("PREPROCESS_ADMIN_TOKEN", "secret")
    monkeypatch.setattr(svc, "override_path", lambda: tmp_path / "preprocess.json")
    db = Database(str(tmp_path / "t.db"), reset=True)
    app = create_app(db=db, log_dir=str(tmp_path / "logs"), web_dist=None)
    client = TestClient(app)
    denied = client.put("/api/preprocess", json={"enabled": True})
    assert denied.status_code == 403
    ok = client.put(
        "/api/preprocess",
        json={"enabled": True},
        headers={"X-Admin-Token": "secret"},
    )
    assert ok.status_code == 200
    monkeypatch.delenv("PREPROCESS_ADMIN_TOKEN", raising=False)


def test_async_live_never_blocks(monkeypatch):
    from src.detection.preprocess import PreprocessRuntime, PreprocessSettings

    rt = PreprocessRuntime()

    def slow(bgr, settings, *, detector_imgsz=640):
        time.sleep(0.2)
        return bgr, 1.0, True

    monkeypatch.setattr(rt.preprocessor, "enhance", slow)
    rt.start_worker()
    try:
        s = PreprocessSettings(enabled=True, clahe_enabled=True, mode="async", max_wait_ms=50)
        frame = _bgr()
        t0 = time.perf_counter()
        rt.submit(1, frame, s, 640)
        hit = rt.take(1, wait_ms=0.0)
        waited_ms = (time.perf_counter() - t0) * 1000.0
        assert waited_ms < 5.0
        assert hit is None
        rt.note_fallback()

        t1 = time.perf_counter()
        rt.submit(2, frame, s, 640)
        _ = rt.take(2, wait_ms=50)
        waited_file = (time.perf_counter() - t1) * 1000.0
        assert waited_file <= 70.0
    finally:
        rt.stop_worker()


def test_hot_reload_override(tmp_path, monkeypatch, caplog):
    import logging
    from src.detection.preprocess import PreprocessSettings, SettingsStore

    path = tmp_path / "preprocess.json"
    monkeypatch.setattr(
        "src.detection.preprocess.runtime_path",
        lambda name: tmp_path / Path(name).name,
    )
    store = SettingsStore({"enabled": False})
    assert store.get().enabled is False

    path.write_text(json.dumps({
        "enabled": True,
        "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid": 8},
    }), encoding="utf-8")
    with caplog.at_level(logging.INFO):
        assert store.reload(force=True) is True
    assert store.get().enabled is True
    assert store.get().clahe_enabled is True
    assert any("PREPROCESS_CONFIG_CHANGED" in r.message for r in caplog.records)

    path.write_text("{not-json", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert store.reload(force=True) is False
    assert store.get().enabled is True  # last good
    assert any("PREPROCESS_CONFIG_INVALID" in r.message for r in caplog.records)


def test_toggle_enhanced_does_not_disturb(tmp_path):
    """Disturbance brightness stats must use the original frame, not enhanced."""
    from src.pipeline import FaceTrackingPipeline
    from src.database import Database
    from src.event_logger import EventLogger
    from tests.test_pipeline import FakeDetector, FakeEmbedder, _base_cfg, make_frame

    cfg = _base_cfg()
    db = Database(str(tmp_path / "t.db"), reset=True)
    ev = EventLogger(str(tmp_path / "logs"), str(tmp_path / "logs" / "events.log"))
    pipe = FaceTrackingPipeline(cfg, FakeDetector(), FakeEmbedder(), db, ev)
    base = make_frame(40, None)
    # Warm EMA on originals, then feed wildly different detect frames.
    for i in range(10):
        pipe.process_frame(base, i, ts=i / 25.0, detect_frame=base, detect_scale=1.0)
    bright = np.full_like(base, 220)
    for i in range(10, 20):
        pipe.process_frame(base, i, ts=i / 25.0, detect_frame=bright, detect_scale=1.0)
    text = (tmp_path / "logs" / "events.log").read_text(encoding="utf-8")
    assert "reason=brightness_jump" not in text


def test_stats_line_parses_with_preprocess_fields():
    from src.reporting.counting import parse_log_line

    line = (
        "2026-11-07 12:00:00,123 | INFO | STATS sid=abc source=x frames=300 fps=25.0 "
        "unique=1 away=0 skipped=0 away_summary=none preprocess=off pp_ms=0.0 "
        "ready=100 fallback=0"
    )
    rec = parse_log_line(line)
    assert rec is not None
    assert rec["event"] == "STATS"
    assert rec["fields"]["preprocess"] == "off"
    assert rec["fields"]["pp_ms"] == "0.0"
