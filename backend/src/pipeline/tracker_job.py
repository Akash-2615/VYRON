"""Background RTSP/file tracker job controlled by the dashboard API.

One job at a time. Supports RTSP, a single local file, or a local playlist
(``video_sources`` from config — play once each, shared unique count).
Uses the same pipeline as ``main.py`` / ``run_rtsp.py``.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger(__name__)


def _normalize_rtsp(url: str) -> str:
    u = (url or "").strip().strip('"').strip("'")
    if not u:
        raise ValueError("URL is empty")
    low = u.lower()
    if not (low.startswith("rtsp://") or low.startswith("rtsps://")):
        raise ValueError("RTSP mode requires a URL starting with rtsp:// or rtsps://")
    return u


def _normalize_local(path: str) -> str:
    u = (path or "").strip().strip('"').strip("'")
    if not u:
        raise ValueError("Local video path is empty")
    p = Path(u).expanduser()
    if not p.is_file():
        raise ValueError(f"Local video not found: {u}")
    return str(p.resolve())


def _normalize_webcam(device: str = "0", *, probe: bool = True) -> str:
    """Webcam device index as a digit string (OpenCV ``VideoCapture(int)``)."""
    d = (device or "0").strip().strip('"').strip("'")
    if d.lower().startswith("webcam:"):
        d = d.split(":", 1)[1].strip()
    if not d.isdigit():
        raise ValueError("Webcam device must be an integer index (0, 1, …)")
    if probe:
        try:
            import cv2
            cap = cv2.VideoCapture(int(d))
            opened = bool(cap.isOpened())
            got = False
            if opened:
                got, _ = cap.read()
            cap.release()
            if not opened or not got:
                raise ValueError(
                    f"Cannot open webcam {d} — grant camera permission to the "
                    "terminal/Python app, or try another device index")
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Webcam {d} probe failed: {exc}") from exc
    return d


_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


def list_local_videos(config_path: str = "config.json") -> List[Dict[str, Any]]:
    """Return local videos for the dashboard picker.

    Includes ``video_sources`` / ``video_source`` from config, plus any video files
    found under common folders (``data/videos/``, ``data/``, ``videos/``).
    """
    from src.core.config import load_config
    cfg = load_config(config_path)
    from src.core.paths import PROJECT_ROOT
    root = PROJECT_ROOT
    raw: List[str] = []
    sources = cfg.get("video_sources") or []
    if isinstance(sources, list):
        raw.extend(str(s) for s in sources if s)
    vs = cfg.get("video_source")
    if vs and str(vs) not in raw:
        raw.insert(0, str(vs))

    for folder in ("data/videos", "data", "videos", "samples", "Video Datasets"):
        d = root / folder
        if not d.is_dir():
            continue
        try:
            files = sorted(
                p for p in d.iterdir()
                if p.is_file() and p.suffix.lower() in _VIDEO_EXTS
            )
        except OSError:
            continue
        for p in files:
            try:
                rel = str(p.relative_to(root))
            except ValueError:
                rel = str(p)
            if rel not in raw:
                raw.append(rel)

    seen: set = set()
    out: List[Dict[str, Any]] = []
    for rel in raw:
        key = rel.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        p = Path(key).expanduser()
        if not p.is_file():
            p = root / key
        exists = p.is_file()
        out.append({
            "path": key,
            "name": Path(key).name,
            "exists": exists,
            "size_bytes": p.stat().st_size if exists else None,
        })
    return out


class TrackerJob:
    """Singleton-style worker: start / stop / status for live tracking."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = {"flag": False}
        self._active_source = None  # FrameSource; interrupted on Stop so RTSP unblocks
        self._state: Dict[str, Any] = {
            "running": False,
            "starting": False,
            "stopping": False,
            "url": None,
            "mode": None,
            "playlist": None,
            "sources": [],
            "source_index": 0,
            "source_total": 0,
            "error": None,
            "started_at": None,
            "frames": 0,
            "unique_visitors": None,
            "backend": None,
        }

    def status(self) -> Dict[str, Any]:
        with self._lock:
            alive = self._thread is not None and self._thread.is_alive()
            out = dict(self._state)
            out["thread_alive"] = alive
            stopping = bool(out.get("stopping"))
            if not alive:
                # Reclaim finished worker so Start is not blocked by a dead Thread object.
                self._thread = None
                out["running"] = False
                out["starting"] = False
                out["stopping"] = False
                self._state["stopping"] = False
            elif stopping:
                # Stop requested — do not keep the UI stuck on "Streaming".
                out["running"] = False
                out["starting"] = False
            else:
                # Thread still alive — UI must show Stop while active/starting.
                out["running"] = bool(out.get("running") or not out.get("starting"))
                out["starting"] = bool(out.get("starting"))
                if not out["running"] and not out["starting"]:
                    out["running"] = True
            return out

    def _bind_source(self, src) -> None:
        with self._lock:
            self._active_source = src

    def _interrupt_source(self) -> None:
        with self._lock:
            src = self._active_source
        if src is None:
            return
        try:
            src.interrupt()
        except Exception:  # noqa: BLE001
            log.exception("FrameSource.interrupt failed")

    def _set_progress(self, frames: int) -> None:
        with self._lock:
            self._state["frames"] = int(frames)

    def _reclaim_worker(self, wait_sec: float = 2.0) -> bool:
        """Signal stop and drop a hung worker reference if it will not exit.

        Returns True when no live worker remains (safe to start).
        """
        with self._lock:
            t = self._thread
            if t is None or not t.is_alive():
                self._thread = None
                self._active_source = None
                self._state["running"] = False
                self._state["starting"] = False
                self._state["stopping"] = False
                return True
            self._stop["flag"] = True
            # Flip UI out of Streaming immediately (join may still take a moment).
            self._state["running"] = False
            self._state["starting"] = False
            self._state["stopping"] = True
        # Unblock RTSP/webcam VideoCapture.read() waiting inside the worker.
        self._interrupt_source()
        t.join(timeout=wait_sec)
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                # Webcam/OpenCV can block forever on release; abandon so Start + clear work.
                log.warning("tracker worker did not exit after stop; reclaiming slot")
                self._thread = None
                self._active_source = None
                self._state["running"] = False
                self._state["starting"] = False
                self._state["stopping"] = False
                self._state["error"] = "Previous tracker was force-stopped (hung worker)."
                return True
            self._thread = None
            self._active_source = None
            self._state["running"] = False
            self._state["starting"] = False
            self._state["stopping"] = False
            return True

    def start(
        self,
        url: str = "",
        config_path: str = "config.json",
        reset: bool = False,
        mode: str = "rtsp",
        playlist: str = "one",
        sources: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        mode_l = (mode or "rtsp").strip().lower()
        if mode_l in ("file", "video", "mp4"):
            mode_l = "local"
        if mode_l in ("cam", "camera", "device"):
            mode_l = "webcam"
        playlist_l = (playlist or "one").strip().lower()
        if playlist_l in ("all", "playlist", "sources"):
            playlist_l = "all"
        else:
            playlist_l = "one"

        resolved: List[str] = []
        if mode_l == "rtsp":
            resolved = [_normalize_rtsp(url)]
            playlist_l = "one"
        elif mode_l == "webcam":
            # Skip exclusive probe — opening the cam twice races Stop/Start on macOS
            # and can leave the UI "Streaming" with a stuck/black capture.
            resolved = [_normalize_webcam(url or "0", probe=False)]
            playlist_l = "one"
        elif mode_l == "local":
            if playlist_l == "all":
                vids = list_local_videos(config_path)
                resolved = [_normalize_local(v["path"]) for v in vids if v.get("exists")]
                if not resolved:
                    raise ValueError(
                        "No local videos found from config video_sources / video_source")
            elif sources:
                resolved = [_normalize_local(s) for s in sources if str(s).strip()]
                if not resolved:
                    raise ValueError("No valid local videos in sources list")
            else:
                resolved = [_normalize_local(url)]
        else:
            raise ValueError("mode must be 'rtsp', 'local', or 'webcam'")

        with self._lock:
            alive = self._thread is not None and self._thread.is_alive()
            active = alive and bool(self._state.get("running") or self._state.get("starting"))
        if active:
            raise RuntimeError(
                "Tracker already running. Stop it before starting a new stream.")
        if alive:
            # Stale worker (Stop timed out / finally cleared flags) — reclaim then start.
            self._reclaim_worker(wait_sec=2.0)

        with self._lock:
            if self._thread and self._thread.is_alive():
                raise RuntimeError(
                    "Tracker already running. Stop it before starting a new stream.")
            self._stop = {"flag": False}
            self._state.update({
                "running": False,
                "starting": True,
                "stopping": False,
                "url": resolved[0],
                "mode": mode_l,
                "playlist": playlist_l,
                "sources": list(resolved),
                "source_index": 1,
                "source_total": len(resolved),
                "error": None,
                "started_at": time.time(),
                "frames": 0,
                "unique_visitors": None,
                "backend": None,
            })
            self._thread = threading.Thread(
                target=self._run,
                args=(resolved, config_path, reset, mode_l, playlist_l),
                name="facetrack-job",
                daemon=True,
            )
            self._thread.start()
        return self.status()

    def stop(self) -> Dict[str, Any]:
        # Short join: interrupt() should unblock RTSP quickly; reclaim if still hung.
        self._reclaim_worker(wait_sec=3.0)
        return self.status()

    def _run(
        self,
        sources: List[str],
        config_path: str,
        reset: bool,
        mode: str,
        playlist: str,
    ) -> None:
        db = None
        events = None
        job_mode = f"dashboard_{mode}"
        try:
            from src.core.config import load_config
            from src.pipeline import FaceTrackingPipeline
            from src.session_svc import start_session, stop_session
            from src.storage import log_db_open, open_database
            # Load backend/main.py explicitly — repo-root main.py is a shim that
            # executes CLI main() and breaks when imported from a worker thread.
            import importlib.util
            from pathlib import Path as _Path
            _cli = _Path(__file__).resolve().parents[2] / "main.py"
            _spec = importlib.util.spec_from_file_location("facetrack_cli_main", _cli)
            if _spec is None or _spec.loader is None:
                raise ImportError(f"Cannot load CLI module at {_cli}")
            _cli_mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_cli_mod)
            run_source = _cli_mod.run_source

            cfg = load_config(config_path)
            cfg.setdefault("output", {})
            cfg["output"]["live_preview"] = cfg["output"].get("live_preview") or "data/runtime/live.jpg"
            cfg["output"]["display"] = False  # UI is the dashboard, not OpenCV
            cfg.setdefault("stream", {})

            st = dict(cfg["storage"])
            # Dashboard Start: keep faces + events.log unless the UI "clear" checkbox
            # sent reset=true. CLI still uses storage.reset_on_start / --fresh / --no-reset.
            do_reset = bool(reset)
            db, db_info = open_database(st, reset=do_reset)  # Postgres only

            with self._lock:
                self._state["backend"] = db.backend

            events = start_session(
                cfg, db, reset=do_reset, sources=sources,
                extra_fields={"mode": job_mode, "playlist": playlist, "backend": db.backend},
            )
            log_db_open(events, db_info)

            from src.detection import HybridFaceDetector, YoloFaceDetector
            from src.recognition import FaceEmbedder

            appearance = None
            if cfg.get("appearance", {}).get("enabled", False):
                try:
                    from src.recognition import AppearanceAnalyzer
                    appearance = AppearanceAnalyzer(cfg["appearance"])
                except Exception:  # noqa: BLE001
                    log.exception("Appearance model failed")

            person_detector = None
            if cfg.get("away_walkers", {}).get("enabled", False):
                try:
                    from src.detection import YoloPersonDetector
                    person_detector = YoloPersonDetector(cfg.get("person_detector", {}))
                except Exception:  # noqa: BLE001
                    log.exception("Person detector failed")

            retina_cfg = cfg.get("retina_face") or {}
            if retina_cfg.get("enabled", True):
                try:
                    detector = HybridFaceDetector(cfg["detector"], retina_cfg)
                    events.log("RETINA_FACE_READY")
                except Exception:  # noqa: BLE001
                    log.exception("RetinaFace unavailable")
                    detector = YoloFaceDetector(cfg["detector"])
            else:
                detector = YoloFaceDetector(cfg["detector"])

            embedder = FaceEmbedder(cfg["recognition"])
            pipeline = FaceTrackingPipeline(
                cfg, detector, embedder, db, events,
                appearance=appearance, person_detector=person_detector)

            with self._lock:
                self._state["starting"] = False
                self._state["running"] = True
                self._state["error"] = None

            out = cfg["output"]
            total_frames = 0
            for i, source in enumerate(sources, 1):
                if self._stop.get("flag"):
                    break
                with self._lock:
                    self._state["url"] = source
                    self._state["source_index"] = i
                    self._state["source_total"] = len(sources)
                events.log(
                    "SESSION_VIDEO", index=f"{i}/{len(sources)}", source=source,
                    unique_so_far=db.unique_count(),
                )
                total_frames += run_source(
                    source, cfg, pipeline, db, events, self._stop,
                    display=False,
                    max_frames=0,
                    out_path="",
                    live_path=out.get("live_preview") or "",
                    live_quality=int(out.get("live_jpeg_quality", 92)),
                    live_max_width=int(out.get("live_max_width", 0) or 0),
                    live_every=int(out.get("live_every_n_frames", 2)),
                    on_source=self._bind_source,
                    on_progress=self._set_progress,
                )

            try:
                pipeline.finalize()
            except Exception:  # noqa: BLE001
                log.exception("pipeline.finalize failed")
            if events is not None:
                stop_session(db, events, mode=job_mode, frames=total_frames, videos=len(sources))
            with self._lock:
                self._state["frames"] = total_frames
                self._state["unique_visitors"] = db.unique_count()
        except Exception as exc:  # noqa: BLE001
            log.exception("Tracker job failed")
            with self._lock:
                self._state["error"] = str(exc)
            if events is not None and db is not None:
                try:
                    stop_session(db, events, mode=job_mode, error=str(exc).replace(" ", "_")[:120])
                except Exception:  # noqa: BLE001
                    pass
        finally:
            with self._lock:
                self._active_source = None
            if db is not None:
                try:
                    db.close()
                except Exception:  # noqa: BLE001
                    pass
            with self._lock:
                self._state["running"] = False
                self._state["starting"] = False
                self._state["stopping"] = False


_JOB: Optional[TrackerJob] = None
_JOB_LOCK = threading.Lock()


def get_tracker_job() -> TrackerJob:
    global _JOB
    with _JOB_LOCK:
        if _JOB is None:
            _JOB = TrackerJob()
        return _JOB
