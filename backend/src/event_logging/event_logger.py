"""events.log writer + crop image store + session helpers.

Mandatory critical events (always INFO in ``events.log``):
  TRACK_STARTED / TRACK_ENDED   — face tracking lifecycle
  EMBEDDING_GENERATED           — ArcFace vector produced
  FACE_REGISTERED               — new unique identity
  FACE_RECOGNIZED               — gallery match of a known face
  ENTRY (+ NEW_ENTRY | RE_ENTRY)— visit open (first time vs returning)
  EXIT                          — visit close
  COUNT_SNAPSHOT                — unique / entry / exit totals
"""
from __future__ import annotations

import logging
import os
import re
import secrets
import shutil
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Set

import cv2
import numpy as np

from src.core import clock as _clock

# Events that belong in the away log when separate_away_log is enabled.
AWAY_EVENTS = {
    "AWAY_TRACK_STARTED", "AWAY_TRACK_ENDED", "AWAY_ENTRY", "AWAY_EXIT",
    "AWAY_EXIT_PENDING", "AWAY_RESUMED", "AWAY_DUPLICATE_TRACK", "AWAY_APPEARANCE",
    "AWAY_FACE_LINK", "FACE_CONTINUED_AS_AWAY",
}

# DEBUG-only pipeline chatter (suppressed at INFO).
DEBUG_EVENTS = {
    "FACE_ROI_REFINE", "FACE_RETINA_ASSIST", "RETINA_FACE_READY", "RETINA_FACE_UNAVAILABLE",
    "EMBED_NO_FACE", "TRACK_LOST", "TRACK_DROPPED",
}

# Documented critical face-visit events (grep aid; not filtered).
CRITICAL_EVENTS = frozenset({
    "TRACK_STARTED", "TRACK_ENDED",
    "EMBEDDING_GENERATED", "FACE_REGISTERED", "FACE_RECOGNIZED",
    "ENTRY", "NEW_ENTRY", "RE_ENTRY", "EXIT",
    "COUNT_SNAPSHOT",
})


def new_session_id(when: Optional[datetime] = None) -> str:
    dt = when or _clock.wall_now()
    return f"{dt.strftime('%Y%m%dT%H%M%S')}_{secrets.token_hex(2)}"


def _rel_path(path: str) -> str:
    """Store paths relative with forward slashes; never truncate."""
    if not path:
        return path
    try:
        p = Path(path)
        if p.is_absolute():
            try:
                from src.core.paths import PROJECT_ROOT
                p = p.relative_to(PROJECT_ROOT)
            except ValueError:
                pass
        return p.as_posix()
    except Exception:  # noqa: BLE001
        return str(path).replace("\\", "/")


class EventLogger:
    """Writes `logs/events.log` (flushed on every line) and saves event crops.

    Every line automatically gets `sid=<session_id>` as the first key=value.
    """

    def __init__(
        self,
        log_dir: str,
        events_log: str,
        *,
        session_id: Optional[str] = None,
        level: str = "INFO",
        separate_away_log: bool = True,
        debug_events: bool = False,
        max_bytes: int = 50_000_000,
        backup_count: int = 10,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events_log = Path(events_log)
        self.events_log.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or new_session_id()
        self.level_name = (level or "INFO").upper()
        self.separate_away_log = bool(separate_away_log)
        self.debug_events = bool(debug_events)
        self._min_level = logging.DEBUG if self.level_name == "DEBUG" else logging.INFO

        self._log = self._bind_logger(
            "events", self.events_log, max_bytes=max_bytes, backup_count=backup_count)
        self._away_log = None
        if self.separate_away_log:
            away_path = self.log_dir / "away_events.log"
            self._away_log = self._bind_logger(
                "away_events", away_path, max_bytes=max_bytes, backup_count=backup_count)

        # Tracks that emitted TRACK_STARTED / AWAY_TRACK_STARTED (paired ENDED).
        self.started_tracks: Set[int] = set()
        self.started_away_tracks: Set[int] = set()
        self.away_stats = {"tracks": 0, "entries": 0, "exits": 0}
        self._system_stopped = False

    @staticmethod
    def _bind_logger(name: str, path: Path, *, max_bytes: int, backup_count: int) -> logging.Logger:
        path.parent.mkdir(parents=True, exist_ok=True)
        log = logging.getLogger(name)
        log.setLevel(logging.DEBUG)
        log.propagate = False
        for h in list(log.handlers):
            log.removeHandler(h)
            h.close()
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
        # Large rotation so SYSTEM_START of the active session is never rotated away mid-run.
        fh = RotatingFileHandler(str(path), maxBytes=max_bytes, backupCount=backup_count,
                                 encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        # Console stays quieter for away logger.
        if name == "away_events":
            sh.setLevel(logging.WARNING)
        log.addHandler(fh)
        log.addHandler(sh)
        return log

    def archive_previous(self) -> Optional[Path]:
        """Move existing events.log to logs/archive/events_<timestamp>.log if non-empty."""
        archived = self._archive_file(self.events_log, "events", self._log, "events")
        if archived is not None:
            self._log = self._bind_logger(
                "events", self.events_log, max_bytes=50_000_000, backup_count=10)
        return archived

    def archive_away_log(self) -> Optional[Path]:
        """Move away_events.log to logs/archive/away_events_<timestamp>.log if non-empty."""
        away_path = self.log_dir / "away_events.log"
        if self._away_log is None:
            # Still archive a leftover file even if this session disabled the away logger.
            return self._archive_file(away_path, "away_events", None, None)
        archived = self._archive_file(away_path, "away_events", self._away_log, "away_events")
        if archived is not None:
            self._away_log = self._bind_logger(
                "away_events", away_path, max_bytes=50_000_000, backup_count=10)
        return archived

    def _archive_file(
        self,
        path: Path,
        prefix: str,
        logger: Optional[logging.Logger],
        rebind_name: Optional[str],
    ) -> Optional[Path]:
        if not path.exists() or path.stat().st_size == 0:
            return None
        archive_dir = self.log_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = _clock.wall_now().strftime("%Y%m%dT%H%M%S")
        dest = archive_dir / f"{prefix}_{stamp}.log"
        if logger is not None:
            for h in list(logger.handlers):
                logger.removeHandler(h)
                h.close()
        shutil.move(str(path), str(dest))
        return dest

    def log(self, event: str, level: int = logging.INFO, **fields) -> None:
        # Route AWAY_* to the away log.
        if event in AWAY_EVENTS and self._away_log is not None:
            if event == "AWAY_TRACK_STARTED":
                tid = fields.get("track")
                if tid is not None:
                    try:
                        self.started_away_tracks.add(int(tid))
                    except (TypeError, ValueError):
                        pass
                self.away_stats["tracks"] += 1
            elif event == "AWAY_TRACK_ENDED":
                tid = fields.get("track")
                try:
                    tid_i = int(tid) if tid is not None else None
                except (TypeError, ValueError):
                    tid_i = None
                # Tentative away tracks never confirmed → drop silently (matches TRACK_ENDED).
                if tid_i is None or tid_i not in self.started_away_tracks:
                    return
                self.started_away_tracks.discard(tid_i)
            elif event == "AWAY_ENTRY":
                self.away_stats["entries"] += 1
            elif event == "AWAY_EXIT":
                self.away_stats["exits"] += 1
            self._emit(self._away_log, event, level, fields)
            return

        # DEBUG-only events suppressed unless enabled.
        if event in DEBUG_EVENTS and not self.debug_events and self._min_level > logging.DEBUG:
            if level < logging.WARNING:
                return

        if level < self._min_level and level < logging.WARNING:
            return

        # TRACK_ENDED only for tracks that had TRACK_STARTED.
        if event == "TRACK_STARTED":
            tid = fields.get("track")
            if tid is not None:
                try:
                    self.started_tracks.add(int(tid))
                except (TypeError, ValueError):
                    pass
        if event == "TRACK_ENDED":
            tid = fields.get("track")
            try:
                tid_i = int(tid) if tid is not None else None
            except (TypeError, ValueError):
                tid_i = None
            if tid_i is None or tid_i not in self.started_tracks:
                # Demote to DEBUG TRACK_DROPPED for tentative tracks.
                if self.debug_events or self._min_level <= logging.DEBUG:
                    self._emit(self._log, "TRACK_DROPPED", logging.DEBUG, fields)
                return
            self.started_tracks.discard(tid_i)

        self._emit(self._log, event, level, fields)

    def _emit(self, logger: logging.Logger, event: str, level: int, fields: dict) -> None:
        cleaned = {}
        for k, v in fields.items():
            if v is None:
                continue
            if k in ("image", "path") and isinstance(v, str):
                cleaned[k] = _rel_path(v)
            else:
                # Values must contain no spaces (log format contract).
                cleaned[k] = str(v).replace(" ", "_")
        # sid is always the first key=value.
        parts = [f"sid={self.session_id}"]
        parts.extend(f"{k}={v}" for k, v in cleaned.items())
        extras = " ".join(parts)
        logger.log(level, "%-20s %s", event, extras)
        for h in logger.handlers:
            h.flush()

    def away_summary(self) -> str:
        s = self.away_stats
        return f"away_tracks={s['tracks']} away_entries={s['entries']} away_exits={s['exits']}"

    def save_crop(self, kind: str, obj_id: int, ts: datetime, crop: Optional[np.ndarray],
                  id_prefix: str = "face") -> Optional[str]:
        """kind: entry|exit|face|away_entry|away_exit|away. Atomic write (tmp + rename)."""
        if crop is None or crop.size == 0:
            return None
        folders = {
            "entry": "entries", "exit": "exits", "face": "faces",
            "appearance": "appearance",
            "away_entry": "away_entries", "away_exit": "away_exits", "away": "away_people",
        }
        folder = self.log_dir / folders[kind] / ts.strftime("%Y-%m-%d")
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{kind}_{id_prefix}{obj_id:04d}_{ts.strftime('%H%M%S_%f')}.jpg"
        final, tmp = folder / name, folder / ("tmp_" + name)
        if not cv2.imwrite(str(tmp), crop):
            self.log("IMAGE_WRITE_FAILED", logging.ERROR, path=str(final))
            return None
        os.replace(tmp, final)
        return _rel_path(str(final))
