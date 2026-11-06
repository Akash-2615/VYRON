"""events.log writer + crop image store."""
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class EventLogger:
    """Writes `logs/events.log` (flushed on every line) and saves event crops to
    `logs/entries/YYYY-MM-DD/` or `logs/exits/YYYY-MM-DD/`."""

    def __init__(self, log_dir: str, events_log: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        Path(events_log).parent.mkdir(parents=True, exist_ok=True)

        self._log = logging.getLogger("events")
        self._log.setLevel(logging.INFO)
        self._log.propagate = False
        for h in list(self._log.handlers):          # rebind so a new instance never writes to a stale file
            self._log.removeHandler(h)
            h.close()
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
        fh = RotatingFileHandler(events_log, maxBytes=10_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        self._log.addHandler(fh)
        self._log.addHandler(sh)

    def log(self, event: str, level: int = logging.INFO, **fields) -> None:
        extras = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
        self._log.log(level, "%-20s %s", event, extras)
        for h in self._log.handlers:
            h.flush()

    def save_crop(self, kind: str, obj_id: int, ts: datetime, crop: Optional[np.ndarray],
                  id_prefix: str = "face") -> Optional[str]:
        """kind: entry|exit|face|away_entry|away_exit|away. Atomic write (tmp + rename)."""
        if crop is None or crop.size == 0:
            return None
        folders = {
            "entry": "entries", "exit": "exits", "face": "faces",
            "away_entry": "away_entries", "away_exit": "away_exits", "away": "away_people",
        }
        folder = self.log_dir / folders[kind] / ts.strftime("%Y-%m-%d")
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{kind}_{id_prefix}{obj_id:04d}_{ts.strftime('%H%M%S_%f')}.jpg"
        final, tmp = folder / name, folder / ("tmp_" + name)
        if not cv2.imwrite(str(tmp), crop):
            self.log("IMAGE_WRITE_FAILED", logging.ERROR, path=final)
            return None
        os.replace(tmp, final)
        return str(final)
