"""Single time source for the tracker.

wall_now()  — local wall-clock datetime for anything stored or shown to humans.
mono()      — time.monotonic() for DURATIONS / deadlines ONLY.

Convert a monotonic instant to wall time with wall_from_mono(). Never add a
monotonic value to wall time, and never format a monotonic value as a date.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import time
from typing import Optional


def wall_now() -> datetime:
    """Local naive wall-clock datetime (consistent convention across the project)."""
    return datetime.now()


def mono() -> float:
    """Monotonic seconds — for durations and presence deadlines only."""
    return time.monotonic()


def iso_ms(dt: Optional[datetime] = None) -> str:
    """ISO-8601 with milliseconds."""
    return (dt or wall_now()).isoformat(timespec="milliseconds")


def wall_from_mono(
    mono_instant: float,
    *,
    now_wall: Optional[datetime] = None,
    now_mono: Optional[float] = None,
) -> datetime:
    """Map a past/future monotonic instant onto wall time.

    wall = wall_now() - (mono() - mono_instant)
    """
    nw = now_wall if now_wall is not None else wall_now()
    nm = float(now_mono) if now_mono is not None else mono()
    return nw - timedelta(seconds=(nm - float(mono_instant)))


def is_anomalous(ts: datetime, *, max_skew_sec: float = 60.0,
                 now: Optional[datetime] = None) -> bool:
    """True if |ts - wall_now| exceeds max_skew_sec."""
    ref = now if now is not None else wall_now()
    return abs((ts - ref).total_seconds()) > float(max_skew_sec)
