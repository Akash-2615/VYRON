"""Session event logging and crop image store.

Must not import pipeline or api. Depends only on core (clock).
"""
from .event_logger import (
    AWAY_EVENTS,
    DEBUG_EVENTS,
    EventLogger,
    new_session_id,
)

__all__ = [
    "AWAY_EVENTS",
    "DEBUG_EVENTS",
    "EventLogger",
    "new_session_id",
]
