"""Visit / presence state machine (time-based grace, disturbance freeze)."""
from .manager import (
    STATE_BODY_ONLY,
    STATE_EXITED,
    STATE_HIDDEN,
    STATE_PRESENT,
    PresenceManager,
    Visit,
    in_exit_zone,
    moving_outward,
    near_edge,
)

__all__ = [
    "STATE_BODY_ONLY",
    "STATE_EXITED",
    "STATE_HIDDEN",
    "STATE_PRESENT",
    "PresenceManager",
    "Visit",
    "in_exit_zone",
    "moving_outward",
    "near_edge",
]
