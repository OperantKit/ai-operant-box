"""Operant chamber implementations (single-subject and dyadic)."""

from ai_operant_box.chamber.environment import OperantBox, ResponseRecord, VirtualClock
from ai_operant_box.chamber.interdependent import (
    InterdependentSchedule,
    LeaderFollowerBox,
    SocialEvent,
)
from ai_operant_box.chamber.schedule import (
    ContinuousReinforcement,
    Extinction,
    FixedInterval,
    ReinforcementSchedule,
)

__all__ = [
    "ContinuousReinforcement",
    "Extinction",
    "FixedInterval",
    "InterdependentSchedule",
    "LeaderFollowerBox",
    "OperantBox",
    "ReinforcementSchedule",
    "ResponseRecord",
    "SocialEvent",
    "VirtualClock",
]
