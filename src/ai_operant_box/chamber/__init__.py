"""Operant chamber implementations (single-subject and dyadic)."""

from ai_operant_box.chamber.environment import OperantBox, ResponseRecord, VirtualClock
from ai_operant_box.chamber.interdependent import (
    InterdependentSchedule,
    LeaderFollowerBox,
    SocialEvent,
)
from ai_operant_box.chamber.mts import (
    MatchingToSampleBox,
    MatchingToSampleSubject,
    TrialOutcome,
    TrialRecord,
)
from ai_operant_box.chamber.schedule import (
    ContinuousReinforcement,
    DifferentialReinforcementOfLowRate,
    Extinction,
    FixedInterval,
    ReinforcementSchedule,
)

__all__ = [
    "ContinuousReinforcement",
    "DifferentialReinforcementOfLowRate",
    "Extinction",
    "FixedInterval",
    "InterdependentSchedule",
    "LeaderFollowerBox",
    "MatchingToSampleBox",
    "MatchingToSampleSubject",
    "OperantBox",
    "ReinforcementSchedule",
    "ResponseRecord",
    "SocialEvent",
    "TrialOutcome",
    "TrialRecord",
    "VirtualClock",
]
