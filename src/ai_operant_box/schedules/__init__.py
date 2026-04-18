"""Variable / random / interdependent schedule families.

These schedules carry their own state (Fleshler-Hoffman-derived ratios or
intervals, dyadic interdependent windows) and are independent from the
single-subject :mod:`ai_operant_box.chamber.schedule` module.
"""

from ai_operant_box.schedules.interdependent import (
    InterdependentSchedule,
    InterdependentState,
)
from ai_operant_box.schedules.interval import (
    RandomIntervalSchedule,
    VariableIntervalSchedule,
)
from ai_operant_box.schedules.ratio import RandomRatioSchedule, VariableRatioSchedule

__all__ = [
    "InterdependentSchedule",
    "InterdependentState",
    "RandomIntervalSchedule",
    "RandomRatioSchedule",
    "VariableIntervalSchedule",
    "VariableRatioSchedule",
]
