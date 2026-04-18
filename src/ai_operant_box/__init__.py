"""ai-operant-box — LLM-as-subject operant chamber simulator.

Public re-exports cover the most commonly used building blocks: the abstract
``Subject`` base, the deterministic numerical agents, the single- and dyadic
chambers, and the schedule families. The LLM-backed subject (which depends on
the optional ``anthropic`` SDK) is *not* re-exported at top level — import it
explicitly from :mod:`ai_operant_box.agent.llm_agent` after installing the
``[llm]`` extra.
"""

from __future__ import annotations

from ai_operant_box.agent.base import Subject
from ai_operant_box.agent.induction_agent import InductionAgent, InductionSubject
from ai_operant_box.agent.scallop_agent import ScallopingAgent
from ai_operant_box.chamber.environment import OperantBox, ResponseRecord, VirtualClock
from ai_operant_box.chamber.interdependent import (
    InterdependentSchedule as DyadicInterdependentSchedule,
)
from ai_operant_box.chamber.interdependent import LeaderFollowerBox, SocialEvent
from ai_operant_box.chamber.schedule import (
    ContinuousReinforcement,
    FixedInterval,
    ReinforcementSchedule,
)

__all__ = [
    "ContinuousReinforcement",
    "DyadicInterdependentSchedule",
    "FixedInterval",
    "InductionAgent",
    "InductionSubject",
    "LeaderFollowerBox",
    "OperantBox",
    "ReinforcementSchedule",
    "ResponseRecord",
    "ScallopingAgent",
    "SocialEvent",
    "Subject",
    "VirtualClock",
]
