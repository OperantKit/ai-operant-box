"""Subject (virtual organism) implementations."""

from ai_operant_box.agent.base import Subject
from ai_operant_box.agent.induction_agent import InductionAgent, InductionSubject
from ai_operant_box.agent.matching_agent import MatchingAgent
from ai_operant_box.agent.scallop_agent import ScallopingAgent

__all__ = [
    "InductionAgent",
    "InductionSubject",
    "MatchingAgent",
    "ScallopingAgent",
    "Subject",
]
