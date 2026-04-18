"""Frozen-dataclass interdependent schedule with explicit external state.

This is a pure-data variant suitable for serialisation and side-by-side
comparison with the leader-follower chamber in
:mod:`ai_operant_box.chamber.interdependent`. The chamber variant keeps
state internally; this variant keeps state in :class:`InterdependentState`,
which the caller advances explicitly.

Reference
---------
Benvenuti, M. F., et al. (2024). The role of schedules of reinforcement in
the experimental analysis of social behavior. *Perspectives on Behavior
Science*. https://doi.org/10.1007/s40614-024-00421-5
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InterdependentState:
    """Mutable state for one session of the interdependent schedule."""
    leader_response_time: float | None = None
    follower_response_time: float | None = None
    total_reinforcers: int = 0
    total_leader_responses: int = 0
    total_follower_responses: int = 0


@dataclass(frozen=True)
class InterdependentSchedule:
    """Pure-data interdependent dyadic reinforcement schedule.

    Reinforcement is delivered when the follower responds within
    ``time_window_sec`` of the leader. With ``require_order=True`` the
    follower must respond strictly after the leader.
    """

    time_window_sec: float = 5.0
    require_order: bool = True

    def __post_init__(self) -> None:
        if self.time_window_sec <= 0:
            raise ValueError(
                f"time_window_sec must be > 0, got {self.time_window_sec}"
            )

    def record_leader_response(
        self, state: InterdependentState, timestamp: float
    ) -> None:
        state.leader_response_time = timestamp
        state.total_leader_responses += 1

    def record_follower_response(
        self, state: InterdependentState, timestamp: float
    ) -> bool:
        state.follower_response_time = timestamp
        state.total_follower_responses += 1

        if state.leader_response_time is None:
            return False

        dt = timestamp - state.leader_response_time
        if self.require_order and dt < 0:
            return False

        if abs(dt) <= self.time_window_sec:
            state.total_reinforcers += 1
            state.leader_response_time = None
            state.follower_response_time = None
            return True

        return False

    def reinforcement_rate(
        self, state: InterdependentState, duration_sec: float
    ) -> float:
        if duration_sec <= 0:
            return 0.0
        return (state.total_reinforcers / duration_sec) * 60.0

    def follower_contingency_ratio(
        self, state: InterdependentState
    ) -> float | None:
        if state.total_follower_responses == 0:
            return None
        return state.total_reinforcers / state.total_follower_responses


__all__ = ["InterdependentSchedule", "InterdependentState"]
