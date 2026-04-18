"""Tests for the chamber.interdependent leader-follower box."""

from __future__ import annotations

from typing import Any, List, Optional

from ai_operant_box.agent.base import Subject
from ai_operant_box.chamber.interdependent import (
    InterdependentSchedule,
    LeaderFollowerBox,
    SocialEvent,
)


class AlwaysRespondingSubject(Subject):
    def __init__(self) -> None:
        self._history: List[dict] = []
        self.last_sr: Any = None

    def sense(self, discriminative_stimulus: Any) -> None:  # noqa: D401
        self._history.append({"event": "sense", "sd": discriminative_stimulus})

    def act(self) -> str:
        return "press"

    def learn(self, reinforcer: Optional[Any]) -> None:
        self.last_sr = reinforcer
        self._history.append({"event": "learn", "sr": reinforcer})

    @property
    def history(self) -> List[dict]:
        return self._history


def test_interdependent_schedule_within_window():
    schedule = InterdependentSchedule(window=5.0, min_irt=0.0)
    schedule.note_leader_response(current_time=10.0)
    reinforced, irt = schedule.evaluate_follower_response(current_time=12.0)
    assert reinforced is True
    assert irt == 2.0


def test_interdependent_schedule_outside_window():
    schedule = InterdependentSchedule(window=5.0)
    schedule.note_leader_response(current_time=0.0)
    reinforced, irt = schedule.evaluate_follower_response(current_time=20.0)
    assert reinforced is False


def test_interdependent_schedule_invalid_args():
    import pytest

    with pytest.raises(ValueError):
        InterdependentSchedule(window=0.0)
    with pytest.raises(ValueError):
        InterdependentSchedule(window=1.0, min_irt=-0.1)


def test_leader_follower_box_runs_and_records_events():
    leader = AlwaysRespondingSubject()
    follower = AlwaysRespondingSubject()
    schedule = InterdependentSchedule(window=2.0, min_irt=0.0)
    box = LeaderFollowerBox(
        leader=leader,
        follower=follower,
        schedule=schedule,
        step_duration=1.0,
    )

    log = box.run(n_steps=4)
    assert len(log) == 4
    assert all(isinstance(e, SocialEvent) for e in log)
    # Both subjects respond every step → at least one reinforcement.
    assert box.reinforcement_rate() > 0.0
    leader_rate, follower_rate = box.dyadic_response_rate()
    assert leader_rate == 1.0
    assert follower_rate == 1.0


def test_leader_follower_box_synchrony_ratio():
    leader = AlwaysRespondingSubject()
    follower = AlwaysRespondingSubject()
    schedule = InterdependentSchedule(window=5.0)
    box = LeaderFollowerBox(leader, follower, schedule, step_duration=1.0)
    box.run(n_steps=3)
    assert box.synchrony_ratio() == 1.0


def test_leader_follower_box_reset_clears_log():
    leader = AlwaysRespondingSubject()
    follower = AlwaysRespondingSubject()
    schedule = InterdependentSchedule(window=5.0)
    box = LeaderFollowerBox(leader, follower, schedule, step_duration=1.0)
    box.run(n_steps=2)
    assert len(box.log) == 2
    box.reset()
    assert len(box.log) == 0
    assert schedule.pending_leader is False
