"""VI / RI read the same virtual clock as FI, never the wall clock."""

from __future__ import annotations

from unittest.mock import patch

from ai_operant_box.chamber.environment import OperantBox, VirtualClock
from ai_operant_box.schedules.interval import (
    RandomIntervalSchedule,
    VariableIntervalSchedule,
)


def _vi(clock: VirtualClock, intervals: list[float]) -> VariableIntervalSchedule:
    with patch(
        "ai_operant_box.schedules.interval._exponential_sequence",
        return_value=intervals,
    ):
        return VariableIntervalSchedule(mean_interval=5.0, clock=clock)


def test_vi_is_satisfied_by_advancing_the_virtual_clock_twice() -> None:
    clock = VirtualClock()
    schedule = _vi(clock, [5.0, 10.0])
    clock.advance(3.0)
    assert schedule.decision(response_made=True) is False
    clock.advance(2.0)
    assert schedule.decision(response_made=True) is True
    assert schedule.last_reinforcement_time == 5.0
    assert schedule.get_time_until_next_reinforcement() == 10.0


def test_vi_ignores_wall_clock_time_spent_waiting_for_the_subject() -> None:
    clock = VirtualClock()
    with patch("time.monotonic", side_effect=[0.0, 1000.0, 2000.0, 3000.0]):
        schedule = _vi(clock, [5.0])
        assert schedule.decision(response_made=True) is False
        assert schedule.get_time_until_next_reinforcement() == 5.0


def test_vi_accepts_the_time_from_the_caller_like_fi() -> None:
    schedule = _vi(VirtualClock(), [5.0, 10.0])
    assert schedule.is_reinforced("Lever Press", 4.0) is False
    assert schedule.is_reinforced("Explore", 6.0) is False
    assert schedule.is_reinforced("Lever Press", 6.0) is True
    assert schedule.is_reinforced("Lever Press", 15.0) is False
    assert schedule.is_reinforced("Lever Press", 16.0) is True


class _AlwaysPress:
    def sense(self, sd: object) -> None:
        pass

    def act(self) -> str:
        return "Lever Press"

    def learn(self, sr: object) -> None:
        pass


def test_vi_in_operant_box_follows_the_box_clock() -> None:
    clock = VirtualClock()
    schedule = _vi(clock, [3.0, 2.0])
    box = OperantBox(subject=_AlwaysPress(), schedule=schedule, clock=clock)
    box.run(duration=6.0, delta=1.0)
    reinforced = [r.timestamp for r in box.records if r.is_reinforced]
    assert reinforced == [3.0, 5.0]


def test_ri_availability_depends_only_on_virtual_elapsed_time() -> None:
    clock = VirtualClock()
    with patch("time.monotonic", side_effect=[0.0, 1000.0, 2000.0]):
        schedule = RandomIntervalSchedule(probability_per_unit_time=0.1, clock=clock)
        with patch("random.random", return_value=0.0):
            # No virtual time has passed: no chance of setting up a reinforcer.
            assert schedule.decision(response_made=True) is False
    clock.advance(1.0)
    with patch("random.random", return_value=0.05):
        assert schedule.decision(response_made=True) is True


def test_ri_accepts_the_time_from_the_caller_like_fi() -> None:
    schedule = RandomIntervalSchedule(probability_per_unit_time=0.1, clock=VirtualClock())
    with patch("random.random", return_value=0.05):
        assert schedule.is_reinforced("Explore", 1.0) is False
        assert schedule.is_reinforced("Lever Press", 1.0) is True
