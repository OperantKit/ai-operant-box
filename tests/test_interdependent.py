"""Tests for the schedules.interdependent dyadic reinforcement schedule.

Reference: Benvenuti, M. F., et al. (2024). The role of schedules of
reinforcement in the experimental analysis of social behavior. *Perspectives
on Behavior Science*. https://doi.org/10.1007/s40614-024-00421-5
"""

from __future__ import annotations

import pytest

from ai_operant_box.schedules.interdependent import (
    InterdependentSchedule,
    InterdependentState,
)


def make_schedule(time_window_sec: float = 5.0, require_order: bool = True):
    schedule = InterdependentSchedule(
        time_window_sec=time_window_sec, require_order=require_order
    )
    state = InterdependentState()
    return schedule, state


@pytest.mark.unit
def test_follower_within_window_earns_reinforcement():
    schedule, state = make_schedule(time_window_sec=5.0, require_order=True)
    schedule.record_leader_response(state, timestamp=10.0)
    result = schedule.record_follower_response(state, timestamp=12.0)
    assert result is True


@pytest.mark.unit
def test_follower_outside_window_no_reinforcement():
    schedule, state = make_schedule(time_window_sec=5.0, require_order=True)
    schedule.record_leader_response(state, timestamp=10.0)
    result = schedule.record_follower_response(state, timestamp=16.0)
    assert result is False


@pytest.mark.unit
def test_follower_before_leader_require_order_true():
    schedule, state = make_schedule(time_window_sec=5.0, require_order=True)
    schedule.record_leader_response(state, timestamp=10.0)
    result = schedule.record_follower_response(state, timestamp=8.0)
    assert result is False


@pytest.mark.unit
def test_follower_before_leader_require_order_false_within_window():
    schedule, state = make_schedule(time_window_sec=5.0, require_order=False)
    schedule.record_leader_response(state, timestamp=10.0)
    result = schedule.record_follower_response(state, timestamp=8.0)
    assert result is True


@pytest.mark.unit
def test_total_reinforcers_increments_on_earned():
    schedule, state = make_schedule(time_window_sec=5.0)
    assert state.total_reinforcers == 0
    schedule.record_leader_response(state, timestamp=0.0)
    schedule.record_follower_response(state, timestamp=2.0)
    assert state.total_reinforcers == 1
    schedule.record_leader_response(state, timestamp=10.0)
    schedule.record_follower_response(state, timestamp=11.0)
    assert state.total_reinforcers == 2


@pytest.mark.unit
def test_time_window_zero_or_negative_raises():
    with pytest.raises(ValueError, match="time_window_sec"):
        InterdependentSchedule(time_window_sec=0.0)
    with pytest.raises(ValueError, match="time_window_sec"):
        InterdependentSchedule(time_window_sec=-1.0)


@pytest.mark.unit
def test_reinforcement_rate_two_reinforcers_in_60s():
    schedule, state = make_schedule(time_window_sec=5.0)
    schedule.record_leader_response(state, timestamp=5.0)
    schedule.record_follower_response(state, timestamp=7.0)
    schedule.record_leader_response(state, timestamp=30.0)
    schedule.record_follower_response(state, timestamp=32.0)
    rate = schedule.reinforcement_rate(state, duration_sec=60.0)
    assert rate == pytest.approx(2.0)


@pytest.mark.unit
def test_follower_contingency_ratio_one_of_three():
    schedule, state = make_schedule(time_window_sec=5.0)
    schedule.record_follower_response(state, timestamp=1.0)
    schedule.record_leader_response(state, timestamp=5.0)
    schedule.record_follower_response(state, timestamp=15.0)
    schedule.record_leader_response(state, timestamp=20.0)
    schedule.record_follower_response(state, timestamp=22.0)
    ratio = schedule.follower_contingency_ratio(state)
    assert ratio == pytest.approx(1 / 3, rel=1e-6)


@pytest.mark.unit
def test_follower_contingency_ratio_no_responses_returns_none():
    schedule, state = make_schedule()
    assert schedule.follower_contingency_ratio(state) is None


@pytest.mark.unit
def test_reinforcement_rate_zero_duration_returns_zero():
    schedule, state = make_schedule()
    assert schedule.reinforcement_rate(state, duration_sec=0.0) == 0.0
    assert schedule.reinforcement_rate(state, duration_sec=-5.0) == 0.0


@pytest.mark.unit
def test_no_leader_response_no_reinforcement():
    schedule, state = make_schedule()
    result = schedule.record_follower_response(state, timestamp=5.0)
    assert result is False
