"""Tests for the FI pattern measures compared against published values."""

from __future__ import annotations

import random

import pytest

from ai_operant_box.agent.bernoulli_agent import BernoulliAgent
from ai_operant_box.agent.scallop_agent import ScallopingAgent
from ai_operant_box.analytics.fi_measures import (
    FIInterval,
    acceleration,
    fi_intervals,
    index_of_curvature,
    interval_measures,
    ordinal_irts,
    pause_criterion,
    pooled_quarter_life,
    quarter_life_interpolated,
    quarter_life_rank,
    summarize_fi,
)
from ai_operant_box.chamber.environment import OperantBox
from ai_operant_box.chamber.schedule import FixedInterval


def test_fi_intervals_drops_warmup_and_assigns_responses() -> None:
    presses = [1.0, 2.0, 30.0, 45.0, 59.0, 60.0, 61.0]
    sr = [30.0, 60.0]
    ivs = fi_intervals(presses, sr)
    assert ivs == [FIInterval(start=30.0, end=60.0, responses=(45.0, 59.0, 60.0))]
    with_warmup = fi_intervals(presses, sr, drop_first=False)
    assert with_warmup[0] == FIInterval(start=0.0, end=30.0, responses=(1.0, 2.0, 30.0))


def test_quarter_life_interpolated_treats_count_as_continuous() -> None:
    # 2 responses at 10 and 20 in [0, 40]: N reaches 0.5 halfway to t=10.
    iv = FIInterval(0.0, 40.0, (10.0, 20.0))
    assert quarter_life_interpolated(iv) == pytest.approx(5.0 / 40.0)
    # Rank-based convention returns the first response's time instead.
    assert quarter_life_rank(iv) == pytest.approx(10.0 / 40.0)


def test_quarter_life_of_uniform_responding_is_a_quarter() -> None:
    iv = FIInterval(0.0, 100.0, tuple(float(t) for t in range(1, 101)))
    assert quarter_life_interpolated(iv) == pytest.approx(0.25)


def test_pooled_quarter_life_uses_ten_bins_and_linear_interpolation() -> None:
    # All responses in the last tenth: 25 % point is a quarter into bin 10.
    ivs = [FIInterval(0.0, 30.0, (28.0, 29.0, 29.5, 30.0))] * 3
    assert pooled_quarter_life(ivs) == pytest.approx(0.925)
    uniform = [FIInterval(0.0, 10.0, tuple(float(t) for t in range(1, 11)))]
    assert pooled_quarter_life(uniform) == pytest.approx(0.25)


def test_interval_measures_prp_t4_running_rate() -> None:
    iv = FIInterval(0.0, 30.0, (15.0, 20.0, 25.0, 28.0, 30.0))
    m = interval_measures(iv)
    assert m.prp_fraction == pytest.approx(0.5)
    assert m.t4_fraction == pytest.approx(28.0 / 30.0)
    assert m.running_rate == pytest.approx(5 / 15.0)


def test_index_of_curvature_matches_fry_et_al_bounds() -> None:
    flat = FIInterval(0.0, 40.0, tuple(float(t) for t in range(1, 41)))
    assert index_of_curvature(flat) == pytest.approx(0.0, abs=0.03)
    last = FIInterval(0.0, 40.0, tuple(31.0 + i for i in range(10)))
    assert index_of_curvature(last) == pytest.approx(0.75)


def test_pause_criterion_is_where_pauses_end_five_percent_of_responses() -> None:
    irts = [1.0] * 95 + [10.0] * 5
    assert pause_criterion(irts, share=0.05) == 10.0
    irts = [1.0] * 90 + [3.0] * 5 + [10.0] * 5
    assert pause_criterion(irts, share=0.05) == 10.0
    assert pause_criterion([1.0] * 20, share=0.05) is None


def test_ordinal_irts_and_acceleration() -> None:
    ivs = [FIInterval(0.0, 30.0, (10.0, 16.0, 20.0, 22.0)) for _ in range(3)]
    means = ordinal_irts(ivs, max_order=3)
    assert means == [(1, 6.0, 3), (2, 4.0, 3), (3, 2.0, 3)]
    acc = acceleration(ivs)
    assert (acc.shorter, acc.longer, acc.equal) == (6, 0, 0)
    assert acc.p_one_sided == pytest.approx(0.5**6)


def _run(subject, steps: int = 3000, interval: float = 30.0):
    box = OperantBox(subject=subject, schedule=FixedInterval(interval))
    for _ in range(steps):
        if isinstance(subject, ScallopingAgent):
            subject.update_time(1.0)
        box.step(1.0)
    presses = [r.timestamp for r in box.records if r.response == "Lever Press"]
    sr = [r.timestamp for r in box.records if r.is_reinforced]
    return presses, sr


def test_bernoulli_control_is_flat_on_the_same_board() -> None:
    random.seed(1)
    presses, sr = _run(BernoulliAgent(p=0.5))
    s = summarize_fi(presses, sr, interval=30.0)
    assert s.pooled_quarter_life == pytest.approx(0.25, abs=0.05)
    assert s.median_curvature == pytest.approx(0.0, abs=0.1)


def test_scallop_control_reaches_literature_direction_on_the_same_board() -> None:
    random.seed(2)
    presses, sr = _run(ScallopingAgent(k=1.0 / 900.0))
    s = summarize_fi(presses, sr, interval=30.0)
    # Literature: Q/FI 0.65-0.80, PRP/FI 0.52-0.69 (Gollub 1964; Dukich & Lee 1973).
    assert 0.55 <= s.pooled_quarter_life <= 0.75
    assert 0.3 <= s.median_prp_fraction <= 0.7
    assert s.median_curvature > 0.3
