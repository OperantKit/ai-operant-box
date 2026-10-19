"""Concurrent VI VI chamber, two-lever subject interface, and matching fit."""

from __future__ import annotations

import math
import random

import pytest

pytest.importorskip("contingency")

from ai_operant_box.agent.base import Subject  # noqa: E402
from ai_operant_box.agent.matching_agent import MatchingAgent  # noqa: E402
from ai_operant_box.analytics.matching import (  # noqa: E402
    choice_totals,
    fit_generalized_matching,
)
from ai_operant_box.chamber.concurrent import (  # noqa: E402
    CONCURRENT_ACTIONS,
    ChoiceRecord,
    ConcurrentBox,
    concurrent_vi_vi,
)


class ScriptedSubject(Subject):
    """Emits a fixed sequence of actions and records what it received."""

    def __init__(self, actions: list[str | None]) -> None:
        self._actions = list(actions)
        self.sensed: list[object] = []
        self.received: list[object] = []

    def sense(self, discriminative_stimulus):
        self.sensed.append(discriminative_stimulus)

    def act(self):
        return self._actions.pop(0)

    def learn(self, reinforcer):
        self.received.append(reinforcer)

    @property
    def history(self):
        return []


def test_actions_vocabulary_names_both_levers():
    assert CONCURRENT_ACTIONS == ("Left Lever", "Right Lever", "Nothing")


def test_box_routes_responses_to_operanda_and_ticks_on_nothing():
    # VI with a tiny mean sets up almost at once; no COD.
    schedule = concurrent_vi_vi(0.5, 0.5, cod=0.0, seed=1)
    subject = ScriptedSubject(["Left Lever", "Nothing", "Right Lever", "Grooming"])
    box = ConcurrentBox(subject, schedule)
    records = [box.step() for _ in range(4)]

    assert isinstance(records[0], ChoiceRecord)
    assert records[0].operandum == "left"
    assert records[1] is None
    assert records[2].operandum == "right"
    assert records[3] is None  # a non-lever action is not a response
    assert [r.timestamp for r in box.records] == [1.0, 3.0]
    # The subject is told which lever produced the food.
    reinforced = [r for r in box.records if r.is_reinforced]
    assert reinforced
    assert subject.received[0] in (None, "Food Pellet")
    assert len(subject.received) == 4
    assert subject.sensed[0] == box.current_sd


def test_cod_suppresses_reinforcement_right_after_a_changeover():
    schedule = concurrent_vi_vi(0.5, 0.5, cod=3.0, seed=2)
    subject = ScriptedSubject(["Left Lever"] * 5 + ["Right Lever"] * 5)
    box = ConcurrentBox(subject, schedule)
    box.run(10)
    right = [r for r in box.records if r.operandum == "right"]
    # Changeover at t=6: responses at t=6, 7, 8 fall inside the 3 s COD.
    assert [r.is_reinforced for r in right[:3]] == [False, False, False]
    assert all(r.cod_suppressed for r in right[:1])
    assert any(r.is_reinforced for r in right[3:])


def test_unknown_mapping_target_is_rejected():
    schedule = concurrent_vi_vi(10.0, 10.0, seed=0)
    with pytest.raises(ValueError):
        ConcurrentBox(ScriptedSubject([]), schedule, operanda={"Left Lever": "middle"})


def test_choice_totals_counts_responses_and_reinforcers_per_lever():
    records = [
        ChoiceRecord(1.0, "left", True, False),
        ChoiceRecord(2.0, "left", False, False),
        ChoiceRecord(3.0, "right", False, True),
        ChoiceRecord(4.0, "right", True, False),
        ChoiceRecord(5.0, "left", False, False),
    ]
    totals = choice_totals(records)
    assert totals.responses == {"left": 3, "right": 2}
    assert totals.reinforcers == {"left": 1, "right": 1}
    assert choice_totals(records, start=2.5).responses == {"left": 1, "right": 2}


def test_fit_recovers_known_sensitivity_and_bias():
    log_r = [-1.0, -0.5, 0.0, 0.5, 1.0]
    log_b = [0.8 * x + 0.1 for x in log_r]
    fit = fit_generalized_matching(
        [10**x for x in log_b], [1.0] * 5, [10**x for x in log_r], [1.0] * 5
    )
    assert fit.a == pytest.approx(0.8)
    assert fit.log_b == pytest.approx(0.1)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.n == 5


def test_fit_rejects_zero_counts_and_too_few_points():
    with pytest.raises(ValueError):
        fit_generalized_matching([1, 0], [1, 1], [1, 1], [1, 2])
    with pytest.raises(ValueError):
        fit_generalized_matching([2], [1], [3], [1])


def test_matching_agent_chooses_by_obtained_reinforcers():
    agent = MatchingAgent(sensitivity=1.0, rng=random.Random(0))
    agent.sense(None)
    assert agent.act() in ("Left Lever", "Right Lever")
    with pytest.raises(ValueError):
        MatchingAgent(sensitivity=-1.0)


def _cod_holds_reinforcers() -> bool:
    """True if contingency-py keeps a reinforcer set up during the COD.

    Older contingency-py releases discard it. Then the lean lever loses
    the reinforcers it sets up while the COD runs, and a subject that
    chooses by obtained reinforcers drifts to exclusive preference.
    """
    from contingency.schedules.concurrent import Concurrent

    return hasattr(Concurrent, "_deliver_pending")


def _run_condition(r_left, r_right, sensitivity, seed, cod):
    # Overall programmed rate 1 reinforcer per 10 s; 1 step = 1 s.
    mean_left = 10.0 * (r_left + r_right) / r_left
    mean_right = 10.0 * (r_left + r_right) / r_right
    schedule = concurrent_vi_vi(mean_left, mean_right, cod=cod, seed=seed)
    agent = MatchingAgent(sensitivity=sensitivity, rng=random.Random(seed), mean_visit=3.0)
    box = ConcurrentBox(agent, schedule)
    box.run(4000)
    return choice_totals(box.records, start=1000.0)


RATIOS = [(1, 1), (1, 3), (3, 1), (1, 9)]


def _fit(sensitivity, cod):
    totals = [
        _run_condition(rl, rr, sensitivity, seed=10 + i, cod=cod)
        for i, (rl, rr) in enumerate(RATIOS)
    ]
    for t in totals:
        assert t.reinforcers["left"] + t.reinforcers["right"] >= 20
    return fit_generalized_matching(
        [t.responses["left"] for t in totals],
        [t.responses["right"] for t in totals],
        [t.reinforcers["left"] for t in totals],
        [t.reinforcers["right"] for t in totals],
    )


def test_numeric_matching_subject_yields_sensitivity_near_one_without_cod():
    fit = _fit(1.0, cod=0.0)
    assert 0.9 <= fit.a <= 1.1
    assert abs(fit.log_b) < 0.1
    assert fit.r_squared > 0.95


@pytest.mark.skipif(
    not _cod_holds_reinforcers(),
    reason="installed contingency-py discards reinforcers set up during the COD",
)
def test_numeric_matching_subject_yields_sensitivity_near_one_with_cod():
    fit = _fit(1.0, cod=2.0)
    assert 0.9 <= fit.a <= 1.1
    assert abs(fit.log_b) < 0.1
    assert fit.r_squared > 0.95


def test_undermatching_subject_yields_lower_sensitivity():
    fit = _fit(0.5, cod=0.0)
    assert 0.3 < fit.a < 0.7
    assert not math.isnan(fit.a)


def test_matching_agent_keeps_its_lever_within_a_visit():
    agent = MatchingAgent(rng=random.Random(3), mean_visit=1e9)
    first = agent.act()
    assert all(agent.act() == first for _ in range(50))
    with pytest.raises(ValueError):
        MatchingAgent(mean_visit=0.5)
