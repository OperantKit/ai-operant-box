"""Tests for InductionAgent (boost-based induction model).

Reference: Baum, W. M., & Grace, R. C. (2020). Matching theory and induction
explain operant performance. *JEAB*, *113*(2), 390-418.
https://doi.org/10.1002/jeab.583
"""

from __future__ import annotations

import pytest

from ai_operant_box.agent.induction_agent import InductionAgent


@pytest.mark.unit
def test_effective_response_prob_starts_at_baseline():
    agent = InductionAgent(base_response_prob=0.3, induction_strength=0.4, boost_decay=0.5)
    assert agent.effective_response_prob == pytest.approx(0.3)


@pytest.mark.unit
def test_learn_reinforcer_increases_effective_prob():
    agent = InductionAgent(base_response_prob=0.2, induction_strength=0.4, boost_decay=0.5)
    baseline = agent.effective_response_prob
    agent.learn(reinforcer="food")
    assert agent.effective_response_prob > baseline


@pytest.mark.unit
def test_learn_none_does_not_increase_prob():
    agent = InductionAgent(base_response_prob=0.3, induction_strength=0.4, boost_decay=0.5)
    before = agent.effective_response_prob
    agent.learn(reinforcer=None)
    assert agent.effective_response_prob == pytest.approx(before)


@pytest.mark.unit
def test_boost_decays_after_sense():
    agent = InductionAgent(base_response_prob=0.1, induction_strength=0.6, boost_decay=0.5)
    agent.learn(reinforcer="food")
    boosted = agent.effective_response_prob
    assert boosted > 0.1

    agent.sense(discriminative_stimulus="light_on")
    after_decay = agent.effective_response_prob
    assert after_decay < boosted


@pytest.mark.unit
def test_effective_response_prob_never_exceeds_one():
    agent = InductionAgent(base_response_prob=0.9, induction_strength=1.0, boost_decay=0.0)
    for _ in range(5):
        agent.learn(reinforcer="food")
    assert agent.effective_response_prob <= 1.0


@pytest.mark.unit
def test_act_returns_valid_responses():
    agent = InductionAgent(base_response_prob=0.5)
    results = {agent.act() for _ in range(50)}
    assert results.issubset({"respond", "pause"})


@pytest.mark.unit
def test_history_grows_after_act():
    agent = InductionAgent(base_response_prob=0.5)
    assert len(agent.history) == 0
    agent.sense("sd1")
    agent.act()
    assert len(agent.history) == 1
    agent.sense("sd2")
    agent.act()
    assert len(agent.history) == 2


@pytest.mark.unit
def test_invalid_base_response_prob_raises():
    with pytest.raises(ValueError, match="base_response_prob"):
        InductionAgent(base_response_prob=1.5)


@pytest.mark.unit
def test_invalid_induction_strength_raises():
    with pytest.raises(ValueError, match="induction_strength"):
        InductionAgent(induction_strength=-0.1)


@pytest.mark.unit
def test_invalid_boost_decay_raises():
    with pytest.raises(ValueError, match="boost_decay"):
        InductionAgent(boost_decay=1.5)
