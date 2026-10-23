"""Helpers for running an LLM subject on Conc VI VI (prompt, changeovers, zero counts)."""

from __future__ import annotations

import math

import pytest

from ai_operant_box.analytics.matching import changeovers, fit_generalized_matching
from ai_operant_box.chamber.concurrent import (
    CONCURRENT_ACTIONS,
    CONCURRENT_SYSTEM_PROMPT,
    ChoiceRecord,
)


def _rec(t: float, key: str) -> ChoiceRecord:
    return ChoiceRecord(timestamp=t, operandum=key, is_reinforced=False, cod_suppressed=False)


def test_concurrent_prompt_lists_every_action_and_no_strategy_hint():
    for action in CONCURRENT_ACTIONS:
        assert f"'{action}'" in CONCURRENT_SYSTEM_PROMPT
    lowered = CONCURRENT_SYSTEM_PROMPT.casefold()
    for hint in ("switch", "alternate", "match", "probab", "interval"):
        assert hint not in lowered


def test_changeovers_counts_switches_between_operanda():
    records = [_rec(1, "left"), _rec(2, "left"), _rec(3, "right"), _rec(5, "left"), _rec(6, "left")]
    assert changeovers(records) == 2
    assert changeovers(records, start=3) == 1
    assert changeovers([]) == 0


def test_fit_with_correction_accepts_zero_counts():
    with pytest.raises(ValueError):
        fit_generalized_matching([10, 0], [10, 20], [5, 1], [5, 9])
    fit = fit_generalized_matching([10, 0], [10, 20], [5, 1], [5, 9], correction=0.5)
    assert fit.log_response_ratios[1] == pytest.approx(math.log10(0.5 / 20.5))
    assert fit.log_reinforcer_ratios[1] == pytest.approx(math.log10(1.5 / 9.5))
    assert fit.n == 2
