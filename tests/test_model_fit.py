"""ModelFitAnalyzer: each response is counted in the same bin as its opportunity."""

from __future__ import annotations

import numpy as np

from ai_operant_box.analytics.model_fit import ModelFitAnalyzer
from ai_operant_box.chamber.environment import ResponseRecord


def _rec(t: float, response: str = "Lever Press", sr: bool = False) -> ResponseRecord:
    return ResponseRecord(timestamp=t, sd="Light ON", response=response, is_reinforced=sr)


def test_reinforced_response_does_not_push_probability_above_one() -> None:
    # FI 3 s, one step per second, a lever press on every step; t = 3 is reinforced.
    records = [_rec(1.0), _rec(2.0), _rec(3.0, sr=True), _rec(4.0)]
    _, probs = ModelFitAnalyzer(records, interval=3.0).get_empirical_probabilities()
    assert probs.size > 0
    assert np.all(probs <= 1.0)


def test_responding_on_every_step_gives_probability_one_in_every_bin() -> None:
    # Two full FI 3 s intervals plus one step: every visited bin has P = 1.
    records = [_rec(float(t), sr=t in (3, 6)) for t in range(1, 8)]
    _, probs = ModelFitAnalyzer(records, interval=3.0).get_empirical_probabilities()
    assert probs.tolist() == [1.0] * len(probs)


def test_response_type_restricts_counted_responses() -> None:
    # Explore on even steps; only lever presses count when response_type is set.
    records = [
        _rec(1.0),
        _rec(2.0, response="Explore"),
        _rec(3.0, sr=True),
        _rec(4.0, response="Explore"),
        _rec(5.0),
        _rec(6.0, sr=True),
    ]
    centers, all_probs = ModelFitAnalyzer(records, interval=3.0).get_empirical_probabilities()
    _, lever_probs = ModelFitAnalyzer(
        records, interval=3.0, response_type="Lever Press"
    ).get_empirical_probabilities()
    assert all_probs.tolist() == [1.0] * len(centers)
    # Bin [1, 2): steps 1 (press) and 4 (Explore); bin [2, 3]: steps 2, 3, 5, 6.
    assert lever_probs.tolist() == [0.5, 0.75]
