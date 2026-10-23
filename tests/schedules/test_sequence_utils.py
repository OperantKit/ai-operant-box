"""Tests for the value generator shared by the VI / VR schedules."""

from __future__ import annotations

import random

import pytest

from ai_operant_box.schedules import utils


@pytest.mark.unit
def test_generator_is_named_for_what_it_does() -> None:
    # The values are drawn by inverse-CDF sampling from an exponential
    # distribution, not by the Fleshler & Hoffman (1962) progression.
    assert hasattr(utils, "_exponential_sequence")
    assert not hasattr(utils, "_fleshler_hoffman_sequence")
    assert "Fleshler-Hoffman" not in (utils._exponential_sequence.__doc__ or "")


@pytest.mark.unit
def test_exponential_sequence_mean_and_minimum() -> None:
    random.seed(0)
    values = utils._exponential_sequence(20.0, 5000)
    assert len(values) == 5000
    assert min(values) >= 1
    assert sum(values) / len(values) == pytest.approx(20.0, rel=0.05)


@pytest.mark.unit
def test_exponential_sequence_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        utils._exponential_sequence(0.0, 10)
    with pytest.raises(ValueError):
        utils._exponential_sequence(10.0, 0)
