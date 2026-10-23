"""Generalized matching law fit for concurrent-schedule sessions.

``log(B1/B2) = a * log(R1/R2) + log b`` (Baum, 1974), with base-10 logs.
``a`` is sensitivity and ``log b`` is bias toward operandum 1. The fit is
ordinary least squares over conditions.

Reference
---------
Baum, W. M. (1974). On two types of deviation from the matching law:
Bias and undermatching. *Journal of the Experimental Analysis of
Behavior*, 22(1), 231-242. https://doi.org/10.1901/jeab.1974.22-231
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChoiceTotals:
    """Responses and reinforcers per operandum over a session window."""

    responses: dict[str, int] = field(default_factory=dict)
    reinforcers: dict[str, int] = field(default_factory=dict)


def choice_totals(
    records: Iterable[Any],
    start: float = float("-inf"),
    operanda: Sequence[str] = ("left", "right"),
) -> ChoiceTotals:
    """Count responses and reinforcers per operandum for ``timestamp >= start``."""
    totals = ChoiceTotals(responses={k: 0 for k in operanda}, reinforcers={k: 0 for k in operanda})
    for record in records:
        if record.timestamp < start:
            continue
        key = record.operandum
        totals.responses[key] = totals.responses.get(key, 0) + 1
        if record.is_reinforced:
            totals.reinforcers[key] = totals.reinforcers.get(key, 0) + 1
    return totals


def changeovers(records: Iterable[Any], start: float = float("-inf")) -> int:
    """Count switches between operanda among responses with ``timestamp >= start``."""
    count = 0
    previous: str | None = None
    for record in records:
        if record.timestamp < start:
            continue
        if previous is not None and record.operandum != previous:
            count += 1
        previous = record.operandum
    return count


@dataclass(frozen=True)
class GeneralizedMatchingFit:
    a: float
    log_b: float
    r_squared: float
    n: int
    log_response_ratios: tuple[float, ...]
    log_reinforcer_ratios: tuple[float, ...]


def fit_generalized_matching(
    b1: Sequence[float],
    b2: Sequence[float],
    r1: Sequence[float],
    r2: Sequence[float],
    correction: float = 0.0,
) -> GeneralizedMatchingFit:
    """Least-squares fit of the generalized matching law over conditions.

    Each index is one condition. Counts must be positive (a zero makes
    the log ratio undefined), and at least two conditions with distinct
    reinforcer ratios are needed. ``correction`` is added to every count
    before taking logs (e.g. 0.5) so short sessions with an empty cell
    can still be fitted; report it with the estimates.
    """
    if not (len(b1) == len(b2) == len(r1) == len(r2)):
        raise ValueError("b1, b2, r1, r2 must have the same length")
    if correction:
        b1, b2, r1, r2 = ([v + correction for v in seq] for seq in (b1, b2, r1, r2))
    if len(b1) < 2:
        raise ValueError("at least two conditions are needed")
    for name, values in (("b1", b1), ("b2", b2), ("r1", r1), ("r2", r2)):
        if any(v <= 0 for v in values):
            raise ValueError(f"{name} contains a non-positive count: {list(values)!r}")

    x = [math.log10(p / q) for p, q in zip(r1, r2, strict=True)]
    y = [math.log10(p / q) for p, q in zip(b1, b2, strict=True)]
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    if sxx == 0:
        raise ValueError("reinforcer ratios do not vary across conditions")
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y, strict=True))
    a = sxy / sxx
    log_b = my - a * mx
    ss_res = sum((yi - (a * xi + log_b)) ** 2 for xi, yi in zip(x, y, strict=True))
    ss_tot = sum((yi - my) ** 2 for yi in y)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return GeneralizedMatchingFit(
        a=a,
        log_b=log_b,
        r_squared=r_squared,
        n=n,
        log_response_ratios=tuple(y),
        log_reinforcer_ratios=tuple(x),
    )


__all__ = [
    "ChoiceTotals",
    "GeneralizedMatchingFit",
    "changeovers",
    "choice_totals",
    "fit_generalized_matching",
]
