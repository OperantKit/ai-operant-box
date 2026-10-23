"""Helpers shared by the variable / random schedule modules.

Values are drawn independently by inverse-CDF sampling from an exponential
distribution and rounded to integers; their mean approaches the requested
``mean`` for large ``iterations``. This is not the Fleshler & Hoffman (1962)
progression, which is a fixed, deterministic list of intervals.

Reference
---------
Fleshler, M., & Hoffman, H. S. (1962). A progression for generating
variable-interval schedules. *Journal of the Experimental Analysis of
Behavior*, *5*(4), 529-530. https://doi.org/10.1901/jeab.1962.5-529
"""

from __future__ import annotations

import math
import random


def _exponential_sequence(mean: float, iterations: int) -> list[int]:
    """Return ``iterations`` integer ratio/interval values with the given mean.

    Each value is an independent draw ``-mean * ln(1 - u)`` from an
    exponential distribution (inverse-CDF sampling), rounded and clamped to a
    minimum of 1 so it remains a valid ratio.
    """
    if mean <= 0 or iterations <= 0:
        raise ValueError("Mean and iterations must be positive.")

    sequence: list[int] = []
    for _ in range(iterations):
        u = random.random()
        value = -mean * math.log(1 - u)
        sequence.append(max(1, round(value)))
    return sequence


__all__ = ["_exponential_sequence"]
