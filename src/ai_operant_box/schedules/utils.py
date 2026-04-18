"""Helpers shared by the variable / random schedule modules.

The Fleshler-Hoffman sequence used here is a Python re-implementation of the
exponential-distribution approximation; it returns rounded integer values
whose mean approaches the requested ``mean`` for large ``iterations``.

Reference
---------
Fleshler, M., & Hoffman, H. S. (1962). A progression for generating
variable-interval schedules. *Journal of the Experimental Analysis of
Behavior*, *5*(4), 529-530. https://doi.org/10.1901/jeab.1962.5-529
"""

from __future__ import annotations

import math
import random


def _fleshler_hoffman_sequence(mean: float, iterations: int) -> list[int]:
    """Return ``iterations`` integer ratio/interval values with the given mean.

    Uses inverse-CDF sampling from an exponential distribution. Values are
    rounded and clamped to a minimum of 1 so they remain valid ratios.
    """
    if mean <= 0 or iterations <= 0:
        raise ValueError("Mean and iterations must be positive.")

    sequence: list[int] = []
    for _ in range(iterations):
        u = random.random()
        value = -mean * math.log(1 - u)
        sequence.append(max(1, round(value)))
    return sequence


__all__ = ["_fleshler_hoffman_sequence"]
