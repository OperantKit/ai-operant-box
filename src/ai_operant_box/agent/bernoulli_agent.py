"""BernoulliAgent — a control subject that responds with a fixed probability.

Each call to :meth:`act` presses the lever with probability ``p``
regardless of reinforcement, so it is insensitive to the contingency.
On an FI schedule it gives a quarter-life near 0.25 and a flat record,
the baseline against which a subject's temporal pattern is judged.
"""

from __future__ import annotations

import random
from typing import Any

from ai_operant_box.agent.base import Subject


class BernoulliAgent(Subject):
    """Responds with probability ``p`` on every step."""

    def __init__(self, p: float, rng: random.Random | None = None) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = p
        self._rng = rng or random
        self._history: list[dict] = []

    def sense(self, discriminative_stimulus: Any) -> None:
        return None

    def act(self) -> str | None:
        return "Lever Press" if self._rng.random() < self.p else None

    def learn(self, reinforcer: Any | None) -> None:
        self._history.append({"reinforcer": reinforcer})

    @property
    def history(self) -> list[dict]:
        return self._history


__all__ = ["BernoulliAgent"]
