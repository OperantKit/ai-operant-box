"""ScallopingAgent — quadratic-time response probability agent.

Models the classic FI-scallop pattern: P(R) rises with the square of the
elapsed time since the last reinforcer.
"""

from __future__ import annotations

import random
from typing import Any

from ai_operant_box.agent.base import Subject


class ScallopingAgent(Subject):
    """Virtual subject that simulates the FI scallop.

    Response probability per call to :meth:`act` is::

        P(R) = min(1, k * t**2)

    where ``t`` is the time since the most recent reinforcer (advanced
    externally via :meth:`update_time`).

    ``rng`` is the random number generator the agent draws from. When it is
    ``None`` the agent uses the module-level :mod:`random` functions, so
    ``random.seed(...)`` controls it; pass ``random.Random(seed)`` to get a
    reproducible stream without touching the module-level state.
    """

    def __init__(self, k: float = 0.0001, rng: random.Random | None = None) -> None:
        self.k = k
        self._rng = rng
        self.time_since_last_sr: float = 0.0
        self.current_time: float = 0.0
        self._history: list[dict] = []

    def sense(self, discriminative_stimulus: Any) -> None:
        # ScallopingAgent is SD-insensitive; recorded only via history if needed.
        return None

    def act(self) -> str | None:
        prob = min(1.0, self.k * (self.time_since_last_sr ** 2))
        draw = self._rng.random() if self._rng is not None else random.random()
        if draw < prob:
            return "Lever Press"
        return None

    def learn(self, reinforcer: Any | None) -> None:
        if reinforcer:
            self.time_since_last_sr = 0.0
        self._history.append({"reinforcer": reinforcer})

    def update_time(self, delta: float) -> None:
        """Advance the internal scallop clock by ``delta`` seconds."""
        self.time_since_last_sr += delta
        self.current_time += delta

    @property
    def history(self) -> list[dict]:
        return self._history


__all__ = ["ScallopingAgent"]
