"""Numerical two-lever subject for concurrent schedules.

:class:`MatchingAgent` presses one lever every step and picks the left
lever with probability ``R_L^s / (R_L^s + R_R^s)``, where ``R_L`` and
``R_R`` are the reinforcers it has obtained on each lever (each starting
at 1) and ``s`` is ``sensitivity``. The lever is redrawn with
probability ``1 / mean_visit`` per step and otherwise kept, so visits
last ``mean_visit`` steps on average; visits longer than the changeover
delay keep the COD from blocking most reinforcers. With ``s = 1`` its response ratio
tracks its obtained-reinforcer ratio, so a generalized-matching fit
should give a sensitivity near 1; ``s < 1`` gives undermatching. It is a
reference subject for checking the chamber and the fit, not a model of
an organism.
"""

from __future__ import annotations

import random
from typing import Any

from ai_operant_box.agent.base import Subject


class MatchingAgent(Subject):
    """Chooses between ``left_action`` and ``right_action`` by obtained reinforcers."""

    def __init__(
        self,
        sensitivity: float = 1.0,
        rng: random.Random | None = None,
        mean_visit: float = 1.0,
        left_action: str = "Left Lever",
        right_action: str = "Right Lever",
    ) -> None:
        if sensitivity < 0:
            raise ValueError(f"sensitivity must be >= 0, got {sensitivity!r}")
        if mean_visit < 1:
            raise ValueError(f"mean_visit must be >= 1, got {mean_visit!r}")
        self.sensitivity = float(sensitivity)
        self.mean_visit = float(mean_visit)
        self.rng = rng or random.Random()
        self.left_action = left_action
        self.right_action = right_action
        self.obtained: dict[str, float] = {left_action: 1.0, right_action: 1.0}
        self._last_action: str | None = None
        self._history: list[dict] = []

    def sense(self, discriminative_stimulus: Any) -> None:
        pass

    def p_left(self) -> float:
        left = self.obtained[self.left_action] ** self.sensitivity
        right = self.obtained[self.right_action] ** self.sensitivity
        return left / (left + right)

    def act(self) -> str:
        p = self.p_left()
        if self._last_action is not None and self.rng.random() >= 1.0 / self.mean_visit:
            action = self._last_action
        else:
            action = self.left_action if self.rng.random() < p else self.right_action
        self._last_action = action
        self._history.append({"action": action, "p_left": p})
        return action

    def learn(self, reinforcer: Any | None) -> None:
        if reinforcer is not None and self._last_action is not None:
            self.obtained[self._last_action] += 1.0
        if self._history:
            self._history[-1]["reinforcer"] = reinforcer

    @property
    def history(self) -> list[dict]:
        return self._history


__all__ = ["MatchingAgent"]
