"""Induction-modulated virtual subjects.

``InductionSubject`` is a NCR-driven subject following Baum & Grace's (2020)
sub-linear power law for non-contingent reinforcer induction.

``InductionAgent`` is a boost-based subject whose effective response probability
is elevated by every reinforcer delivery and decays geometrically per
time-step.

Reference
---------
Baum, W. M., & Grace, R. C. (2020). Matching theory and induction explain
operant performance. *Journal of the Experimental Analysis of Behavior*,
*113*(2), 390-418. https://doi.org/10.1002/jeab.583
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from ai_operant_box.agent.base import Subject

# ---------------------------------------------------------------------------
# InductionSubject — NCR power-law induction (Baum & Grace, 2020)
# ---------------------------------------------------------------------------


class InductionSubject(Subject):
    """Virtual subject whose baseline response rate is elevated by induction.

    The response probability per :meth:`act` call is::

        P(R | t) = min(1, k * t**2 + alpha * ncr_count**beta)

    where:

    - ``k``     — scallop coefficient (FI-like temporal gradient; 0 disables).
    - ``alpha`` — induction strength scaling the NCR-driven component.
    - ``beta``  — induction exponent (default 0.5; sub-linear per Baum & Grace).
    - ``ncr_count`` — cumulative non-contingent reinforcer deliveries.
    """

    def __init__(
        self,
        induction_strength: float = 0.1,
        induction_exponent: float = 0.5,
        k: float = 0.0,
    ) -> None:
        self.induction_strength = induction_strength
        self.induction_exponent = induction_exponent
        self.k = k

        self._ncr_count: int = 0
        self._time_since_last_sr: float = 0.0
        self._history: list[dict] = []
        self._current_sd: Any = None

    def sense(self, discriminative_stimulus: Any) -> None:
        self._current_sd = discriminative_stimulus

    def act(self) -> str | None:
        scallop_component = self.k * (self._time_since_last_sr ** 2)
        induction_component = (
            self.induction_strength * (self._ncr_count ** self.induction_exponent)
            if self._ncr_count > 0
            else 0.0
        )
        prob = min(1.0, scallop_component + induction_component)
        if random.random() < prob:
            return "Lever Press"
        return None

    def learn(self, reinforcer: Any | None) -> None:
        if reinforcer == "NCR":
            self._ncr_count += 1
        elif reinforcer is not None:
            self._time_since_last_sr = 0.0
        self._history.append({
            "reinforcer": reinforcer,
            "ncr_count": self._ncr_count,
        })

    @property
    def history(self) -> list[dict]:
        return self._history

    def update_time(self, delta: float) -> None:
        """Advance the scallop timer (call once per time-step)."""
        self._time_since_last_sr += delta

    def deliver_ncr(self) -> None:
        """Deliver one non-contingent reinforcer (shorthand for experiments)."""
        self.learn("NCR")

    def reset(self) -> None:
        """Reset all state for a new session."""
        self._ncr_count = 0
        self._time_since_last_sr = 0.0
        self._history = []


# ---------------------------------------------------------------------------
# InductionAgent — boost-based induction (Baum & Grace, 2020)
# ---------------------------------------------------------------------------


@dataclass
class InductionState:
    """Mutable induction-boost state."""
    boost: float = 0.0
    decay_rate: float = 0.5


class InductionAgent(Subject):
    """Virtual subject with phylogenetic induction modulation.

    Each reinforcer delivery boosts the response probability by
    ``induction_strength``; the boost decays geometrically each
    :meth:`sense` call.

    Parameters
    ----------
    base_response_prob : float
        Baseline response probability per timestep (in [0, 1]).
    induction_strength : float
        Per-reinforcer boost (in [0, 1]).
    boost_decay : float
        Geometric decay rate of the boost per timestep (in [0, 1]).
        ``0`` = boost persists, ``1`` = removed in one step.
    """

    def __init__(
        self,
        base_response_prob: float = 0.3,
        induction_strength: float = 0.4,
        boost_decay: float = 0.5,
    ) -> None:
        if not (0.0 <= base_response_prob <= 1.0):
            raise ValueError(
                f"base_response_prob must be in [0, 1], got {base_response_prob}"
            )
        if not (0.0 <= induction_strength <= 1.0):
            raise ValueError(
                f"induction_strength must be in [0, 1], got {induction_strength}"
            )
        if not (0.0 <= boost_decay <= 1.0):
            raise ValueError(f"boost_decay must be in [0, 1], got {boost_decay}")

        self.base_response_prob = base_response_prob
        self.induction_strength = induction_strength
        self.boost_decay = boost_decay

        self._induction = InductionState(decay_rate=boost_decay)
        self._last_sd: Any = None
        self._history: list[dict] = []

    @property
    def effective_response_prob(self) -> float:
        return min(self.base_response_prob + self._induction.boost, 1.0)

    def sense(self, discriminative_stimulus: Any) -> None:
        self._last_sd = discriminative_stimulus
        self._induction.boost *= 1.0 - self._induction.decay_rate

    def act(self) -> str:
        response = "respond" if random.random() < self.effective_response_prob else "pause"
        self._history.append({
            "sd": self._last_sd,
            "response": response,
            "effective_prob": self.effective_response_prob,
            "induction_boost": self._induction.boost,
        })
        return response

    def learn(self, reinforcer: Any | None) -> None:
        if reinforcer is not None:
            self._induction.boost = min(
                self._induction.boost + self.induction_strength, 1.0
            )
        if self._history:
            self._history[-1]["reinforcer"] = reinforcer

    @property
    def history(self) -> list[dict]:
        return list(self._history)


__all__ = ["InductionAgent", "InductionState", "InductionSubject"]
