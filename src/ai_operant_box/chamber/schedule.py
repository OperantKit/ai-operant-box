"""Single-subject reinforcement schedules used by :class:`OperantBox`."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ReinforcementSchedule(ABC):
    """Abstract base for FR / VR / FI / VI / CRF / EXT schedules."""

    @abstractmethod
    def is_reinforced(self, response: str, current_time: float) -> bool:
        """Return ``True`` if the response meets the reinforcement criterion."""


class ContinuousReinforcement(ReinforcementSchedule):
    """CRF — every target response is reinforced."""

    def __init__(self, target_response: str = "Lever Press") -> None:
        self.target_response = target_response

    def is_reinforced(self, response: str, current_time: float) -> bool:
        return response == self.target_response


class FixedInterval(ReinforcementSchedule):
    """FI — first target response after ``interval`` seconds is reinforced."""

    def __init__(self, interval: float, target_response: str = "Lever Press") -> None:
        self.interval = interval
        self.target_response = target_response
        self.last_reinforcement_time: float = 0.0

    def is_reinforced(self, response: str, current_time: float) -> bool:
        if response != self.target_response:
            return False
        if current_time - self.last_reinforcement_time >= self.interval:
            self.last_reinforcement_time = current_time
            return True
        return False


__all__ = ["ContinuousReinforcement", "FixedInterval", "ReinforcementSchedule"]
