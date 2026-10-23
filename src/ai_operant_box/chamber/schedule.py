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


class Extinction(ReinforcementSchedule):
    """EXT — no response is reinforced."""

    def is_reinforced(self, response: str, current_time: float) -> bool:
        return False


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


class DifferentialReinforcementOfLowRate(ReinforcementSchedule):
    """DRL — a target response is reinforced if ``interval`` s have passed
    since the last counted response.

    The clock starts at session time 0, so the first response is timed from
    the session start (it is not reinforced unconditionally). Every counted
    response restarts the clock, reinforced or not. Responses within
    ``post_reinforcement_s`` after a reinforcer (the reinforcer delivery
    period) are not counted: they are never reinforced and do not restart the
    clock, which restarts at the end of that period instead.
    """

    def __init__(self, interval: float, target_response: str = "Lever Press",
                 post_reinforcement_s: float = 0.0) -> None:
        if interval <= 0:
            raise ValueError("interval must be positive")
        if post_reinforcement_s < 0:
            raise ValueError("post_reinforcement_s must be non-negative")
        self.interval = interval
        self.target_response = target_response
        self.post_reinforcement_s = post_reinforcement_s
        self.timer_start: float = 0.0
        self.last_reinforcement_time: float | None = None

    def is_reinforced(self, response: str, current_time: float) -> bool:
        if response != self.target_response:
            return False
        if (self.last_reinforcement_time is not None
                and current_time < self.last_reinforcement_time + self.post_reinforcement_s):
            return False
        reinforced = current_time - self.timer_start >= self.interval
        if reinforced:
            self.last_reinforcement_time = current_time
            self.timer_start = current_time + self.post_reinforcement_s
        else:
            self.timer_start = current_time
        return reinforced


__all__ = [
    "ContinuousReinforcement",
    "DifferentialReinforcementOfLowRate",
    "Extinction",
    "FixedInterval",
    "ReinforcementSchedule",
]
