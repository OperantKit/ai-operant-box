"""Variable Interval (VI) and Random Interval (RI) schedules.

Both schedules read the same virtual clock as :class:`FixedInterval`, never
the wall clock, so the time an LLM subject spends producing a response does
not count toward the interval. Pass the chamber's clock as ``clock``, or pass
the current time explicitly through ``decision(..., current_time=...)`` /
``is_reinforced(response, current_time)``.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from ai_operant_box.chamber.environment import VirtualClock
from ai_operant_box.schedules.utils import _exponential_sequence


class VariableIntervalSchedule:
    """VI — reinforce the first response after a variable interval.

    Intervals are independent draws from an exponential distribution with
    mean ``mean_interval`` (seconds), rounded to whole seconds.
    """

    def __init__(
        self,
        mean_interval: float,
        iterations: int = 12,
        clock: Callable[[], float] | None = None,
        target_response: str = "Lever Press",
    ) -> None:
        if mean_interval <= 0:
            raise ValueError("Mean interval must be positive.")
        self.mean_interval = mean_interval
        self.iterations = iterations
        self.clock = clock if clock is not None else VirtualClock()
        self.target_response = target_response
        self._generate_sequence()
        self.last_reinforcement_time = self.clock()
        self.next_reinforcement_interval = self.sequence[self.current_sequence_index]

    def _generate_sequence(self) -> None:
        self.sequence = _exponential_sequence(self.mean_interval, self.iterations)
        self.current_sequence_index = 0

    def _time(self, current_time: float | None) -> float:
        return self.clock() if current_time is None else current_time

    def decision(
        self, response_made: bool = False, current_time: float | None = None
    ) -> bool:
        now = self._time(current_time)
        time_since_last_reinforcement = now - self.last_reinforcement_time
        if (
            time_since_last_reinforcement >= self.next_reinforcement_interval
            and response_made
        ):
            self.last_reinforcement_time = now
            self.current_sequence_index = (self.current_sequence_index + 1) % len(
                self.sequence
            )
            self.next_reinforcement_interval = self.sequence[
                self.current_sequence_index
            ]
            return True
        return False

    def is_reinforced(self, response: str, current_time: float) -> bool:
        """:class:`OperantBox` entry point, same signature as ``FixedInterval``."""
        return self.decision(response == self.target_response, current_time)

    def get_time_until_next_reinforcement(
        self, current_time: float | None = None
    ) -> float:
        time_since_last_reinforcement = (
            self._time(current_time) - self.last_reinforcement_time
        )
        return max(0.0, self.next_reinforcement_interval - time_since_last_reinforcement)

    def get_current_target_interval(self) -> float:
        return self.next_reinforcement_interval


class RandomIntervalSchedule:
    """RI — reinforcer becomes available with constant per-second probability."""

    def __init__(
        self,
        probability_per_unit_time: float,
        clock: Callable[[], float] | None = None,
        target_response: str = "Lever Press",
    ) -> None:
        if not (0.0 <= probability_per_unit_time <= 1.0):
            raise ValueError("Probability per unit time must be between 0.0 and 1.0.")
        self.probability_per_unit_time = probability_per_unit_time
        self.clock = clock if clock is not None else VirtualClock()
        self.target_response = target_response
        self.last_check_time = self.clock()
        self.reinforcement_available = False

    def decision(
        self, response_made: bool = False, current_time: float | None = None
    ) -> bool:
        now = self.clock() if current_time is None else current_time
        time_elapsed = now - self.last_check_time
        self.last_check_time = now

        if time_elapsed > 0:
            effective_probability = min(
                1.0, time_elapsed * self.probability_per_unit_time
            )
            if random.random() < effective_probability:
                self.reinforcement_available = True

        if self.reinforcement_available and response_made:
            self.reinforcement_available = False
            return True
        return False

    def is_reinforced(self, response: str, current_time: float) -> bool:
        """:class:`OperantBox` entry point, same signature as ``FixedInterval``."""
        return self.decision(response == self.target_response, current_time)


__all__ = ["RandomIntervalSchedule", "VariableIntervalSchedule"]
