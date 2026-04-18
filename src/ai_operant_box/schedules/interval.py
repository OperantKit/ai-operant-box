"""Variable Interval (VI) and Random Interval (RI) schedules."""

from __future__ import annotations

import random
import time

from ai_operant_box.schedules.utils import _fleshler_hoffman_sequence


class VariableIntervalSchedule:
    """VI — reinforce the first response after a variable interval.

    Intervals are drawn from a Fleshler-Hoffman exponential approximation
    with mean ``mean_interval`` (seconds).
    """

    def __init__(self, mean_interval: float, iterations: int = 12) -> None:
        if mean_interval <= 0:
            raise ValueError("Mean interval must be positive.")
        self.mean_interval = mean_interval
        self.iterations = iterations
        self._generate_sequence()
        self.last_reinforcement_time = time.monotonic()
        self.next_reinforcement_interval = self.sequence[self.current_sequence_index]

    def _generate_sequence(self) -> None:
        self.sequence = _fleshler_hoffman_sequence(self.mean_interval, self.iterations)
        self.current_sequence_index = 0

    def decision(self, response_made: bool = False) -> bool:
        current_time = time.monotonic()
        time_since_last_reinforcement = current_time - self.last_reinforcement_time
        if (
            time_since_last_reinforcement >= self.next_reinforcement_interval
            and response_made
        ):
            self.last_reinforcement_time = current_time
            self.current_sequence_index = (self.current_sequence_index + 1) % len(
                self.sequence
            )
            self.next_reinforcement_interval = self.sequence[
                self.current_sequence_index
            ]
            return True
        return False

    def get_time_until_next_reinforcement(self) -> float:
        current_time = time.monotonic()
        time_since_last_reinforcement = current_time - self.last_reinforcement_time
        return max(0.0, self.next_reinforcement_interval - time_since_last_reinforcement)

    def get_current_target_interval(self) -> float:
        return self.next_reinforcement_interval


class RandomIntervalSchedule:
    """RI — reinforcer becomes available with constant per-second probability."""

    def __init__(self, probability_per_unit_time: float) -> None:
        if not (0.0 <= probability_per_unit_time <= 1.0):
            raise ValueError("Probability per unit time must be between 0.0 and 1.0.")
        self.probability_per_unit_time = probability_per_unit_time
        self.last_check_time = time.monotonic()
        self.reinforcement_available = False

    def decision(self, response_made: bool = False) -> bool:
        current_time = time.monotonic()
        time_elapsed = current_time - self.last_check_time
        self.last_check_time = current_time

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


__all__ = ["RandomIntervalSchedule", "VariableIntervalSchedule"]
