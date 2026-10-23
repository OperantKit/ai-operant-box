"""Variable Ratio (VR) and Random Ratio (RR) schedules."""

from __future__ import annotations

import random

from ai_operant_box.schedules.utils import _exponential_sequence


class VariableRatioSchedule:
    """VR — reinforce after a variable number of responses.

    The per-trial ratios are independent draws from an exponential
    distribution with mean ``mean_ratio``, rounded to integers.
    """

    def __init__(self, mean_ratio: int, iterations: int = 12) -> None:
        if mean_ratio <= 0:
            raise ValueError("Mean ratio must be positive.")
        self.mean_ratio = mean_ratio
        self.iterations = iterations
        self._generate_sequence()
        self.response_count = 0
        self.reinforce_at_response = self.sequence[self.current_sequence_index]

    def _generate_sequence(self) -> None:
        self.sequence = _exponential_sequence(float(self.mean_ratio), self.iterations)
        self.current_sequence_index = 0

    def decision(self, response_made: bool = True) -> bool:
        if response_made:
            self.response_count += 1
        if self.response_count >= self.reinforce_at_response:
            self.response_count = 0
            self.current_sequence_index = (self.current_sequence_index + 1) % len(
                self.sequence
            )
            self.reinforce_at_response = self.sequence[self.current_sequence_index]
            return True
        return False

    def get_current_target_response(self) -> int:
        return self.reinforce_at_response

    def get_responses_since_last_reinforcement(self) -> int:
        return self.response_count


class RandomRatioSchedule:
    """RR — each response is reinforced with constant probability."""

    def __init__(self, probability: float) -> None:
        if not (0.0 <= probability <= 1.0):
            raise ValueError("Probability must be between 0.0 and 1.0.")
        self.probability = probability

    def decision(self, response_made: bool = True) -> bool:
        if response_made and random.random() < self.probability:
            return True
        return False


__all__ = ["RandomRatioSchedule", "VariableRatioSchedule"]
