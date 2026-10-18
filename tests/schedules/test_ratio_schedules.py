"""Tests for VariableRatioSchedule and RandomRatioSchedule."""

from __future__ import annotations

import random
import unittest
from unittest.mock import patch

from ai_operant_box.schedules.ratio import RandomRatioSchedule, VariableRatioSchedule


class TestRatioSchedules(unittest.TestCase):

    def test_variable_ratio_schedule_initialization(self):
        with self.assertRaises(ValueError):
            VariableRatioSchedule(mean_ratio=0)
        with self.assertRaises(ValueError):
            VariableRatioSchedule(mean_ratio=-5)

        schedule = VariableRatioSchedule(mean_ratio=10)
        self.assertIsInstance(schedule, VariableRatioSchedule)
        self.assertEqual(schedule.mean_ratio, 10)
        self.assertEqual(len(schedule.sequence), 12)
        self.assertEqual(schedule.response_count, 0)
        self.assertGreater(schedule.reinforce_at_response, 0)

    def test_variable_ratio_schedule_reinforcement_delivery(self):
        with patch(
            "ai_operant_box.schedules.ratio._fleshler_hoffman_sequence",
            return_value=[5, 10, 5],
        ):
            schedule = VariableRatioSchedule(mean_ratio=7)

            for i in range(4):
                self.assertFalse(
                    schedule.decision(response_made=True),
                    f"Should not reinforce at {i + 1} responses",
                )
            self.assertTrue(
                schedule.decision(response_made=True),
                "Should reinforce at 5 responses",
            )
            self.assertEqual(schedule.response_count, 0)
            self.assertEqual(schedule.reinforce_at_response, 10)

            for i in range(9):
                self.assertFalse(
                    schedule.decision(response_made=True),
                    f"Should not reinforce at {i + 1} responses after first",
                )
            self.assertTrue(
                schedule.decision(response_made=True),
                "Should reinforce at 10 responses",
            )
            self.assertEqual(schedule.response_count, 0)
            self.assertEqual(schedule.reinforce_at_response, 5)

            for i in range(4):
                self.assertFalse(
                    schedule.decision(response_made=True),
                    f"Should not reinforce at {i + 1} responses after second",
                )
            self.assertTrue(
                schedule.decision(response_made=True),
                "Should reinforce at 5 responses",
            )
            self.assertEqual(schedule.response_count, 0)
            self.assertEqual(schedule.reinforce_at_response, 5)

    def test_variable_ratio_schedule_mean_approximation(self):
        # The ratio sequence is 100 exponential draws, so its mean has an SD of
        # about 2 and an unseeded run falls outside the 20% tolerance for
        # roughly 1 seed in 15. Fix the seed and restore the global state.
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(42)
        mean_ratio = 20
        schedule = VariableRatioSchedule(mean_ratio=mean_ratio, iterations=100)
        num_reinforcements = 1000
        total_responses = 0
        actual_reinforcements = 0

        for _ in range(num_reinforcements * mean_ratio):
            if schedule.decision(response_made=True):
                actual_reinforcements += 1
            total_responses += 1

            if actual_reinforcements >= num_reinforcements:
                break

        if actual_reinforcements > 0:
            calculated_mean = total_responses / actual_reinforcements
            self.assertAlmostEqual(calculated_mean, mean_ratio, delta=mean_ratio * 0.2)
        else:
            self.fail("No reinforcements occurred, cannot calculate mean.")

    def test_random_ratio_schedule_initialization(self):
        with self.assertRaises(ValueError):
            RandomRatioSchedule(probability=-0.1)
        with self.assertRaises(ValueError):
            RandomRatioSchedule(probability=1.1)

        schedule = RandomRatioSchedule(probability=0.5)
        self.assertIsInstance(schedule, RandomRatioSchedule)
        self.assertEqual(schedule.probability, 0.5)

    def test_random_ratio_schedule_reinforcement_delivery_deterministic(self):
        schedule = RandomRatioSchedule(probability=0.5)

        with patch("random.random", return_value=0.4):
            self.assertTrue(schedule.decision(response_made=True))

        with patch("random.random", return_value=0.6):
            self.assertFalse(schedule.decision(response_made=True))

        with patch("random.random", return_value=0.0):
            self.assertTrue(schedule.decision(response_made=True))

        with patch("random.random", return_value=0.99):
            schedule_low_prob = RandomRatioSchedule(probability=0.01)
            self.assertFalse(schedule_low_prob.decision(response_made=True))

    def test_random_ratio_schedule_reinforcement_frequency(self):
        probability = 0.2
        schedule = RandomRatioSchedule(probability=probability)
        num_trials = 10000
        reinforcements = 0

        for _ in range(num_trials):
            if schedule.decision(response_made=True):
                reinforcements += 1

        observed_probability = reinforcements / num_trials
        self.assertAlmostEqual(observed_probability, probability, delta=0.05)


if __name__ == "__main__":
    unittest.main()
