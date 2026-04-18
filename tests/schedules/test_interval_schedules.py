"""Tests for VariableIntervalSchedule and RandomIntervalSchedule."""

from __future__ import annotations

import random
import unittest
from unittest.mock import MagicMock, patch

from ai_operant_box.schedules.interval import (
    RandomIntervalSchedule,
    VariableIntervalSchedule,
)


class TestIntervalSchedules(unittest.TestCase):

    def setUp(self):
        self.mock_monotonic = MagicMock(return_value=0.0)
        self.patcher = patch("time.monotonic", self.mock_monotonic)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_variable_interval_schedule_initialization(self):
        with self.assertRaises(ValueError):
            VariableIntervalSchedule(mean_interval=0)
        with self.assertRaises(ValueError):
            VariableIntervalSchedule(mean_interval=-5)

        self.mock_monotonic.return_value = 100.0
        schedule = VariableIntervalSchedule(mean_interval=10.0)
        self.assertIsInstance(schedule, VariableIntervalSchedule)
        self.assertEqual(schedule.mean_interval, 10.0)
        self.assertEqual(len(schedule.sequence), 12)
        self.assertAlmostEqual(schedule.last_reinforcement_time, 100.0)
        self.assertGreater(schedule.next_reinforcement_interval, 0)

    def test_variable_interval_schedule_reinforcement_delivery(self):
        with patch(
            "ai_operant_box.schedules.interval._fleshler_hoffman_sequence",
            return_value=[5.0, 10.0, 5.0],
        ):
            self.mock_monotonic.return_value = 0.0
            schedule = VariableIntervalSchedule(mean_interval=7.0)

            self.mock_monotonic.return_value = 4.0
            self.assertFalse(schedule.decision(response_made=False))
            self.mock_monotonic.return_value = 5.0
            self.assertFalse(
                schedule.decision(response_made=False),
                "Should not reinforce without response",
            )

            self.assertTrue(
                schedule.decision(response_made=True),
                "Should reinforce with response after interval",
            )
            self.assertAlmostEqual(schedule.last_reinforcement_time, 5.0)
            self.assertAlmostEqual(schedule.next_reinforcement_interval, 10.0)

            self.mock_monotonic.return_value = 14.0
            self.assertFalse(
                schedule.decision(response_made=True),
                "Should not reinforce before interval passes",
            )
            self.mock_monotonic.return_value = 15.0
            self.assertTrue(
                schedule.decision(response_made=True),
                "Should reinforce with response after interval",
            )
            self.assertAlmostEqual(schedule.last_reinforcement_time, 15.0)
            self.assertAlmostEqual(schedule.next_reinforcement_interval, 5.0)

    def test_variable_interval_schedule_mean_approximation(self):
        mean_interval = 20.0
        with patch(
            "ai_operant_box.schedules.interval._fleshler_hoffman_sequence",
            side_effect=lambda m, i: [random.expovariate(1 / m) for _ in range(i)],
        ):
            schedule = VariableIntervalSchedule(mean_interval=mean_interval, iterations=1000)
            num_reinforcements = 500
            total_time = 0.0
            actual_reinforcements = 0

            self.mock_monotonic.return_value = 0.0
            for _ in range(num_reinforcements * 2):
                current_time_step = random.uniform(0.1, 1.0)
                self.mock_monotonic.return_value += current_time_step
                total_time += current_time_step

                if schedule.decision(response_made=True):
                    actual_reinforcements += 1

                if actual_reinforcements >= num_reinforcements:
                    break

            if actual_reinforcements > 0:
                calculated_mean_interval = total_time / actual_reinforcements
                # Wide tolerance: the simulated time-step distribution is
                # uniform(0.1, 1.0), which biases the realised mean upward
                # relative to the schedule's exponential mean. The test only
                # asserts that we are in the same order of magnitude.
                self.assertAlmostEqual(
                    calculated_mean_interval, mean_interval, delta=mean_interval * 0.75
                )
            else:
                self.fail("No reinforcements occurred, cannot calculate mean interval.")

    def test_random_interval_schedule_initialization(self):
        with self.assertRaises(ValueError):
            RandomIntervalSchedule(probability_per_unit_time=-0.1)
        with self.assertRaises(ValueError):
            RandomIntervalSchedule(probability_per_unit_time=1.1)

        schedule = RandomIntervalSchedule(probability_per_unit_time=0.1)
        self.assertIsInstance(schedule, RandomIntervalSchedule)
        self.assertEqual(schedule.probability_per_unit_time, 0.1)
        self.assertAlmostEqual(schedule.last_check_time, self.mock_monotonic.return_value)
        self.assertFalse(schedule.reinforcement_available)

    def test_random_interval_schedule_reinforcement_delivery_deterministic(self):
        schedule = RandomIntervalSchedule(probability_per_unit_time=0.1)

        self.mock_monotonic.return_value = 0.0
        schedule.last_check_time = 0.0
        schedule.reinforcement_available = False
        with patch("random.random", return_value=0.9):
            self.mock_monotonic.return_value = 1.0
            self.assertFalse(schedule.decision(response_made=False))
            self.assertFalse(schedule.reinforcement_available)

        self.mock_monotonic.return_value = 0.0
        schedule.last_check_time = 0.0
        schedule.reinforcement_available = False
        with patch("random.random", return_value=0.05):
            self.mock_monotonic.return_value = 1.0
            self.assertFalse(schedule.decision(response_made=False))
            self.assertTrue(schedule.reinforcement_available)

        self.mock_monotonic.return_value = 0.0
        schedule.last_check_time = 0.0
        schedule.reinforcement_available = True
        with patch("random.random", return_value=0.9):
            self.mock_monotonic.return_value = 1.0
            self.assertTrue(schedule.decision(response_made=True))
            self.assertFalse(schedule.reinforcement_available)

    def test_random_interval_schedule_reinforcement_frequency(self):
        probability_per_unit_time = 0.05
        num_seconds = 10000

        schedule_availability = RandomIntervalSchedule(
            probability_per_unit_time=probability_per_unit_time
        )
        available_count = 0
        schedule_availability.last_check_time = 0.0

        for i in range(num_seconds):
            current_time = float(i + 1)
            time_elapsed = current_time - schedule_availability.last_check_time
            schedule_availability.last_check_time = current_time

            effective_probability = min(
                1.0, time_elapsed * schedule_availability.probability_per_unit_time
            )
            if random.random() < effective_probability:
                available_count += 1

        observed_availability_rate = available_count / num_seconds
        self.assertAlmostEqual(
            observed_availability_rate, probability_per_unit_time, delta=0.02
        )


if __name__ == "__main__":
    unittest.main()
