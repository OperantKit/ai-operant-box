"""DRL (differential reinforcement of low rate) on the single-lever chamber."""

from __future__ import annotations

from ai_operant_box.chamber import DifferentialReinforcementOfLowRate, OperantBox

PRESS = "Lever Press"


def _decide(schedule: DifferentialReinforcementOfLowRate, times: list[float]) -> list[bool]:
    return [schedule.is_reinforced(PRESS, t) for t in times]


def test_first_press_is_timed_from_session_start() -> None:
    assert _decide(DifferentialReinforcementOfLowRate(10), [10.0]) == [True]
    assert _decide(DifferentialReinforcementOfLowRate(10), [9.0]) == [False]


def test_irt_shorter_than_interval_is_not_reinforced() -> None:
    assert _decide(DifferentialReinforcementOfLowRate(10), [10.0, 19.0]) == [True, False]
    assert _decide(DifferentialReinforcementOfLowRate(10), [10.0, 21.0]) == [True, True]


def test_unreinforced_press_resets_the_timer() -> None:
    # 19 -> 28 is 9 s: the unreinforced press at 19 restarted the clock.
    assert _decide(DifferentialReinforcementOfLowRate(10), [10.0, 19.0, 28.0]) == [True, False, False]


def test_other_responses_are_never_reinforced_and_do_not_reset() -> None:
    schedule = DifferentialReinforcementOfLowRate(10)
    assert schedule.is_reinforced("Nothing", 5.0) is False
    assert schedule.is_reinforced(PRESS, 10.0) is True


class _AlwaysPress:
    current_sd = None

    def sense(self, sd: str) -> None:
        self.current_sd = sd

    def act(self) -> str:
        return PRESS

    def learn(self, sr: object) -> None:
        pass


def test_pressing_every_step_earns_nothing_in_1200_steps() -> None:
    box = OperantBox(subject=_AlwaysPress(), schedule=DifferentialReinforcementOfLowRate(10))
    for _ in range(1200):
        box.step(1.0)
    assert len(box.records) == 1200
    assert sum(r.is_reinforced for r in box.records) == 0


def test_presses_during_post_reinforcement_window_are_not_counted() -> None:
    schedule = DifferentialReinforcementOfLowRate(10, post_reinforcement_s=4.0)
    # Reinforced at 10; the press at 12 is inside 10..14, so it neither counts
    # nor resets the clock. The next decision is timed from 10 + 4 = 14.
    assert _decide(schedule, [10.0, 12.0, 23.0]) == [True, False, False]
    schedule = DifferentialReinforcementOfLowRate(10, post_reinforcement_s=4.0)
    assert _decide(schedule, [10.0, 12.0, 24.0]) == [True, False, True]


def test_default_post_reinforcement_window_is_zero() -> None:
    schedule = DifferentialReinforcementOfLowRate(10)
    assert schedule.post_reinforcement_s == 0.0
    assert _decide(schedule, [10.0, 20.0]) == [True, True]
