"""OperantBox driving a contingency-py schedule (``step(now, event)``) and punishment."""

from __future__ import annotations

from typing import Any

import pytest

from ai_operant_box.chamber import FixedInterval, OperantBox

pytest.importorskip("contingency")

from contingency.schedules import FR, VI, Overlay, Sidman  # noqa: E402


class ScriptedSubject:
    """Emits the given actions in turn and records learn / punish calls."""

    def __init__(self, actions: list[str]) -> None:
        self.actions = list(actions)
        self.learned: list[Any] = []
        self.punished: list[float] = []

    def sense(self, sd: Any) -> None:
        pass

    def act(self) -> str:
        return self.actions.pop(0)

    def learn(self, sr: Any | None) -> None:
        self.learned.append(sr)

    def punish(self, amount: float) -> None:
        self.punished.append(amount)


class NoPunishSubject(ScriptedSubject):
    punish = None  # type: ignore[assignment]


def test_fr_reinforces_every_nth_lever_press() -> None:
    subject = ScriptedSubject(["Lever Press"] * 4)
    box = OperantBox(subject=subject, schedule=FR(2))
    for _ in range(4):
        box.step(1.0)
    assert [r.is_reinforced for r in box.records] == [False, True, False, True]
    assert [r.is_punished for r in box.records] == [False] * 4
    assert subject.punished == []


def test_only_target_response_is_sent_to_the_schedule() -> None:
    subject = ScriptedSubject(["Grooming", "Lever Press"])
    box = OperantBox(subject=subject, schedule=FR(1))
    box.step(1.0)
    box.step(1.0)
    assert [(r.response, r.is_reinforced) for r in box.records] == [
        ("Grooming", False),
        ("Lever Press", True),
    ]


def test_overlay_punisher_is_read_as_punishment_not_reinforcement() -> None:
    subject = ScriptedSubject(["Lever Press"] * 3)
    box = OperantBox(subject=subject, schedule=Overlay(FR(3), FR(1)))
    for _ in range(3):
        box.step(1.0)
    assert [r.is_reinforced for r in box.records] == [False, False, True]
    assert [r.is_punished for r in box.records] == [True, True, True]
    assert [r.punishment for r in box.records] == [1.0, 1.0, 1.0]
    assert subject.punished == [1.0, 1.0, 1.0]
    assert subject.learned == [None, None, "Food Pellet"]


def test_sidman_shock_without_response_is_delivered_and_logged() -> None:
    subject = ScriptedSubject(["Nothing"] * 10)
    box = OperantBox(subject=subject, schedule=Sidman(shock_shock_interval=5.0,
                                                       response_shock_interval=20.0))
    for _ in range(10):
        box.step(1.0)
    assert box.records == []
    assert [t for t, _ in box.punishments] == [5.0, 10.0]
    assert subject.punished == [1.0, 1.0]


def test_response_resets_sidman_shock_clock() -> None:
    subject = ScriptedSubject(["Nothing", "Lever Press"] + ["Nothing"] * 8)
    box = OperantBox(subject=subject, schedule=Sidman(shock_shock_interval=5.0,
                                                       response_shock_interval=6.0))
    for _ in range(10):
        box.step(1.0)
    assert [t for t, _ in box.punishments] == [8.0]


def test_subject_without_punish_still_runs() -> None:
    subject = NoPunishSubject(["Lever Press"] * 2)
    box = OperantBox(subject=subject, schedule=Overlay(VI(100.0, seed=0), FR(1)))
    box.step(1.0)
    box.step(1.0)
    assert [r.is_punished for r in box.records] == [True, True]


def test_legacy_schedule_records_default_to_unpunished() -> None:
    subject = ScriptedSubject(["Lever Press"] * 2)
    box = OperantBox(subject=subject, schedule=FixedInterval(2.0))
    box.step(1.0)
    box.step(1.0)
    assert [r.is_reinforced for r in box.records] == [False, True]
    assert [(r.is_punished, r.punishment) for r in box.records] == [(False, 0.0)] * 2
    assert box.punishments == []
    assert subject.punished == []
