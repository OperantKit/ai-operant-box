"""Smoke test of the SD-R-SR loop driven by ``OperantBox.step()``."""

from __future__ import annotations

from typing import Any, List, Optional

from ai_operant_box.agent.base import Subject
from ai_operant_box.chamber.environment import OperantBox
from ai_operant_box.chamber.schedule import ContinuousReinforcement


class MockSubject(Subject):
    def __init__(self) -> None:
        self._history: List[dict] = []
        self.last_sd: Any = None
        self.last_sr: Any = None

    def sense(self, sd: Any) -> None:
        self.last_sd = sd

    def act(self) -> str:
        return "Lever Press"

    def learn(self, sr: Optional[Any]) -> None:
        self.last_sr = sr
        self._history.append({"sd": self.last_sd, "r": "Lever Press", "sr": sr})

    @property
    def history(self) -> List[dict]:
        return self._history


def test_operant_loop():
    subject = MockSubject()
    schedule = ContinuousReinforcement(target_response="Lever Press")
    box = OperantBox(subject, schedule)

    record = box.step()

    assert record is not None
    assert record.response == "Lever Press"
    assert record.is_reinforced is True
    assert subject.last_sr == "Food Pellet"
    assert len(box.records) == 1


def test_extinction_loop():
    subject = MockSubject()
    schedule = ContinuousReinforcement(target_response="Nothing")
    box = OperantBox(subject, schedule)

    record = box.step()

    assert record is not None
    assert record.response == "Lever Press"
    assert record.is_reinforced is False
    assert subject.last_sr is None
