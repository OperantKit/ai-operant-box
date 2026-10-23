"""EXT — no response is reinforced; a CRF→EXT switch mid-session."""

from typing import Any

from ai_operant_box.agent.base import Subject
from ai_operant_box.chamber import ContinuousReinforcement, Extinction, OperantBox


class _AlwaysPress(Subject):
    def sense(self, discriminative_stimulus: Any) -> None:
        pass

    def act(self) -> str:
        return "Lever Press"

    def learn(self, reinforcer: Any | None) -> None:
        pass

    @property
    def history(self) -> list[dict]:
        return []


def test_extinction_never_reinforces() -> None:
    ext = Extinction()
    assert not any(ext.is_reinforced("Lever Press", float(t)) for t in range(100))
    assert not ext.is_reinforced("Grooming", 1.0)


def test_switching_box_from_crf_to_extinction() -> None:
    box = OperantBox(subject=_AlwaysPress(), schedule=ContinuousReinforcement())
    for _ in range(5):
        box.step(1.0)
    box.schedule = Extinction()
    for _ in range(5):
        box.step(1.0)
    assert [r.is_reinforced for r in box.records] == [True] * 5 + [False] * 5
