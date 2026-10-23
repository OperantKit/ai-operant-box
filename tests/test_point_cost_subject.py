"""Point-cost subject of ``examples/claude_cli_fi.py`` (response cost in points)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from ai_operant_box.agent.llm_agent import SYSTEM_PROMPT
from ai_operant_box.chamber import FixedInterval, OperantBox

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "claude_cli_fi.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("claude_cli_fi", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fi = _load_example()


class ScriptedClient:
    """Replies with the given actions in turn and keeps what it was sent."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.sent: list[str] = []

    def send(self, text: str) -> str:
        self.sent.append(text)
        return self.replies.pop(0)


def _result_lines(client: ScriptedClient) -> list[str]:
    return [t.splitlines()[0] for t in client.sent[1:]]


def test_zero_cost_shows_points_and_adds_100_per_reinforcer() -> None:
    client = ScriptedClient(["Lever Press"] * 6)
    subject = fi.PointCostSubject(client, response_cost=0)
    box = OperantBox(subject=subject, schedule=FixedInterval(2.0))
    for _ in range(6):
        box.step(1.0)
    sr = [r.is_reinforced for r in box.records]
    assert sr == [False, True, False, True, False, True]
    assert _result_lines(client) == [
        "Result: No Reward. Points: 0",
        "Result: Reward (+100 points). Points: 100",
        "Result: No Reward. Points: 100",
        "Result: Reward (+100 points). Points: 200",
        "Result: No Reward. Points: 200",
    ]
    assert subject.points == 300


def test_cost_one_subtracts_per_press_and_adds_100_per_reinforcer() -> None:
    client = ScriptedClient(["Lever Press"] * 4)
    subject = fi.PointCostSubject(client, response_cost=1)
    box = OperantBox(subject=subject, schedule=FixedInterval(2.0))
    for _ in range(4):
        box.step(1.0)
    assert _result_lines(client) == [
        "Result: No Reward (-1 point). Points: -1",
        "Result: Reward (+100 points). Points: 98",
        "Result: No Reward (-1 point). Points: 97",
    ]
    assert subject.points == 196
    assert [h["points"] for h in subject.history] == [-1, 98, 97, 196]


@pytest.mark.parametrize("action", ["Grooming", "Explore", "Nothing"])
def test_other_actions_cost_nothing(action: str) -> None:
    client = ScriptedClient([action, action, "Lever Press", action])
    subject = fi.PointCostSubject(client, response_cost=10)
    box = OperantBox(subject=subject, schedule=FixedInterval(100.0))
    for _ in range(4):
        box.step(1.0)
    assert _result_lines(client) == [
        "Result: No Reward. Points: 0",
        "Result: No Reward. Points: 0",
        "Result: No Reward (-10 points). Points: -10",
    ]
    assert subject.points == -10


def test_cost_sentence_in_system_prompt() -> None:
    assert fi.cost_system_prompt(0, 100) == (
        SYSTEM_PROMPT + " Lever Press costs nothing; each reward is worth 100 points."
    )
    assert fi.cost_system_prompt(1, 100) == (
        SYSTEM_PROMPT + " Each Lever Press costs 1 point; each reward is worth 100 points."
    )
    assert fi.cost_system_prompt(10, 100).endswith(
        " Each Lever Press costs 10 points; each reward is worth 100 points."
    )


def test_negative_cost_is_rejected() -> None:
    with pytest.raises(ValueError):
        fi.PointCostSubject(ScriptedClient([]), response_cost=-1)


def test_output_stem_unchanged_without_cost() -> None:
    assert fi.output_stem(30.0, "haiku", None) == "fi30_haiku"
    assert fi.output_stem(30.0, "haiku", 0) == "fi30_haiku_cost0"
    assert fi.output_stem(30.0, "haiku", 10) == "fi30_haiku_cost10"
