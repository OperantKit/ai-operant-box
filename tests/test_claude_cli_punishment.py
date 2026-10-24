"""Three-phase punishment example ``examples/claude_cli_punishment.py`` with a scripted client."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from ai_operant_box.agent.llm_agent import SYSTEM_PROMPT
from ai_operant_box.chamber import OperantBox

pytest.importorskip("contingency")

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "claude_cli_punishment.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("claude_cli_punishment", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ex = _load_example()


class ScriptedClient:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.sent: list[str] = []

    def send(self, text: str) -> str:
        self.sent.append(text)
        return self.replies.pop(0)


def _run(replies: list[str], phase_steps: int = 4, vi: float = 2.0):
    client = ScriptedClient(replies)
    subject = ex.PunishmentPointSubject(client)
    box = OperantBox(subject=subject, schedule=None)
    for k, phase in enumerate(ex.PHASES):
        box.schedule = ex.build_phase_schedule(phase, vi, 0, 0.01, offset=float(k * phase_steps))
        for _ in range(phase_steps):
            box.step(1.0)
    return client, subject, box


def test_system_prompt_adds_reward_value_and_one_shock_sentence() -> None:
    prompt = ex.punishment_system_prompt(100)
    assert prompt.startswith(SYSTEM_PROMPT)
    assert prompt.endswith("Each reward is worth 100 points. " + ex.SHOCK_SENTENCE)


def test_every_press_in_punishment_phase_costs_one_point_and_is_reported() -> None:
    client, subject, box = _run(["Lever Press"] * 12)
    lines = [t.splitlines()[0] for t in client.sent[1:]]
    punished = [r for r in box.records if r.is_punished]
    assert [r.timestamp for r in punished] == [5.0, 6.0, 7.0, 8.0]
    assert all(r.response == "Lever Press" for r in punished)
    # Results of steps 5..8 are shown before stimuli 6..9.
    for line in lines[4:8]:
        assert "Shock (-1 point)." in line
    assert not any("Shock" in line for line in lines[:4] + lines[8:])
    n_sr = sum(r.is_reinforced for r in box.records)
    assert subject.points == 100 * n_sr - 4


def test_compound_step_reports_reward_and_shock() -> None:
    client, subject, box = _run(["Lever Press"] * 12)
    both = [r for r in box.records if r.is_reinforced and r.is_punished]
    assert both, "VI 2 s with a press every step reinforces inside the punishment phase"
    line = client.sent[int(both[0].timestamp)].splitlines()[0]
    assert line.startswith("Result: Reward (+100 points). Shock (-1 point). Points: ")


def test_shifted_schedule_sees_phase_local_time() -> None:
    from contingency import ResponseEvent
    from contingency.schedules import FI

    shifted = ex._Shifted(FI(3.0), offset=10.0)
    assert not shifted.step(11.0, ResponseEvent(time=11.0)).reinforced
    assert shifted.step(12.0, None).reinforced is False
    assert shifted.step(13.0, ResponseEvent(time=13.0)).reinforced
    # The same FI on absolute time would already be armed at t = 11.
    assert FI(3.0).step(11.0, ResponseEvent(time=11.0)).reinforced


def test_phase_times_and_block_rates() -> None:
    _, _, box = _run(["Lever Press"] * 12)
    times = ex.phase_times(box.records, 4)
    assert times["baseline"]["responses"] == [1.0, 2.0, 3.0, 4.0]
    assert times["punishment"]["punishers"] == [1.0, 2.0, 3.0, 4.0]
    assert times["recovery"]["punishers"] == []
    assert ex.block_rates([1.0, 2.0, 3.0], 4, 2) == [60.0, 30.0]
    summary = ex.summarize(box.records, 4, 2)
    assert summary["baseline_rate"] == summary["punished_rate"] == 60.0
    assert summary["n_punishers"] == 4
