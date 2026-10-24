"""Three-phase resurgence procedure of ``examples/claude_cli_resurgence.py``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("contingency.schedules.phased")
pytest.importorskip("session_analyzer.analytics.resurgence")

from ai_operant_box.agent.llm_agent import StreamingLLMSubject  # noqa: E402
from ai_operant_box.chamber.concurrent import (  # noqa: E402
    CONCURRENT_ACTIONS,
    ChoiceRecord,
    ConcurrentBox,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "claude_cli_resurgence.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("claude_cli_resurgence", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rs = _load_example()


class ScriptedClient:
    """Replies with the given actions in turn."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)

    def send(self, text: str) -> str:
        return self.replies.pop(0)


def _run_alternating(alt_vi: float | None, seed: int = 0) -> ConcurrentBox:
    steps = rs.PHASE_STEPS
    total = sum(steps)
    # Runs of 5 presses per lever, so the 2 s COD does not suppress every press.
    replies = ["Left Lever" if (i // 5) % 2 == 0 else "Right Lever" for i in range(total)]
    subject = StreamingLLMSubject(ScriptedClient(replies), actions=CONCURRENT_ACTIONS)
    box = ConcurrentBox(subject, rs.build_schedule(alt_vi=alt_vi, seed=seed))
    box.run(total)
    return box


def test_default_phases_are_120_80_20_steps() -> None:
    assert rs.PHASE_STEPS == (120, 80, 20)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reinforcers_follow_the_phase_contingencies(seed: int) -> None:
    box = _run_alternating(alt_vi=15.0, seed=seed)
    p1, p2, _ = rs.PHASE_STEPS
    sr = [r for r in box.records if r.is_reinforced]
    phase1 = [r for r in sr if r.timestamp <= p1]
    phase2 = [r for r in sr if p1 < r.timestamp <= p1 + p2]
    phase3 = [r for r in sr if r.timestamp > p1 + p2]
    assert phase1 and all(r.operandum == "left" for r in phase1)
    assert phase2 and all(r.operandum == "right" for r in phase2)
    assert phase3 == []


def test_extinction_alternative_gives_no_reinforcer_in_phase_2() -> None:
    box = _run_alternating(alt_vi=None)
    p1 = rs.PHASE_STEPS[0]
    assert not [r for r in box.records if r.is_reinforced and r.timestamp > p1]


def _rec(t: float, operandum: str) -> ChoiceRecord:
    return ChoiceRecord(timestamp=t, operandum=operandum, is_reinforced=False, cod_suppressed=False)


def test_block_rates_are_responses_per_minute_per_20_steps() -> None:
    # 10 left presses in the first block (t = 1..10), 1 right press in the last block.
    records = [_rec(float(t), "left") for t in range(1, 11)] + [_rec(220.0, "right")]
    rates = rs.block_rates(records)
    assert [len(rates["left"][k]) for k in ("phase1", "phase2", "phase3")] == [6, 4, 1]
    assert rates["left"]["phase1"][0] == pytest.approx(30.0)  # 10 per 20 s
    assert rates["left"]["phase1"][1:] == [0.0] * 5
    assert rates["right"]["phase3"] == [pytest.approx(3.0)]


def test_metrics_use_the_last_two_blocks_of_phase_2() -> None:
    left = {"phase1": [30.0] * 6, "phase2": [12.0, 9.0, 6.0, 3.0], "phase3": [15.0]}
    right = {"phase1": [0.0] * 6, "phase2": [30.0] * 4, "phase3": [6.0]}
    m = rs.resurgence_metrics({"left": left, "right": right})
    assert m["baseline_target_rate"] == pytest.approx(30.0)
    assert m["suppression_target_rate"] == pytest.approx(4.5)
    assert m["resurgence_magnitude"] == pytest.approx(10.5)
    assert m["resurged"] is True
