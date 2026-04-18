"""Multi-session comparison test (skips PNG rendering when matplotlib missing)."""

from __future__ import annotations

import pytest

from ai_operant_box.agent.scallop_agent import ScallopingAgent
from ai_operant_box.chamber.environment import OperantBox, VirtualClock
from ai_operant_box.chamber.schedule import FixedInterval


def run_session(interval: float, duration: float = 600.0):
    clock = VirtualClock()
    schedule = FixedInterval(interval=interval)
    agent = ScallopingAgent(k=0.005)
    box = OperantBox(subject=agent, schedule=schedule, clock=clock)

    delta = 0.5
    steps = int(duration / delta)
    for _ in range(steps):
        agent.update_time(delta)
        box.step(delta)

    return box.records


def test_multi_session_records_have_data():
    records_fi30 = run_session(30.0)
    records_fi60 = run_session(60.0)

    assert len(records_fi30) > 0
    assert len(records_fi60) > 0


def test_multi_session_recorder_renders_png(tmp_path):
    pytest.importorskip("matplotlib")

    from ai_operant_box.analytics.record import MultiSessionRecorder

    records_fi30 = run_session(30.0, duration=120.0)
    records_fi60 = run_session(60.0, duration=120.0)

    recorder = MultiSessionRecorder()
    recorder.add_session("FI 30s", records_fi30)
    recorder.add_session("FI 60s", records_fi60)

    output_path = tmp_path / "comparison.png"
    recorder.save_comparison_png(str(output_path), title="FI 30s vs FI 60s")

    assert output_path.exists()
