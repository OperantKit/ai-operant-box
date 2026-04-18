"""End-to-end FI-scallop simulation (no PNG output, deterministic counts)."""

from __future__ import annotations

from ai_operant_box.agent.scallop_agent import ScallopingAgent
from ai_operant_box.chamber.environment import OperantBox, VirtualClock
from ai_operant_box.chamber.schedule import FixedInterval


def test_fi_scallop_simulation():
    clock = VirtualClock()
    schedule = FixedInterval(interval=30.0)
    agent = ScallopingAgent(k=0.005)
    box = OperantBox(subject=agent, schedule=schedule, clock=clock)

    duration = 600.0
    delta = 0.5
    steps = int(duration / delta)

    for _ in range(steps):
        agent.update_time(delta)
        box.step(delta)

    reinforcements = [r for r in box.records if r.is_reinforced]
    assert len(box.records) > 0
    assert len(reinforcements) > 0
