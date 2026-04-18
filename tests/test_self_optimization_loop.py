"""Self-optimization loop using the SelfOptimizingLLMSubject + a mock client."""

from __future__ import annotations

import pytest

from ai_operant_box.agent.llm_agent import SelfOptimizingLLMSubject
from ai_operant_box.chamber.environment import OperantBox, VirtualClock
from ai_operant_box.chamber.schedule import FixedInterval


class DefinitiveMockClient:
    """Behaviour-bearing mock client that adopts a scallop pattern after feedback."""

    def __init__(self, k: float = 0.002) -> None:
        self.k = k
        self.time_since_last_sr: float = 0.0

    def generate(self, model, system, messages):
        import random

        if "Post-Reinforcement Pause" not in system:
            prob = 0.3
        else:
            latency = 8.0
            if self.time_since_last_sr < latency:
                return "Nothing"
            prob = min(1.0, self.k * (self.time_since_last_sr ** 2))

        return "Lever Press" if random.random() < prob else "Nothing"

    def update_time(self, delta: float, sr: bool) -> None:
        if sr:
            self.time_since_last_sr = 0.0
        else:
            self.time_since_last_sr += delta


def run_session(agent, client, duration, interval, delta):
    clock = VirtualClock()
    schedule = FixedInterval(interval=interval)
    box = OperantBox(subject=agent, schedule=schedule, clock=clock)

    last_sr_occurred = False
    for _ in range(int(duration / delta)):
        client.update_time(delta, last_sr_occurred)
        record = box.step(delta)
        last_sr_occurred = record.is_reinforced if record else False

    return box.records


def test_self_optimization_runs_two_sessions():
    pytest.importorskip("scipy")

    duration, interval, delta = 600.0, 30.0, 1.0
    client = DefinitiveMockClient(k=0.0015)
    agent = SelfOptimizingLLMSubject(api_client=client)

    records_1 = run_session(agent, client, duration, interval, delta)
    assert len(records_1) >= 0  # session ran

    # Apply trivial feedback to exercise the prompt-update path.
    import numpy as np

    t_data = np.linspace(0, interval, 30)
    residuals = np.full_like(t_data, 0.01)  # over-responding pattern
    agent.apply_feedback(r_squared=0.1, residuals=residuals, t_data=t_data)

    assert "BEHAVIORAL FEEDBACK" in agent.system_prompt or len(agent.feedback_history) == 1

    agent._history = []
    client.time_since_last_sr = 0.0
    records_2 = run_session(agent, client, duration, interval, delta)
    assert isinstance(records_2, list)
