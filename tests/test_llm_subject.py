"""LLMSubject behaviour with a mock client (no Anthropic SDK required)."""

from __future__ import annotations

from ai_operant_box.agent.llm_agent import LLMSubject


class MockClient:
    def __init__(self, response: str = "Lever Press") -> None:
        self.response = response
        self.calls: list = []

    def generate(self, model, system, messages):
        self.calls.append({"model": model, "system": system, "messages": list(messages)})
        return self.response


def test_llm_subject_emits_response_on_act():
    client = MockClient(response="Lever Press")
    subject = LLMSubject(api_client=client)
    subject.sense("Light ON")
    action = subject.act()
    assert action == "Lever Press"
    assert len(subject.history) == 1
    assert subject.history[0]["sd"] == "Light ON"


def test_llm_subject_stores_consequence():
    client = MockClient(response="Lever Press")
    subject = LLMSubject(api_client=client)
    subject.sense("Light ON")
    subject.act()
    subject.learn("Food Pellet")
    assert subject.history[-1]["sr"] == "Food Pellet"


def test_llm_subject_includes_history_in_prompt():
    client = MockClient(response="Lever Press")
    subject = LLMSubject(api_client=client)

    # Step 1
    subject.sense("Light ON")
    subject.act()
    subject.learn("Food Pellet")

    # Step 2 — second call should include the previous SD-R-SR triplet.
    subject.sense("Tone ON")
    subject.act()
    last_call = client.calls[-1]
    contents = [m["content"] for m in last_call["messages"]]
    # Previous SD, action, and result must appear
    assert any("Light ON" in c for c in contents)
    assert any("Lever Press" in c for c in contents)
    assert any("Reward" in c for c in contents)
    # Current SD must appear
    assert contents[-1] == "Stimulus: Tone ON"


def test_llm_subject_default_system_prompt_is_used():
    client = MockClient()
    subject = LLMSubject(api_client=client)
    subject.sense("X")
    subject.act()
    assert "operant" in client.calls[-1]["system"].lower()


def test_llm_subject_custom_system_prompt():
    client = MockClient()
    subject = LLMSubject(api_client=client, system_prompt="custom-prompt")
    subject.sense("X")
    subject.act()
    assert client.calls[-1]["system"] == "custom-prompt"


def _run_completed_trials(subject: LLMSubject, client: MockClient, n: int) -> None:
    """Complete ``n`` trials: every 6th rewarded (3 of 20), presses on 12 of 20."""
    for i in range(n):
        client.response = "Lever Press" if i % 5 < 3 else "Explore"
        subject.sense("Light ON")
        subject.act()
        subject.learn("Food Pellet" if i % 6 == 5 else None)
    client.response = "Lever Press"
    subject.sense("Light ON")
    subject.act()


def test_history_window_keeps_only_recent_trials():
    client = MockClient()
    subject = LLMSubject(api_client=client, history_window=5)
    _run_completed_trials(subject, client, 20)
    messages = client.calls[-1]["messages"]
    assert len(messages) == 3 * 5 + 1
    assert messages[-1]["content"] == "Stimulus: Light ON"
    # the 5 trials shown are the last 5 (indices 15..19): actions P P P E E
    shown = [m["content"] for m in messages if m["role"] == "assistant"]
    assert shown == ["Lever Press", "Lever Press", "Lever Press", "Explore", "Explore"]


def test_history_window_none_keeps_all_trials():
    client = MockClient()
    subject = LLMSubject(api_client=client)
    _run_completed_trials(subject, client, 20)
    assert len(client.calls[-1]["messages"]) == 61


def test_history_window_zero_shows_only_current_stimulus():
    client = MockClient()
    subject = LLMSubject(api_client=client, history_window=0)
    _run_completed_trials(subject, client, 20)
    assert client.calls[-1]["messages"] == [{"role": "user", "content": "Stimulus: Light ON"}]


def test_negative_history_window_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        LLMSubject(api_client=MockClient(), history_window=-1)


def test_summary_only_replaces_trials_with_counts_in_system():
    client = MockClient()
    subject = LLMSubject(api_client=client, system_prompt="base", summary_only=True)
    _run_completed_trials(subject, client, 20)
    last = client.calls[-1]
    assert last["messages"] == [{"role": "user", "content": "Stimulus: Light ON"}]
    assert last["system"] == (
        "base\n\nSummary so far: 3 rewards received, 12 Lever Press responses made."
    )
    # the stored system prompt is not modified
    assert subject.system_prompt == "base"
