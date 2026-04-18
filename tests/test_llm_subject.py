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
