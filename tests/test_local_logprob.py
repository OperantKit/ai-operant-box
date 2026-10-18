"""Tests for the local-model log-probability subject (no model file needed).

A fake backend stands in for llama.cpp: it tokenizes character by character
and returns logits chosen by the test, so the probability arithmetic can be
checked exactly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ai_operant_box.agent.local_logprob import (
    LocalLogprobClient,
    LocalLogprobSubject,
    action_probabilities,
    chatml_prompt,
    p_response_by_time_since_reinforcer,
)
from ai_operant_box.chamber.environment import OperantBox
from ai_operant_box.chamber.schedule import FixedInterval

VOCAB = 128  # byte-level fake vocabulary


class FakeBackend:
    """Character-level tokenizer; logits favour a configurable next char."""

    def __init__(self, preferred: dict[str, float] | None = None) -> None:
        # bonus logit per next character, keyed by the character
        self.preferred = preferred or {}
        self.calls: list[list[int]] = []

    def tokenize(self, text: str) -> list[int]:
        return [ord(c) % VOCAB for c in text]

    def next_logits(self, tokens: list[int]) -> np.ndarray:
        self.calls.append(list(tokens))
        logits = np.zeros(VOCAB)
        for ch, bonus in self.preferred.items():
            logits[ord(ch) % VOCAB] += bonus
        return logits


def test_chatml_prompt_ends_with_open_assistant_turn():
    prompt = chatml_prompt("sys", [("user", "u1"), ("assistant", "a1"), ("user", "u2")])
    assert prompt.startswith("<|im_start|>system\nsys<|im_end|>\n")
    assert "<|im_start|>assistant\na1<|im_end|>\n" in prompt
    assert prompt.endswith("<|im_start|>user\nu2<|im_end|>\n<|im_start|>assistant\n")


def test_uniform_logits_give_sequence_probability_by_token_count():
    # With all-zero logits every token has probability 1/VOCAB, so an
    # action of n tokens has log-probability -n*log(VOCAB).
    client = LocalLogprobClient(FakeBackend(), system_prompt="s", actions=("ab", "c"))
    scores = client.score("x")
    assert scores.sequence_logprob["ab"] == pytest.approx(-2 * math.log(VOCAB))
    assert scores.sequence_logprob["c"] == pytest.approx(-1 * math.log(VOCAB))
    assert scores.first_token_logprob["ab"] == pytest.approx(-math.log(VOCAB))
    # renormalised over the two actions: 1/VOCAB^2 vs 1/VOCAB
    p_ab = 1 / (1 + VOCAB)
    assert scores.probabilities["ab"] == pytest.approx(p_ab)
    assert sum(scores.probabilities.values()) == pytest.approx(1.0)


def test_scoring_does_not_commit_and_commit_appends_assistant_turn():
    backend = FakeBackend()
    client = LocalLogprobClient(backend, system_prompt="s", actions=("a", "b"))
    client.score("u1")
    assert client.messages == [("user", "u1")]
    client.commit("a")
    client.score("u2")
    assert client.messages == [("user", "u1"), ("assistant", "a"), ("user", "u2")]
    # the last prompt evaluated contains the committed reply
    last_prompt = "".join(chr(t) for t in backend.calls[-1])
    assert "<|im_start|>assistant\na<|im_end|>" in last_prompt


def test_rejects_actions_with_identical_token_sequences():
    with pytest.raises(ValueError):
        LocalLogprobClient(FakeBackend(), system_prompt="s", actions=("a", "a"))


def test_temperature_flattens_and_sharpens():
    logp = {"a": math.log(0.9), "b": math.log(0.1)}
    p1 = action_probabilities(logp, temperature=1.0)
    hot = action_probabilities(logp, temperature=2.0)
    cold = action_probabilities(logp, temperature=0.5)
    assert p1["a"] == pytest.approx(0.9)
    assert 0.5 < hot["a"] < p1["a"] < cold["a"] < 1.0
    with pytest.raises(ValueError):
        action_probabilities(logp, temperature=0.0)


def test_client_temperature_applies_to_token_logits():
    # 'a' gets +2 logit; at higher temperature its share must shrink.
    backend = FakeBackend(preferred={"a": 2.0})
    p_t1 = LocalLogprobClient(backend, "s", actions=("a", "b"), temperature=1.0).score("u")
    p_t2 = LocalLogprobClient(backend, "s", actions=("a", "b"), temperature=2.0).score("u")
    assert p_t1.probabilities["a"] == pytest.approx(1 / (1 + math.exp(-2.0)))
    assert p_t2.probabilities["a"] == pytest.approx(1 / (1 + math.exp(-1.0)))


def test_subject_greedy_policy_and_history_records_probabilities():
    backend = FakeBackend(preferred={"L": 3.0})
    # single-token actions so that the +3 bonus on "L" decides the argmax
    client = LocalLogprobClient(backend, "s", actions=("L", "N"))
    subject = LocalLogprobSubject(client, policy="greedy")
    box = OperantBox(subject=subject, schedule=FixedInterval(3, target_response="L"))
    for _ in range(6):
        box.step(1.0)
    assert [h["r"] for h in subject.history] == ["L"] * 6
    expected = 1 / (1 + math.exp(-3.0))
    assert all(h["p"]["L"] == pytest.approx(expected) for h in subject.history)
    # FI 3 with a press every second: reinforced at t=3 and t=6
    assert [r.timestamp for r in box.records if r.is_reinforced] == [3.0, 6.0]
    # the second step's user message reports the first step's consequence
    assert client.messages[2] == ("user", "Result: No Reward\nStimulus: Light ON")


def test_subject_sample_policy_is_reproducible_with_seed():
    def run(seed: int) -> list[str]:
        client = LocalLogprobClient(FakeBackend(), "s", actions=("a", "b"))
        subject = LocalLogprobSubject(client, policy="sample", seed=seed)
        for _ in range(20):
            subject.sense("Light ON")
            subject.act()
            subject.learn(None)
        return [h["r"] for h in subject.history]

    assert run(1) == run(1)
    assert len(set(run(1))) == 2  # uniform over two actions: both appear in 20 draws


def test_p_response_by_time_since_reinforcer():
    # steps at t=1..6, reinforced at t=3; P(R) recorded per step
    times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    reinforced = [False, False, True, False, False, False]
    curve = p_response_by_time_since_reinforcer(times, p, reinforced)
    # session start counts as t0=0: steps 1,2,3 -> elapsed 1,2,3; then 4,5,6 -> 1,2,3
    assert curve == {1.0: pytest.approx(0.25), 2.0: pytest.approx(0.35), 3.0: pytest.approx(0.45)}


MODEL_ENV = "AOB_GGUF"


def _real_model_path() -> str | None:
    import os
    from pathlib import Path

    path = os.environ.get(MODEL_ENV)
    return path if path and Path(path).is_file() else None


@pytest.mark.integration
def test_llama_cpp_prefix_reuse_matches_fresh_evaluation():
    pytest.importorskip("llama_cpp")
    path = _real_model_path()
    if path is None:
        pytest.skip(f"no GGUF model (set {MODEL_ENV})")
    from ai_operant_box.agent.local_logprob import LlamaCppBackend

    cached = LocalLogprobClient(LlamaCppBackend(path, n_ctx=2048))
    for text in ("Stimulus: Light ON", "Result: Reward\nStimulus: Light ON"):
        cached.score(text)
        cached.commit("Lever Press")
    last = cached.score("Result: No Reward\nStimulus: Light ON")

    fresh = LocalLogprobClient(LlamaCppBackend(path, n_ctx=2048))
    fresh.messages = list(cached.messages[:-1])
    again = fresh.score("Result: No Reward\nStimulus: Light ON")
    for action in cached.actions:
        assert last.sequence_logprob[action] == pytest.approx(
            again.sequence_logprob[action], abs=0.05
        )
