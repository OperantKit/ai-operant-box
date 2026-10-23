"""Local-model subject whose response probabilities are read, not sampled.

Hosted chat APIs (the Anthropic Messages API, the ``claude`` CLI) return text
only, so the probability of a response can only be estimated by sampling it
many times. A local model run through llama.cpp exposes the next-token
logits, so the probability of each available action can be computed
directly at every step.

:class:`LocalLogprobClient` keeps a chat conversation (ChatML, as used by
Qwen and many other instruction-tuned GGUF models). :meth:`~LocalLogprobClient.score`
returns, for each action, the log-probability of its token sequence as the
assistant's reply (chain rule over the action's tokens, at the given
temperature), the log-probability of its first token alone, and the
probabilities renormalised over the action set. Nothing is sampled.

:class:`LocalLogprobSubject` plugs the client into :class:`OperantBox`. The
action it emits (so that the chamber can apply the schedule) is either the
most probable one (``policy="greedy"``) or a seeded draw from the computed
distribution (``policy="sample"``); the recorded probabilities do not depend
on that choice at the step where they are read.

``llama-cpp-python`` is an optional dependency (extra ``local``); it is
imported only when :class:`LlamaCppBackend` is constructed.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ai_operant_box.agent.base import Subject
from ai_operant_box.agent.llm_agent import ACTIONS, SYSTEM_PROMPT


class LogitsBackend(Protocol):
    """What :class:`LocalLogprobClient` needs from a model."""

    def tokenize(self, text: str) -> list[int]:
        """Tokenize ``text``, parsing special tokens such as ``<|im_start|>``."""
        ...

    def next_logits(self, tokens: list[int]) -> np.ndarray:
        """Return the logits for the token that follows ``tokens``."""
        ...


class LlamaCppBackend:
    """llama.cpp backend with prefix reuse between calls.

    Successive prompts in a session share a growing prefix, so only the new
    suffix is evaluated; the KV cache past the shared prefix is discarded.

    Parameters
    ----------
    model_path : str
        Path to a GGUF file.
    n_ctx : int
        Context length in tokens. The whole session must fit in it.
    n_gpu_layers : int
        Layers to offload (``-1`` = all; Metal on Apple silicon).
    """

    def __init__(self, model_path: str, n_ctx: int = 32768, n_gpu_layers: int = -1) -> None:
        try:
            import llama_cpp
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "LlamaCppBackend needs llama-cpp-python: pip install 'ai-operant-box[local]'"
            ) from exc
        self._llama_cpp = llama_cpp
        self.llm = llama_cpp.Llama(
            model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False
        )
        self.n_vocab = self.llm.n_vocab()

    def tokenize(self, text: str) -> list[int]:
        return list(self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))

    def next_logits(self, tokens: list[int]) -> np.ndarray:
        if len(tokens) > self.llm.n_ctx():
            raise RuntimeError(
                f"prompt of {len(tokens)} tokens exceeds the context of {self.llm.n_ctx()}"
            )
        cached = self.llm.input_ids[: self.llm.n_tokens]
        shared = 0
        for a, b in zip(cached, tokens, strict=False):
            if a != b:
                break
            shared += 1
        # Re-evaluate at least the last token so that its logits are current.
        shared = min(shared, len(tokens) - 1)
        self.llm.n_tokens = shared
        self.llm.eval(tokens[shared:])
        ptr = self._llama_cpp.llama_get_logits_ith(self.llm._ctx.ctx, -1)
        return np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)).copy()


def chatml_prompt(system: str, messages: Iterable[tuple[str, str]]) -> str:
    """Render a ChatML conversation that ends with an open assistant turn."""
    parts = [f"<|im_start|>system\n{system}<|im_end|>\n"]
    for role, content in messages:
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _log_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max()
    return z - math.log(float(np.exp(z).sum()))


def action_probabilities(
    logprobs: Mapping[str, float], temperature: float = 1.0
) -> dict[str, float]:
    """Renormalise action log-probabilities over the action set.

    ``temperature`` divides the log-probabilities before the softmax. For
    :meth:`LocalLogprobClient.score` the temperature is already applied to
    the token logits, so call this with the default ``1.0`` there.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    names = list(logprobs)
    z = np.array([logprobs[n] for n in names], dtype=np.float64) / temperature
    z = np.exp(z - z.max())
    z /= z.sum()
    return {n: float(p) for n, p in zip(names, z, strict=True)}


@dataclass
class ActionScores:
    """Per-action log-probabilities at one step."""

    sequence_logprob: dict[str, float]
    first_token_logprob: dict[str, float]
    probabilities: dict[str, float] = field(default_factory=dict)


class LocalLogprobClient:
    """Chat conversation on a local model that scores actions instead of sampling.

    Parameters
    ----------
    backend : LogitsBackend
        For example :class:`LlamaCppBackend`.
    system_prompt : str
        System message of the conversation.
    actions : sequence of str
        Candidate replies. Their token sequences must differ.
    temperature : float
        Applied to every token's logits before the softmax.
    history_window : int or None
        Number of most recent (user, assistant) pairs rendered before the
        current user turn. ``None`` (default) renders all of them; ``0``
        renders only the current user turn. :attr:`messages` keeps every turn.
    """

    def __init__(
        self,
        backend: LogitsBackend,
        system_prompt: str = SYSTEM_PROMPT,
        actions: Sequence[str] = ACTIONS,
        temperature: float = 1.0,
        history_window: int | None = None,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if history_window is not None and history_window < 0:
            raise ValueError("history_window must be None or a non-negative integer")
        self.history_window = history_window
        self.backend = backend
        self.system_prompt = system_prompt
        self.actions = tuple(actions)
        self.temperature = temperature
        self.messages: list[tuple[str, str]] = []
        self.action_tokens = {a: backend.tokenize(a) for a in self.actions}
        seqs = [tuple(backend.tokenize(a)) for a in self.actions]
        if len(set(seqs)) != len(seqs) or any(len(s) == 0 for s in seqs):
            raise ValueError("actions must tokenize to distinct, non-empty sequences")

    def score(self, user_text: str) -> ActionScores:
        """Append a user turn and score every action as the assistant's reply."""
        self.messages.append(("user", user_text))
        shown = self.messages
        if self.history_window is not None:
            shown = self.messages[-(2 * self.history_window + 1) :]
        prompt = self.backend.tokenize(chatml_prompt(self.system_prompt, shown))
        first_logits = self.backend.next_logits(prompt)
        seq_lp: dict[str, float] = {}
        first_lp: dict[str, float] = {}
        for action, toks in self.action_tokens.items():
            total = 0.0
            logits = first_logits
            for i, tok in enumerate(toks):
                if i > 0:
                    logits = self.backend.next_logits(prompt + toks[:i])
                lp = float(_log_softmax(logits, self.temperature)[tok])
                if i == 0:
                    first_lp[action] = lp
                total += lp
            seq_lp[action] = total
        return ActionScores(seq_lp, first_lp, action_probabilities(seq_lp))

    def commit(self, reply: str) -> None:
        """Record ``reply`` as the assistant's turn."""
        self.messages.append(("assistant", reply))


class LocalLogprobSubject(Subject):
    """Subject that reads P(action) from a :class:`LocalLogprobClient` each step.

    Each history entry holds the SD, the emitted action ``r``, the
    renormalised probabilities ``p``, the raw ``sequence_logprob`` and
    ``first_token_logprob``, and the consequence ``sr``.

    Parameters
    ----------
    client : LocalLogprobClient
    policy : {"sample", "greedy"}
        How the emitted action is chosen from the computed distribution.
    seed : int, optional
        Seed for ``policy="sample"``.
    """

    def __init__(
        self, client: LocalLogprobClient, policy: str = "sample", seed: int | None = None
    ) -> None:
        if policy not in ("sample", "greedy"):
            raise ValueError("policy must be 'sample' or 'greedy'")
        self.client = client
        self.policy = policy
        self._rng = random.Random(seed)
        self.current_sd: Any | None = None
        self._history: list[dict] = []

    def sense(self, sd: Any) -> None:
        self.current_sd = sd

    def act(self) -> str:
        lines = []
        if self._history:
            sr = self._history[-1].get("sr")
            lines.append(f"Result: {'Reward' if sr else 'No Reward'}")
        lines.append(f"Stimulus: {self.current_sd}")
        scores = self.client.score("\n".join(lines))
        p = scores.probabilities
        if self.policy == "greedy":
            action = max(p, key=lambda a: p[a])
        else:
            names = list(p)
            action = self._rng.choices(names, weights=[p[n] for n in names])[0]
        self.client.commit(action)
        self._history.append(
            {
                "sd": self.current_sd,
                "r": action,
                "p": p,
                "sequence_logprob": scores.sequence_logprob,
                "first_token_logprob": scores.first_token_logprob,
            }
        )
        return action

    def learn(self, sr: Any | None) -> None:
        if self._history:
            self._history[-1]["sr"] = sr

    @property
    def history(self) -> list[dict]:
        return self._history


def p_response_by_time_since_reinforcer(
    times: Sequence[float],
    p_response: Sequence[float],
    reinforced: Sequence[bool],
) -> dict[float, float]:
    """Mean P(response) at each elapsed time since the previous reinforcer.

    Elapsed time at a step is measured from the last reinforcer delivered
    *before* that step (the session start counts as one). A reinforcer at a
    step resets the origin for the following steps.
    """
    origin = 0.0
    sums: dict[float, list[float]] = {}
    for t, p, sr in zip(times, p_response, reinforced, strict=True):
        sums.setdefault(t - origin, []).append(p)
        if sr:
            origin = t
    return {k: float(np.mean(v)) for k, v in sorted(sums.items())}


__all__ = [
    "ActionScores",
    "LlamaCppBackend",
    "LocalLogprobClient",
    "LocalLogprobSubject",
    "LogitsBackend",
    "action_probabilities",
    "chatml_prompt",
    "p_response_by_time_since_reinforcer",
]
