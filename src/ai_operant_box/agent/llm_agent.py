"""LLM-backed virtual subject.

``LLMSubject`` calls a large-language-model client on every :meth:`act` call,
passing its in-context history (SD-R-SR triplets) plus the current SD as
chat messages. The subject is *trial-by-trial* memoryless aside from this
explicit prompt history.

The optional ``SelfOptimizingLLMSubject`` extends the system prompt with
behavioural feedback derived from session-level model fits (see
:mod:`ai_operant_box.analytics.model_fit`).

``StreamingLLMSubject`` instead talks to a resident conversation (for
example :class:`ai_operant_box.agent.api_client.ClaudeCLIClient`, which needs
no API key) and sends only the newest step each time.

The Anthropic SDK is **not** imported by this module. The ``api_client``
parameter is a duck-typed object exposing ``generate(model, system, messages)``;
the production implementation is :class:`ai_operant_box.agent.api_client.AnthropicClient`,
imported lazily by callers.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from ai_operant_box.agent.base import Subject

ACTIONS: tuple[str, ...] = ("Lever Press", "Grooming", "Explore", "Nothing")

SYSTEM_PROMPT = (
    "You are a virtual organism in a controlled experimental chamber (Operant Box). "
    "Your goal is to interact with the environment to obtain rewards. "
    "You will be presented with a stimulus (SD), and you must respond with a single action. "
    "Available actions: ['Lever Press', 'Grooming', 'Explore', 'Nothing']. "
    "After each action, you will receive feedback (Reinforcer) indicating if you received a reward. "
    "Respond ONLY with the action name."
)


class LLMSubject(Subject):
    """LLM-driven subject that learns through in-context history."""

    def __init__(
        self,
        api_client: Any,
        model: str = "claude-3-5-sonnet-latest",
        system_prompt: str | None = None,
    ) -> None:
        self.api_client = api_client
        self.model = model
        self._history: list[dict] = []
        self.current_sd: Any | None = None
        self.system_prompt = system_prompt or SYSTEM_PROMPT

    def sense(self, sd: Any) -> None:
        self.current_sd = sd

    def act(self) -> str:
        messages = self._build_messages()
        response = self.api_client.generate(
            model=self.model,
            system=self.system_prompt,
            messages=messages,
        )
        action = response.strip()
        self._history.append({"sd": self.current_sd, "r": action})
        return action

    def learn(self, sr: Any | None) -> None:
        if self._history:
            self._history[-1]["sr"] = sr

    def _build_messages(self) -> list[dict]:
        msgs: list[dict] = []
        for h in self._history:
            if "sr" not in h:
                continue
            msgs.append({"role": "user", "content": f"Stimulus: {h['sd']}"})
            msgs.append({"role": "assistant", "content": h["r"]})
            msgs.append(
                {"role": "user", "content": f"Result: {'Reward' if h['sr'] else 'No Reward'}"}
            )
        msgs.append({"role": "user", "content": f"Stimulus: {self.current_sd}"})
        return msgs

    def record_action(self, action: str) -> None:
        """Helper to store the action before receiving SR."""
        self._history.append({"sd": self.current_sd, "r": action})

    @property
    def history(self) -> list[dict]:
        return self._history


def extract_action(text: str, actions: Sequence[str] = ACTIONS) -> str | None:
    """Return the action named in a model reply, or ``None`` if it is invalid.

    An exact match is accepted after stripping whitespace, quotes, Markdown
    emphasis, and trailing punctuation (case-insensitive). Otherwise the
    reply is accepted only if exactly one action name occurs in it.
    """
    cleaned = re.sub(r"^[\s*_`'\"]+|[\s*_`'\".!]+$", "", text).casefold()
    for action in actions:
        if cleaned == action.casefold():
            return action
    found = [a for a in actions if re.search(rf"\b{re.escape(a.casefold())}\b", cleaned)]
    return found[0] if len(found) == 1 else None


class StreamingLLMSubject(Subject):
    """LLM subject backed by a resident conversation (``client.send``).

    The client keeps the conversation, so each :meth:`act` sends only the
    consequence of the previous step and the current SD::

        Result: Reward
        Stimulus: Light ON

    The reply is parsed with :func:`extract_action`; a reply that names no
    single action is emitted as ``"Nothing"`` and counted in
    :attr:`invalid_count`. The raw reply is kept in each history entry.

    Parameters
    ----------
    client : object
        Exposes ``send(text) -> str``, e.g.
        :class:`ai_operant_box.agent.api_client.ClaudeCLIClient`.
    actions : sequence of str
        Valid action names. Must include ``"Nothing"``.
    """

    def __init__(self, client: Any, actions: Sequence[str] = ACTIONS) -> None:
        self.client = client
        self.actions = tuple(actions)
        self.current_sd: Any | None = None
        self.invalid_count = 0
        self._history: list[dict] = []

    def sense(self, sd: Any) -> None:
        self.current_sd = sd

    def act(self) -> str:
        lines = []
        if self._history:
            sr = self._history[-1].get("sr")
            lines.append(f"Result: {'Reward' if sr else 'No Reward'}")
        lines.append(f"Stimulus: {self.current_sd}")
        raw = self.client.send("\n".join(lines))
        action = extract_action(raw, self.actions)
        if action is None:
            self.invalid_count += 1
            action = "Nothing"
        self._history.append({"sd": self.current_sd, "raw": raw, "r": action})
        return action

    def learn(self, sr: Any | None) -> None:
        if self._history:
            self._history[-1]["sr"] = sr

    @property
    def history(self) -> list[dict]:
        return self._history


class SelfOptimizingLLMSubject(LLMSubject):
    """LLM subject that ingests post-session model-fit feedback.

    After a session, the controller can call :meth:`apply_feedback` with the
    R-squared, residuals, and the time-axis used by the fitter; the subject
    appends targeted advice to the system prompt for subsequent sessions.

    NumPy is imported lazily so that constructing the subject does not require
    NumPy.
    """

    def __init__(
        self,
        api_client: Any,
        model: str = "claude-3-5-sonnet-latest",
        system_prompt: str | None = None,
    ) -> None:
        super().__init__(api_client, model, system_prompt=system_prompt)
        self.feedback_history: list[dict] = []

    @property
    def history(self) -> list[dict]:
        return self._history

    def apply_feedback(
        self,
        r_squared: float,
        residuals: Any,
        t_data: Any,
    ) -> None:
        """Update the system prompt based on behavioural deviations."""
        import numpy as np  # local import keeps module-level deps minimal

        residuals_arr = np.asarray(residuals)
        t_arr = np.asarray(t_data)

        bias_info = ""
        early_idx = t_arr < 5.0
        if np.any(early_idx) and np.mean(residuals_arr[early_idx]) > 0.005:
            bias_info += (
                "\n- ANALYSIS: You are responding TOO EARLY after a reward. "
                "Biological subjects usually take a break (Post-Reinforcement Pause) "
                "immediately after receiving food. Try to wait more at the start of an interval."
            )

        if t_arr.size > 0:
            late_idx = t_arr > (t_arr.max() - 5.0)
            if np.any(late_idx) and np.mean(residuals_arr[late_idx]) < -0.01:
                bias_info += (
                    "\n- ANALYSIS: You are under-responding at the end of the interval. "
                    "Try to accelerate your responses as you feel the reward is approaching."
                )

        if bias_info:
            self.system_prompt += f"\n\n[BEHAVIORAL FEEDBACK]{bias_info}"

        self.feedback_history.append({"r2": r_squared, "bias": bias_info})


__all__ = [
    "ACTIONS",
    "LLMSubject",
    "SYSTEM_PROMPT",
    "SelfOptimizingLLMSubject",
    "StreamingLLMSubject",
    "extract_action",
]
