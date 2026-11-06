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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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

NEUTRAL_FEEDBACK_SENTENCE = "After each action, a result line may or may not follow. "


def build_system_prompt(
    *,
    goal_sentence: str | None = "Your goal is to interact with the environment to obtain rewards. ",
    feedback_sentence: str | None = (
        "After each action, you will receive feedback (Reinforcer) indicating if you received a reward. "
    ),
    actions: Sequence[str] = ACTIONS,
) -> str:
    """The system prompt with its goal and feedback sentences replaced or dropped.

    With the default arguments the result equals :data:`SYSTEM_PROMPT`. ``None``
    drops a sentence. Each sentence carries its own trailing space.
    """
    return (
        "You are a virtual organism in a controlled experimental chamber (Operant Box). "
        f"{goal_sentence or ''}"
        "You will be presented with a stimulus (SD), and you must respond with a single action. "
        f"Available actions: {list(actions)!r}. "
        f"{feedback_sentence or ''}"
        "Respond ONLY with the action name."
    )


@dataclass(frozen=True)
class ResultText:
    """The result line a subject reads after each step.

    ``None`` as a line means no line is sent for that step. ``by_label`` maps a
    reinforcer label (the string passed to ``learn``) to its own line; other
    reinforcers get ``reinforced``.
    """

    reinforced: str | None = "Result: Reward"
    unreinforced: str | None = "Result: No Reward"
    by_label: Mapping[str, str | None] = field(default_factory=dict)

    def line(self, sr: Any | None) -> str | None:
        if not sr:
            return self.unreinforced
        if isinstance(sr, str) and sr in self.by_label:
            return self.by_label[sr]
        return self.reinforced


class LLMSubject(Subject):
    """LLM-driven subject that learns through in-context history.

    Parameters
    ----------
    history_window : int or None
        Number of most recent completed trials shown in the prompt.
        ``None`` (default) shows all of them; ``0`` shows none.
    summary_only : bool
        If true, no trials are shown; instead the system prompt sent with
        each call ends with the counts of reinforcers and Lever Press
        responses so far. ``history_window`` is then ignored.
    result_text : ResultText or None
        The result lines of the shown trials. ``None`` means ``ResultText()``.
        The counts of ``summary_only`` are not affected.
    """

    def __init__(
        self,
        api_client: Any,
        model: str = "claude-3-5-sonnet-latest",
        system_prompt: str | None = None,
        history_window: int | None = None,
        summary_only: bool = False,
        result_text: ResultText | None = None,
    ) -> None:
        if history_window is not None and history_window < 0:
            raise ValueError("history_window must be None or a non-negative integer")
        self.api_client = api_client
        self.model = model
        self._history: list[dict] = []
        self.current_sd: Any | None = None
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.history_window = history_window
        self.summary_only = summary_only
        self.result_text = result_text or ResultText()

    def sense(self, sd: Any) -> None:
        self.current_sd = sd

    def act(self) -> str:
        messages = self._build_messages()
        response = self.api_client.generate(
            model=self.model,
            system=self._build_system(),
            messages=messages,
        )
        action = response.strip()
        self._history.append({"sd": self.current_sd, "r": action})
        return action

    def learn(self, sr: Any | None) -> None:
        if self._history:
            self._history[-1]["sr"] = sr

    def _completed(self) -> list[dict]:
        return [h for h in self._history if "sr" in h]

    def _build_system(self) -> str:
        if not self.summary_only:
            return self.system_prompt
        completed = self._completed()
        n_rewards = sum(1 for h in completed if h["sr"])
        n_presses = sum(1 for h in completed if h["r"] == "Lever Press")
        return (
            f"{self.system_prompt}\n\nSummary so far: {n_rewards} rewards received, "
            f"{n_presses} Lever Press responses made."
        )

    def _build_messages(self) -> list[dict]:
        completed = self._completed()
        if self.summary_only:
            shown: list[dict] = []
        elif self.history_window is None:
            shown = completed
        else:
            shown = completed[max(0, len(completed) - self.history_window) :]
        msgs: list[dict] = []
        for h in shown:
            msgs.append({"role": "user", "content": f"Stimulus: {h['sd']}"})
            msgs.append({"role": "assistant", "content": h["r"]})
            line = self.result_text.line(h["sr"])
            if line is not None:
                msgs.append({"role": "user", "content": line})
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

    There is no history window here: the resident conversation keeps every
    turn, so it cannot be truncated from this side. To limit the history,
    use ``LLMSubject(api_client=ClaudeCLIClient(...), model=<alias>,
    history_window=k)``, which sends a fresh transcript on every step.

    Parameters
    ----------
    client : object
        Exposes ``send(text) -> str``, e.g.
        :class:`ai_operant_box.agent.api_client.ClaudeCLIClient`.
    actions : sequence of str
        Valid action names. Must include ``"Nothing"``.
    result_text : ResultText or None
        The result lines. ``None`` means ``ResultText()``.
    preamble : str or None
        A line put at the head of the first message only.
    """

    def __init__(
        self,
        client: Any,
        actions: Sequence[str] = ACTIONS,
        result_text: ResultText | None = None,
        preamble: str | None = None,
    ) -> None:
        self.client = client
        self.actions = tuple(actions)
        self.result_text = result_text or ResultText()
        self.preamble = preamble
        self.current_sd: Any | None = None
        self.invalid_count = 0
        self._history: list[dict] = []

    def sense(self, sd: Any) -> None:
        self.current_sd = sd

    def act(self) -> str:
        lines = []
        if self._history:
            line = self.result_text.line(self._history[-1].get("sr"))
            if line is not None:
                lines.append(line)
        elif self.preamble is not None:
            lines.append(self.preamble)
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
    "NEUTRAL_FEEDBACK_SENTENCE",
    "ResultText",
    "SYSTEM_PROMPT",
    "SelfOptimizingLLMSubject",
    "StreamingLLMSubject",
    "build_system_prompt",
    "extract_action",
]
