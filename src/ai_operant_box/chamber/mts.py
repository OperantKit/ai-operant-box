"""Matching-to-sample chamber for a subject held in a resident conversation.

One conversation turn is one trial: the subject sees a sample and numbered
comparison stimuli and answers with a number. The contingency itself lives in
:class:`contingency.schedules.MatchingToSample`, so probe trials (never
reinforced), intermittent consequences and block balancing all come from the
engine. :class:`MatchingToSampleBox` only turns replies into response events
on a virtual clock and turns outcomes into text.

A reply that does not name exactly one valid number is not retried: it is
counted in :attr:`MatchingToSampleSubject.invalid_count` and the trial ends the
way a choice timeout does, which the engine scores as incorrect.

``contingency`` is an optional dependency and is imported when a block runs.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from contingency.schedules import MatchingToSample


def extract_choice(raw: str, n_comparisons: int) -> int | None:
    """Return the 1-based number in ``raw``, or ``None`` unless exactly one valid number occurs."""
    numbers = re.findall(r"\d+", raw)
    if len(numbers) != 1:
        return None
    number = int(numbers[0])
    return number if 1 <= number <= n_comparisons else None


@dataclass(frozen=True)
class TrialOutcome:
    """What the feedback text of one trial may depend on."""

    reinforced: bool
    probe: bool
    points: int


@dataclass(frozen=True)
class TrialRecord:
    """One completed trial. ``choice`` is the chosen stimulus, ``None`` after an invalid reply."""

    trial_index: int
    sample: str
    comparisons: tuple[str, ...]
    choice: str | None
    correct: str
    is_correct: bool
    probe: bool
    reinforced: bool
    raw_reply: str
    points: int


def points_feedback(outcome: TrialOutcome) -> str | None:
    """Report a point after reinforcement and "No point" after everything else, probes included."""
    if outcome.reinforced:
        return f"Result: Correct (+1 point). Points: {outcome.points}"
    return f"Result: No point. Points: {outcome.points}"


def silent_probe_feedback(outcome: TrialOutcome) -> str | None:
    """Like :func:`points_feedback`, but say nothing at all after a probe trial."""
    return None if outcome.probe else points_feedback(outcome)


class MatchingToSampleSubject:
    """Subject answering through a resident conversation (``client.send``).

    Result lines and instructions wait until the next trial and go out in the
    same turn, so each trial is exactly one ``send``. Points and the invalid
    reply count persist across boxes, which lets one subject move through the
    stages of an experiment.
    """

    def __init__(self, client: Any) -> None:
        self.client = client
        self.points = 0
        self.invalid_count = 0
        self._pending: list[str] = []

    def instruct(self, line: str) -> None:
        """Queue a line for the top of the next trial's text."""
        self._pending.append(line)

    def choose(self, sample: str, comparisons: Sequence[str]) -> tuple[int | None, str]:
        """Present one trial and return the chosen 1-based number (or ``None``) and the raw reply."""
        options = "  ".join(f"{i}) {name}" for i, name in enumerate(comparisons, start=1))
        numbers = [str(i) for i in range(1, len(comparisons) + 1)]
        prompt = f"Choose {', '.join(numbers[:-1])} or {numbers[-1]}."
        lines = [*self._pending, f"Sample: {sample}", f"Comparisons: {options}", prompt]
        self._pending = []
        raw = self.client.send("\n".join(lines))
        choice = extract_choice(raw, len(comparisons))
        if choice is None:
            self.invalid_count += 1
        return choice, raw


class MatchingToSampleBox:
    """Runs blocks of a :class:`~contingency.schedules.MatchingToSample` schedule, one step a trial.

    Parameters
    ----------
    subject : MatchingToSampleSubject
        Keeps its points between boxes.
    schedule : MatchingToSample
        Must have ``trial_types`` and ``sample_duration=0``.
    block_length : int
        Trials in one :meth:`run_block`; the engine's block is its baseline
        and probe trial types together.
    feedback : callable
        Maps a :class:`TrialOutcome` to the result line, or ``None`` for no line.
    choice_timeout : float
        The schedule's ``choice_timeout``. A mismatch fails loudly on the
        first invalid reply.
    """

    def __init__(
        self,
        subject: MatchingToSampleSubject,
        schedule: MatchingToSample,
        *,
        block_length: int,
        feedback: Callable[[TrialOutcome], str | None] = points_feedback,
        choice_timeout: float = 1.0,
    ) -> None:
        if block_length < 1:
            raise ValueError("block_length must be >= 1")
        if choice_timeout <= 0:
            raise ValueError("choice_timeout must be > 0")
        self.subject = subject
        self.schedule = schedule
        self.block_length = block_length
        self.feedback = feedback
        self.choice_timeout = choice_timeout
        self._now = 0.0

    def run_block(self) -> list[TrialRecord]:
        return [self._run_trial() for _ in range(self.block_length)]

    def _run_trial(self) -> TrialRecord:
        from contingency import ResponseEvent

        schedule = self.schedule
        schedule.step(self._now)  # with iti=0 this tick starts the next trial
        if schedule.current_trial_type is None:
            raise ValueError("the schedule must be built with trial_types")
        if schedule.phase != "choice":
            raise ValueError("the schedule must have sample_duration=0 and iti=0")
        sample = schedule.current_sample
        comparisons = tuple(schedule.current_comparisons)

        number, raw = self.subject.choose(sample, comparisons)
        if number is None:
            self._now += self.choice_timeout + 1e-6
            outcome = schedule.step(self._now)
        else:
            self._now += self.choice_timeout / 2
            operandum = f"choice_{number - 1}"
            outcome = schedule.step(self._now, ResponseEvent(time=self._now, operandum=operandum))
        meta = outcome.meta
        if meta["trial_outcome"] is None:
            raise ValueError("the trial did not end: choice_timeout differs from the schedule's")

        if outcome.reinforced:
            self.subject.points += 1
        points = self.subject.points
        probe = bool(meta["probe"])
        line = self.feedback(TrialOutcome(outcome.reinforced, probe, points))
        if line is not None:
            self.subject.instruct(line)
        return TrialRecord(
            trial_index=int(meta["trial_index"]),
            sample=sample,
            comparisons=comparisons,
            choice=None if number is None else comparisons[number - 1],
            correct=str(meta["correct"]),
            is_correct=meta["trial_outcome"] == "correct",
            probe=probe,
            reinforced=outcome.reinforced,
            raw_reply=raw,
            points=points,
        )


__all__ = [
    "MatchingToSampleBox",
    "MatchingToSampleSubject",
    "TrialOutcome",
    "TrialRecord",
    "extract_choice",
    "points_feedback",
    "silent_probe_feedback",
]
