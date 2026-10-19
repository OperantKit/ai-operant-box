"""Two-lever chamber driven by a contingency-py concurrent schedule.

:class:`ConcurrentBox` runs the same SD-R-SR step as :class:`OperantBox`,
but the subject chooses between operanda. The subject interface is the
ordinary :class:`~ai_operant_box.agent.base.Subject`: ``act()`` returns
an action label, and ``operanda`` maps the labels that count as lever
presses to the operandum keys of the schedule (by default
``"Left Lever" -> "left"`` and ``"Right Lever" -> "right"``). Any other
label, ``None`` or ``"Nothing"`` is not a response; the schedule still
receives a time tick so its clocks stay aligned.

The reinforcement decision, including the changeover delay (COD), is
delegated to :class:`contingency.schedules.concurrent.Concurrent`
(Catania, 1966; Herrnstein, 1961). ``contingency-py`` is imported
lazily and comes with the ``concurrent`` extra.

References
----------
Catania, A. C. (1966). Concurrent performances: Reinforcement
interaction and response independence. *Journal of the Experimental
Analysis of Behavior*, 9(3), 253-263.
https://doi.org/10.1901/jeab.1966.9-253

Herrnstein, R. J. (1961). Relative and absolute strength of response
as a function of frequency of reinforcement. *Journal of the
Experimental Analysis of Behavior*, 4(3), 267-272.
https://doi.org/10.1901/jeab.1961.4-267
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ai_operant_box.chamber.environment import VirtualClock

CONCURRENT_ACTIONS: tuple[str, ...] = ("Left Lever", "Right Lever", "Nothing")

DEFAULT_OPERANDA: dict[str, str] = {"Left Lever": "left", "Right Lever": "right"}


@dataclass
class ChoiceRecord:
    """One response on one operandum."""

    timestamp: float
    operandum: str
    is_reinforced: bool
    cod_suppressed: bool


def concurrent_vi_vi(
    mean_left: float,
    mean_right: float,
    cod: float = 2.0,
    seed: int | None = None,
    n_intervals: int = 12,
) -> Any:
    """Build ``Conc VI mean_left VI mean_right`` with a COD (contingency-py).

    The two VI components get distinct seeds derived from ``seed`` so a
    run is reproducible. Operandum keys are ``"left"`` and ``"right"``.
    """
    from contingency.schedules.concurrent import Concurrent
    from contingency.schedules.interval import VI

    seed_left = None if seed is None else 2 * seed
    seed_right = None if seed is None else 2 * seed + 1
    return Concurrent(
        {
            "left": VI(mean_left, n_intervals=n_intervals, seed=seed_left),
            "right": VI(mean_right, n_intervals=n_intervals, seed=seed_right),
        },
        cod=cod,
    )


class ConcurrentBox:
    """Discrete-time chamber with two (or more) operanda.

    Parameters
    ----------
    subject : Subject
        Emits action labels from :meth:`act`.
    schedule
        A contingency-py schedule with ``step(now, event)``, normally a
        :class:`~contingency.schedules.concurrent.Concurrent` (see
        :func:`concurrent_vi_vi`).
    clock : optional
        As in :class:`OperantBox`. Defaults to a fresh :class:`VirtualClock`.
    operanda : mapping, optional
        Action label -> operandum key of ``schedule``.
    """

    def __init__(
        self,
        subject: Any,
        schedule: Any,
        clock: Callable[[], float] | None = None,
        operanda: Mapping[str, str] | None = None,
    ) -> None:
        self.subject = subject
        self.schedule = schedule
        self.clock = clock or VirtualClock()
        self.operanda: dict[str, str] = dict(operanda or DEFAULT_OPERANDA)
        known = getattr(schedule, "components", None)
        if known is not None:
            unknown = sorted(set(self.operanda.values()) - set(known))
            if unknown:
                raise ValueError(
                    f"operanda {unknown!r} are not components of the schedule ({sorted(known)!r})"
                )
        self.records: list[ChoiceRecord] = []
        self.current_sd: Any = "Light ON"

    def _now(self, delta: float) -> float:
        advance = getattr(self.clock, "advance", None)
        if callable(advance):
            advance(delta)
        if callable(self.clock):
            return float(self.clock())
        return float(getattr(self.clock, "current_time", 0.0))

    def step(self, delta: float = 1.0) -> ChoiceRecord | None:
        """Advance time, present SD, capture R, and deliver SR."""
        from contingency.entities import ResponseEvent

        now = self._now(delta)
        self.subject.sense(self.current_sd)
        response = self.subject.act()
        operandum = self.operanda.get(response) if response else None

        if operandum is None:
            self.schedule.step(now, None)
            self.subject.learn(None)
            return None

        outcome = self.schedule.step(now, ResponseEvent(time=now, operandum=operandum))
        is_reinforced = bool(outcome.reinforced)
        self.subject.learn("Food Pellet" if is_reinforced else None)
        record = ChoiceRecord(
            timestamp=now,
            operandum=operandum,
            is_reinforced=is_reinforced,
            cod_suppressed=bool(outcome.meta.get("cod_suppressed", False)),
        )
        self.records.append(record)
        return record

    def run(self, duration: float, delta: float = 1.0) -> None:
        for _ in range(int(duration / delta)):
            self.step(delta)


__all__ = [
    "CONCURRENT_ACTIONS",
    "DEFAULT_OPERANDA",
    "ChoiceRecord",
    "ConcurrentBox",
    "concurrent_vi_vi",
]
