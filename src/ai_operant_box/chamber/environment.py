"""Single-subject operant chamber + virtual monotonic clock.

The clock is a small in-package shim. The v0 implementation deferred to
``experiment_io.hw.backends.virtual.ManualClock``; that dependency is dropped
here. When ``experiment-io`` stabilises, :class:`VirtualClock` can be replaced
with the canonical HAL clock.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class VirtualClock:
    """Monotonic counter advanced explicitly by :meth:`advance` / :meth:`tick`.

    Mirrors the ``ManualClock`` protocol used by ``experiment-io``: callable
    objects return the current time, ``advance(delta)`` moves the time
    forward, and ``current_time`` exposes the same value as a property.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)

    @property
    def now(self) -> float:
        return self._now

    @property
    def current_time(self) -> float:
        return self._now

    def advance(self, delta: float) -> None:
        self._now += float(delta)

    def tick(self, delta: float) -> None:
        """Alias for :meth:`advance` retained for backward compatibility."""
        self.advance(delta)

    def __call__(self) -> float:
        return self._now


@dataclass
class ResponseRecord:
    timestamp: float
    sd: Any
    response: str | None
    is_reinforced: bool
    is_punished: bool = False
    punishment: float = 0.0


def _read_outcome(outcome: Any) -> tuple[bool, float]:
    """Split a contingency-py ``Outcome`` into (reinforced, punishment magnitude).

    ``Overlay`` marks which component fired in ``meta["base_fired"]`` /
    ``meta["punisher_fired"]``; on a compound step the base reinforcer is
    taken to have contingency's default magnitude 1, so the punishment is
    ``1 - net_magnitude``. Otherwise a shock (``meta["aversive_event"]`` of
    ``"shock"`` / ``"shock_onset"``) or a negative magnitude is punishment
    and any other delivery is reinforcement.
    """
    if not outcome.reinforced or outcome.reinforcer is None:
        return False, 0.0
    meta = outcome.meta or {}
    magnitude = float(outcome.reinforcer.magnitude)
    if "punisher_fired" in meta:
        reinforced = bool(meta.get("base_fired"))
        if not meta["punisher_fired"]:
            return reinforced, 0.0
        if reinforced:
            return True, 1.0 - float(meta.get("net_magnitude", magnitude))
        return False, -magnitude
    if meta.get("aversive_event") in ("shock", "shock_onset") or magnitude < 0:
        return False, abs(magnitude)
    return True, 0.0


class OperantBox:
    """Discrete-time chamber implementing one SD-R-SR step per :meth:`step`.

    Parameters
    ----------
    subject : Subject
        The virtual organism interacting with the chamber.
    schedule : ReinforcementSchedule or contingency-py schedule
        Determines whether each response is reinforced. An object with
        ``is_reinforced(response, t)`` is asked once per response. Otherwise
        an object with ``step(now, event)`` (a contingency-py ``Schedule``) is
        stepped on every step, with a ``ResponseEvent`` when the subject emits
        ``target_response`` and ``None`` otherwise, so time-based events such
        as Sidman shocks arrive without a response. Its outcome is read as
        reinforcement and/or punishment; punishment calls
        ``subject.punish(amount)`` after ``subject.learn`` when the subject has
        that method, and is logged in :attr:`punishments` as ``(time, amount)``.
    clock : optional
        Either a :class:`VirtualClock` (or any object exposing ``advance``
        and ``current_time``), or a zero-arg callable returning the current
        time. Defaults to a fresh :class:`VirtualClock`.
    """

    def __init__(
        self,
        subject: Any,
        schedule: Any,
        clock: Callable[[], float] | None = None,
        target_response: str = "Lever Press",
    ) -> None:
        self.subject = subject
        self.schedule = schedule
        self.clock = clock or VirtualClock()
        self.target_response = target_response
        self.records: list[ResponseRecord] = []
        self.punishments: list[tuple[float, float]] = []
        self.current_sd: Any = "Light ON"

    def step(self, delta: float = 1.0) -> ResponseRecord | None:
        """Advance time, present SD, capture R, and deliver SR."""
        advance = getattr(self.clock, "advance", None)
        if callable(advance):
            advance(delta)
        if callable(self.clock):
            current_time = self.clock()
        else:
            current_time = getattr(self.clock, "current_time", 0.0)

        # 1. SD
        self.subject.sense(self.current_sd)

        # 2. R
        response = self.subject.act()

        if not hasattr(self.schedule, "is_reinforced") and hasattr(self.schedule, "step"):
            return self._step_contingency(response, current_time)

        if response and response != "Nothing":
            # 3. SR
            is_reinforced = self.schedule.is_reinforced(response, current_time)
            sr = "Food Pellet" if is_reinforced else None

            # 4. Learn
            self.subject.learn(sr)

            record = ResponseRecord(
                timestamp=current_time,
                sd=self.current_sd,
                response=response,
                is_reinforced=is_reinforced,
            )
            self.records.append(record)
            return record

        # No response: still notify the subject so it can update its history.
        self.subject.learn(None)
        return None

    def _step_contingency(self, response: str | None, now: float) -> ResponseRecord | None:
        from contingency import ResponseEvent

        emitted = bool(response) and response != "Nothing"
        event = ResponseEvent(time=now) if response == self.target_response else None
        is_reinforced, punishment = _read_outcome(self.schedule.step(now, event))
        sr = "Food Pellet" if is_reinforced else None
        self.subject.learn(sr)
        is_punished = punishment > 0
        if is_punished:
            self.punishments.append((now, punishment))
            punish = getattr(self.subject, "punish", None)
            if callable(punish):
                punish(punishment)
        if not emitted:
            return None
        record = ResponseRecord(
            timestamp=now,
            sd=self.current_sd,
            response=response,
            is_reinforced=is_reinforced,
            is_punished=is_punished,
            punishment=punishment,
        )
        self.records.append(record)
        return record

    def run(
        self,
        duration: float,
        delta: float = 1.0,
        visualizer: Any | None = None,
    ) -> None:
        steps = int(duration / delta)
        for _ in range(steps):
            record = self.step(delta)
            if visualizer and record:
                visualizer.update(record)


__all__ = ["OperantBox", "ResponseRecord", "VirtualClock"]
