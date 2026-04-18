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


class OperantBox:
    """Discrete-time chamber implementing one SD-R-SR step per :meth:`step`.

    Parameters
    ----------
    subject : Subject
        The virtual organism interacting with the chamber.
    schedule : ReinforcementSchedule
        Determines whether each response is reinforced.
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
    ) -> None:
        self.subject = subject
        self.schedule = schedule
        self.clock = clock or VirtualClock()
        self.records: list[ResponseRecord] = []
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
