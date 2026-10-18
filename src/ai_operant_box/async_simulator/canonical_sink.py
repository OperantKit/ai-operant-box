"""Convert :class:`BehavioralEvent` stream to canonical :mod:`experiment_core`
events and emit to an :class:`experiment_core.EventSink`.

The asyncio simulator records its native :class:`BehavioralEvent` stream
into a :class:`DataLogger` (in-memory). For downstream consumption by
the operantkit pipeline (``session-recorder`` OKL v1 logs,
``session-analyzer``, ``session-visualizer`` live tail, etc.) this module
provides a post-hoc converter that walks the recorded events and emits
the subset that maps cleanly to
:data:`experiment_core.events.SessionEvent`.

Mapping (current minimum viable):

============================  =======================================
``BehavioralEvent.event_type``  Canonical
============================  =======================================
``RESPONSE``                    :class:`experiment_core.ResponseEvent`
``REINFORCER_ON``               :class:`experiment_core.ReinforcerStartEvent`
                                (``potency = magnitude or 1.0``)
``REINFORCER_OFF``              :class:`experiment_core.ReinforcerEndEvent`
*everything else*               (skipped — has no canonical equivalent)
============================  =======================================

``SESSION_START`` / ``SESSION_END`` / ``SD_ON`` / ``PUNISHER_ON`` and
domain-specific events have no canonical SessionEvent counterpart yet
and are intentionally dropped at this layer (they remain in the
DataLogger).
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from ai_operant_box.async_simulator.models import BehavioralEvent


def mirror_events_to_sink(
    events: Iterable[BehavioralEvent],
    sink: object,
    *,
    session_start: datetime,
) -> int:
    """Convert ``events`` and emit canonical SessionEvents to ``sink``.

    ``sink`` is duck-typed against :class:`experiment_core.EventSink`
    (must expose ``emit(event)``). Object-typed here so this module can
    be imported even when ``experiment-core`` is not installed; the
    actual experiment-core types are imported lazily.

    ``session_start`` is the wall-clock origin used to convert
    :attr:`BehavioralEvent.timestamp` (``datetime``) to canonical
    seconds (``float``).

    Returns the count of canonical events emitted (i.e. the number of
    BehavioralEvents that mapped to a SessionEvent).
    """
    # Lazy import keeps ai-operant-box installable without
    # experiment-core when the canonical adapter is unused.
    from experiment_core import (
        ReinforcerEndEvent,
        ReinforcerStartEvent,
        ResponseEvent,
    )

    response_id = 1
    reinforcer_id = 1
    emitted = 0
    pending_end_ids: list[int] = []

    for event in events:
        timestamp = (event.timestamp - session_start).total_seconds()

        if event.event_type == "RESPONSE":
            sink.emit(ResponseEvent(id=response_id, timestamp=timestamp))
            response_id += 1
            emitted += 1
            continue

        if event.event_type == "REINFORCER_ON":
            potency = float(event.magnitude) if event.magnitude is not None else 1.0
            sink.emit(
                ReinforcerStartEvent(
                    id=reinforcer_id,
                    timestamp=timestamp,
                    potency=potency,
                )
            )
            pending_end_ids.append(reinforcer_id)
            reinforcer_id += 1
            emitted += 1
            continue

        if event.event_type == "REINFORCER_OFF":
            # Pair with the most recent unmatched start, FIFO.
            paired_id = pending_end_ids.pop(0) if pending_end_ids else (reinforcer_id - 1)
            sink.emit(ReinforcerEndEvent(id=paired_id, timestamp=timestamp))
            emitted += 1
            continue

    return emitted


__all__ = ["mirror_events_to_sink"]
