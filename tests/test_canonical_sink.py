"""Tests for :func:`mirror_events_to_sink` — convert
:class:`BehavioralEvent` stream to canonical
:mod:`experiment_core` events.

These tests require ``experiment-core`` and ``session-recorder`` to be
importable. The integration round-trip uses
:class:`session_recorder.OklSink` followed by
:func:`session_recorder.read_log`.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

experiment_core = pytest.importorskip("experiment_core")
session_recorder = pytest.importorskip("session_recorder")

from ai_operant_box.async_simulator.canonical_sink import (  # noqa: E402
    mirror_events_to_sink,
)
from ai_operant_box.async_simulator.models import BehavioralEvent  # noqa: E402


def _be(event_type: str, t_offset_s: float, **kwargs) -> BehavioralEvent:
    """Build a BehavioralEvent at ``session_start + t_offset_s`` seconds."""
    base = datetime(2026, 4, 27, 14, 0, 0)
    return BehavioralEvent(
        session_id="s1",
        subject_id="sub1",
        event_type=event_type,
        timestamp=base + timedelta(seconds=t_offset_s),
        **kwargs,
    )


@pytest.fixture
def session_start() -> datetime:
    return datetime(2026, 4, 27, 14, 0, 0)


@pytest.mark.unit
def test_response_event_is_mapped(session_start) -> None:
    sink = experiment_core.InMemorySink()
    events = [_be("RESPONSE", 0.5)]
    n = mirror_events_to_sink(events, sink, session_start=session_start)
    assert n == 1
    assert len(sink.events) == 1
    assert isinstance(sink.events[0], experiment_core.ResponseEvent)
    assert sink.events[0].id == 1
    assert sink.events[0].timestamp == 0.5


@pytest.mark.unit
def test_response_ids_are_sequential(session_start) -> None:
    sink = experiment_core.InMemorySink()
    events = [_be("RESPONSE", t) for t in (0.1, 0.2, 0.3)]
    mirror_events_to_sink(events, sink, session_start=session_start)
    ids = [e.id for e in sink.events]
    assert ids == [1, 2, 3]


@pytest.mark.unit
def test_reinforcer_on_maps_to_start_with_potency(session_start) -> None:
    sink = experiment_core.InMemorySink()
    events = [_be("REINFORCER_ON", 1.0, magnitude=0.75)]
    mirror_events_to_sink(events, sink, session_start=session_start)
    assert len(sink.events) == 1
    e = sink.events[0]
    assert isinstance(e, experiment_core.ReinforcerStartEvent)
    assert e.id == 1
    assert e.timestamp == 1.0
    assert e.potency == 0.75


@pytest.mark.unit
def test_reinforcer_on_without_magnitude_defaults_to_potency_1(session_start) -> None:
    sink = experiment_core.InMemorySink()
    events = [_be("REINFORCER_ON", 1.0)]
    mirror_events_to_sink(events, sink, session_start=session_start)
    assert sink.events[0].potency == 1.0


@pytest.mark.unit
def test_reinforcer_off_pairs_with_preceding_on(session_start) -> None:
    sink = experiment_core.InMemorySink()
    events = [
        _be("REINFORCER_ON", 1.0),
        _be("REINFORCER_OFF", 1.5),
        _be("REINFORCER_ON", 2.0),
        _be("REINFORCER_OFF", 2.5),
    ]
    mirror_events_to_sink(events, sink, session_start=session_start)
    starts = [e for e in sink.events if isinstance(e, experiment_core.ReinforcerStartEvent)]
    ends = [e for e in sink.events if isinstance(e, experiment_core.ReinforcerEndEvent)]
    assert len(starts) == 2
    assert len(ends) == 2
    # Each REINFORCER_OFF carries the matching id of its REINFORCER_ON.
    assert ends[0].id == starts[0].id
    assert ends[1].id == starts[1].id


@pytest.mark.unit
def test_session_start_end_sd_on_punisher_are_skipped(session_start) -> None:
    sink = experiment_core.InMemorySink()
    events = [
        _be("SESSION_START", 0.0),
        _be("SD_ON", 0.1, sd_id="sd-a"),
        _be("RESPONSE", 0.5),
        _be("PUNISHER_ON", 0.6, magnitude=1.0),
        _be("REINFORCER_ON", 1.0, magnitude=0.5),
        _be("SESSION_END", 2.0),
    ]
    n = mirror_events_to_sink(events, sink, session_start=session_start)
    # Only RESPONSE + REINFORCER_ON should map.
    assert n == 2
    assert len(sink.events) == 2


@pytest.mark.unit
def test_round_trip_through_oslsink(tmp_path, session_start) -> None:
    """End-to-end: BehavioralEvent → mirror → OklSink → read_log → events."""
    path = tmp_path / "session.osl.txt"
    sink = session_recorder.OklSink(output_path=path)
    sink.write_header(
        session_name="ai-operant-box-test",
        clock_type="WALLCLOCK",
        session_start=0.0,
    )
    events = [
        _be("RESPONSE", 0.1),
        _be("RESPONSE", 0.5),
        _be("REINFORCER_ON", 1.0, magnitude=0.9),
    ]
    mirror_events_to_sink(events, sink, session_start=session_start)

    log = session_recorder.read_log(str(path))
    assert len(log.events) == 3
    responses = [e for e in log.events if isinstance(e, experiment_core.ResponseEvent)]
    assert len(responses) == 2
    assert responses[0].timestamp == 0.1
    assert responses[1].timestamp == 0.5
    starts = [
        e for e in log.events if isinstance(e, experiment_core.ReinforcerStartEvent)
    ]
    assert len(starts) == 1
    assert starts[0].potency == 0.9
