"""Tests for async_simulator.logger.DataLogger."""

from __future__ import annotations

import pytest

from ai_operant_box.async_simulator.logger import DataLogger
from ai_operant_box.async_simulator.models import BehavioralEvent, Session


@pytest.fixture
def data_logger():
    return DataLogger()


def test_log_event(data_logger):
    event = BehavioralEvent(session_id="s1", subject_id="sub1", event_type="TEST_EVENT")
    data_logger.log_event(event)
    assert len(data_logger._events) == 1
    assert data_logger._events[0] == event


def test_get_session_events(data_logger):
    event1 = BehavioralEvent(session_id="s_test", subject_id="subA", event_type="E1")
    event2 = BehavioralEvent(session_id="s_test", subject_id="subA", event_type="E2")
    event3 = BehavioralEvent(session_id="s_other", subject_id="subB", event_type="E3")
    data_logger.log_event(event1)
    data_logger.log_event(event2)
    data_logger.log_event(event3)

    session_events = data_logger.get_session_events("s_test")
    assert len(session_events) == 2
    assert event1 in session_events
    assert event2 in session_events
    assert event3 not in session_events


def test_log_session(data_logger):
    session = Session(session_id="session_log_test", subject_id="subX", program_name="progY")
    data_logger.log_session(session)
    assert len(data_logger._sessions) == 1
    assert data_logger._sessions[0] == session


def test_get_all_events(data_logger):
    event1 = BehavioralEvent(session_id="s_all1", subject_id="subA", event_type="E1")
    event2 = BehavioralEvent(session_id="s_all2", subject_id="subB", event_type="E2")
    data_logger.log_event(event1)
    data_logger.log_event(event2)

    all_events = data_logger.get_all_events()
    assert len(all_events) == 2
    assert event1 in all_events
    assert event2 in all_events
