"""Tests for async_simulator.models DTO defaults."""

from __future__ import annotations

from datetime import datetime

from ai_operant_box.async_simulator.models import (
    BehavioralEvent,
    Consequence,
    Response,
    Schedule,
    Session,
    Stimulus,
)


def test_behavioral_event_creation():
    event = BehavioralEvent(session_id="s1", subject_id="sub1", event_type="SD_ON")
    assert event.event_id is not None
    assert isinstance(event.timestamp, datetime)
    assert event.session_id == "s1"
    assert event.event_type == "SD_ON"


def test_stimulus_creation():
    stimulus = Stimulus(stimulus_type="visual", content="red square")
    assert stimulus.stimulus_id is not None
    assert stimulus.stimulus_type == "visual"
    assert stimulus.content == "red square"


def test_response_creation():
    response = Response(response_type="button_press", conditions={"duration": 0.5})
    assert response.response_id is not None
    assert response.response_type == "button_press"
    assert response.conditions == {"duration": 0.5}


def test_consequence_creation():
    consequence = Consequence(
        consequence_type="reinforcer", delivery_type="food", magnitude=1.0
    )
    assert consequence.consequence_id is not None
    assert consequence.consequence_type == "reinforcer"
    assert consequence.delivery_type == "food"
    assert consequence.magnitude == 1.0


def test_schedule_creation():
    schedule = Schedule(schedule_type="FR", parameters={"ratio": 10})
    assert schedule.schedule_id is not None
    assert schedule.schedule_type == "FR"
    assert schedule.parameters == {"ratio": 10}


def test_session_creation():
    session = Session(subject_id="sub1", program_name="test_program")
    assert session.session_id is not None
    assert isinstance(session.start_time, datetime)
    assert session.subject_id == "sub1"
    assert session.status == "running"
