"""Tests for the asyncio environment / subject / schedule manager (legacy v0)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from ai_operant_box.async_simulator.core import (
    LLMSubject,
    OperantEnvironment,
    ScheduleManager,
)
from ai_operant_box.async_simulator.models import (
    Consequence,
    Response,
    Schedule,
    Stimulus,
)


@pytest.fixture
def operant_environment():
    return OperantEnvironment()


@pytest.fixture
def llm_subject():
    return LLMSubject(subject_id="test_llm")


@pytest.fixture
def fr1_schedule():
    return Schedule(
        schedule_id="FR1_schedule", schedule_type="FR", parameters={"ratio": 1}
    )


@pytest.fixture
def schedule_manager(fr1_schedule):
    return ScheduleManager(schedules=[fr1_schedule])


@pytest.mark.asyncio
async def test_operant_environment_present_stimulus(operant_environment):
    stimulus = Stimulus(stimulus_type="visual", content="test light")
    await operant_environment.present_stimulus(stimulus)
    assert operant_environment._current_stimulus == stimulus


@pytest.mark.asyncio
async def test_operant_environment_deliver_consequence(operant_environment):
    consequence = Consequence(
        consequence_type="reinforcer", delivery_type="food", magnitude=1.0
    )
    await operant_environment.deliver_consequence(consequence)


@pytest.mark.asyncio
async def test_llm_subject_receive_stimulus(llm_subject):
    stimulus = Stimulus(stimulus_type="text", content="Hello LLM")
    llm_subject.receive_stimulus(stimulus)
    assert llm_subject._current_stimulus == stimulus
    assert f"Received Stimulus: {stimulus.content}" in llm_subject._event_history


@pytest.mark.asyncio
async def test_llm_subject_generate_response(llm_subject):
    stimulus = Stimulus(stimulus_type="text", content="Please press button A.")
    llm_subject.receive_stimulus(stimulus)
    response = await llm_subject.generate_response()
    assert response is not None
    assert response.response_type == "button_click"
    assert "Generated Response:" in llm_subject._event_history[1]
    assert llm_subject._current_stimulus is None


@pytest.mark.asyncio
async def test_llm_subject_receive_consequence(llm_subject):
    consequence = Consequence(
        consequence_type="reinforcer", delivery_type="token", magnitude=5.0
    )
    llm_subject.receive_consequence(consequence)
    assert (
        f"Received Consequence: {consequence.consequence_type} ({consequence.magnitude})"
        in llm_subject._event_history
    )


def test_schedule_manager_check_sd_presentation_initial(schedule_manager):
    sd = schedule_manager.check_sd_presentation(datetime.now())
    assert sd is None


def test_schedule_manager_check_sd_presentation_after_delay(
    schedule_manager, fr1_schedule
):
    now = datetime.now()
    schedule_manager._last_sd_presentation_time[fr1_schedule.schedule_id] = (
        now - timedelta(seconds=6)
    )
    sd = schedule_manager.check_sd_presentation(now)
    assert sd is not None
    assert sd.stimulus_id == "sd-light"


def test_schedule_manager_check_consequence_delivery_fr1(
    schedule_manager, fr1_schedule
):
    response = Response(response_type="button_press")
    consequence = schedule_manager.check_consequence_delivery(response, datetime.now())
    assert consequence is not None
    assert consequence.consequence_type == "reinforcer"
    assert schedule_manager._response_counts[fr1_schedule.schedule_id] == 0
