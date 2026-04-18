"""End-to-end test of run_operant_session (asyncio loop, legacy v0 architecture)."""

from __future__ import annotations

from itertools import chain, repeat
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_operant_box.async_simulator.core import (
    LLMSubject,
    OperantEnvironment,
    ScheduleManager,
)
from ai_operant_box.async_simulator.logger import DataLogger
from ai_operant_box.async_simulator.models import (
    Consequence,
    Response,
    Schedule,
    Session,
    Stimulus,
)
from ai_operant_box.async_simulator.simulator import run_operant_session


@pytest.fixture
def mock_environment():
    env = OperantEnvironment()
    env.present_stimulus = AsyncMock()
    env.deliver_consequence = AsyncMock()
    return env


@pytest.fixture
def mock_llm_subject():
    sub = LLMSubject(subject_id="test_sub")
    sub.receive_stimulus = MagicMock()
    sub.generate_response = AsyncMock(return_value=Response(response_type="test_resp"))
    sub.receive_consequence = MagicMock()
    return sub


@pytest.fixture
def mock_schedule_manager():
    sm = ScheduleManager(
        schedules=[
            Schedule(
                schedule_id="FR1_schedule",
                schedule_type="FR",
                parameters={"ratio": 1},
            )
        ]
    )
    # Return SD on the first call, then None forever after.
    sd_iter = chain(
        [Stimulus(stimulus_type="visual", content="sd_test")],
        repeat(None),
    )
    sm.check_sd_presentation = MagicMock(side_effect=lambda *a, **kw: next(sd_iter))
    sm.check_consequence_delivery = MagicMock(
        return_value=Consequence(
            consequence_type="reinforcer", delivery_type="food", magnitude=1.0
        )
    )
    return sm


@pytest.fixture
def mock_data_logger():
    dl = DataLogger()
    dl.log_event = MagicMock()
    dl.log_session = MagicMock()
    return dl


@pytest.fixture
def test_session():
    return Session(subject_id="test_sub", program_name="test_program")


@pytest.mark.asyncio
async def test_run_operant_session_completes(
    mock_environment,
    mock_llm_subject,
    mock_schedule_manager,
    mock_data_logger,
    test_session,
):
    duration = 1.0

    await run_operant_session(
        mock_environment,
        mock_llm_subject,
        mock_schedule_manager,
        mock_data_logger,
        test_session,
        duration,
    )

    assert test_session.status == "completed"
    assert test_session.end_time is not None

    mock_llm_subject.generate_response.assert_called()
    mock_llm_subject.receive_stimulus.assert_called_once()
    mock_llm_subject.receive_consequence.assert_called()

    mock_environment.present_stimulus.assert_called_once()
    mock_environment.deliver_consequence.assert_called()

    # SESSION_START / SD_ON / RESPONSE / REINFORCER_ON / SESSION_END all logged.
    event_types = {
        call.args[0].event_type
        for call in mock_data_logger.log_event.call_args_list
        if call.args
    }
    for required in {"SESSION_START", "SD_ON", "RESPONSE", "REINFORCER_ON", "SESSION_END"}:
        assert required in event_types
