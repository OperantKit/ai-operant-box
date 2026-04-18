"""Top-level coroutine that drives the asyncio operant session loop."""

from __future__ import annotations

import asyncio
from datetime import datetime

from ai_operant_box.async_simulator.core import (
    LLMSubject,
    OperantEnvironment,
    ScheduleManager,
)
from ai_operant_box.async_simulator.logger import DataLogger
from ai_operant_box.async_simulator.models import BehavioralEvent, Session


async def run_operant_session(
    environment: OperantEnvironment,
    subject: LLMSubject,
    schedule_manager: ScheduleManager,
    data_logger: DataLogger,
    session: Session,
    duration_seconds: float,
) -> None:
    start_time = datetime.now()
    session.start_time = start_time
    data_logger.log_event(
        BehavioralEvent(
            session_id=session.session_id,
            subject_id=session.subject_id,
            event_type="SESSION_START",
            details={"duration": duration_seconds},
        )
    )
    data_logger.log_session(session)

    current_time = start_time
    while (
        current_time - start_time
    ).total_seconds() < duration_seconds and session.status == "running":
        stimulus_to_present = schedule_manager.check_sd_presentation(current_time)
        if stimulus_to_present:
            await environment.present_stimulus(stimulus_to_present)
            data_logger.log_event(
                BehavioralEvent(
                    session_id=session.session_id,
                    subject_id=session.subject_id,
                    event_type="SD_ON",
                    sd_id=stimulus_to_present.stimulus_id,
                    details={"content": stimulus_to_present.content},
                )
            )
            subject.receive_stimulus(stimulus_to_present)

        response_generated = await subject.generate_response()
        if response_generated:
            data_logger.log_event(
                BehavioralEvent(
                    session_id=session.session_id,
                    subject_id=session.subject_id,
                    event_type="RESPONSE",
                    response_id=response_generated.response_id,
                    details=response_generated.conditions,
                )
            )

            consequence_to_deliver = schedule_manager.check_consequence_delivery(
                response_generated, datetime.now()
            )
            if consequence_to_deliver:
                await environment.deliver_consequence(consequence_to_deliver)
                data_logger.log_event(
                    BehavioralEvent(
                        session_id=session.session_id,
                        subject_id=session.subject_id,
                        event_type=f"{consequence_to_deliver.consequence_type.upper()}_ON",
                        reinforcer_id=consequence_to_deliver.consequence_id,
                        magnitude=consequence_to_deliver.magnitude,
                        details={"delay": consequence_to_deliver.delay},
                    )
                )
                subject.receive_consequence(consequence_to_deliver)

        current_time = datetime.now()
        await asyncio.sleep(0.1)

    session.end_time = datetime.now()
    session.status = "completed"
    data_logger.log_event(
        BehavioralEvent(
            session_id=session.session_id,
            subject_id=session.subject_id,
            event_type="SESSION_END",
        )
    )
    data_logger.log_session(session)


__all__ = ["run_operant_session"]
