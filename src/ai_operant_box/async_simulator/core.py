"""Asyncio environment / subject / schedule manager (legacy v0 design).

The classes here form a parallel architecture to the discrete-time SD-R-SR
loop in :mod:`ai_operant_box.chamber`. They are preserved for tests and for
callers who want DTO-style event streams via
:class:`~ai_operant_box.async_simulator.models.BehavioralEvent`.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from ai_operant_box.async_simulator.models import (
    Consequence,
    Response,
    Schedule,
    Stimulus,
)


class OperantEnvironment:
    """Presents stimuli, detects responses, delivers consequences."""

    def __init__(self) -> None:
        self._current_stimulus: Stimulus | None = None
        self._pending_response: Response | None = None

    async def present_stimulus(self, stimulus: Stimulus) -> None:
        self._current_stimulus = stimulus
        await asyncio.sleep(0.01)

    def detect_response(self) -> Response | None:
        return self._pending_response

    async def deliver_consequence(self, consequence: Consequence) -> None:
        await asyncio.sleep(consequence.delay)


class LLMSubject:
    """Wraps an LLM as a subject in the asyncio loop (placeholder logic)."""

    def __init__(self, subject_id: str) -> None:
        self.subject_id = subject_id
        self._current_stimulus: Stimulus | None = None
        self._event_history: list[str] = []

    def receive_stimulus(self, stimulus: Stimulus) -> None:
        self._current_stimulus = stimulus
        self._event_history.append(f"Received Stimulus: {stimulus.content}")

    async def generate_response(self) -> Response | None:
        if self._current_stimulus is None:
            return None
        llm_output = (
            f"Responding to '{self._current_stimulus.content}' with a simulated action."
        )
        response_type = "text_output"
        if "button" in self._current_stimulus.content.lower():
            response_type = "button_click"

        await asyncio.sleep(0.5)

        dummy_response = Response(
            response_id=f"resp-{datetime.now().timestamp()}",
            response_type=response_type,
            conditions={"llm_output": llm_output},
        )
        self._event_history.append(
            f"Generated Response: {dummy_response.response_type} ({llm_output})"
        )
        self._current_stimulus = None
        return dummy_response

    def receive_consequence(self, consequence: Consequence) -> None:
        self._event_history.append(
            f"Received Consequence: {consequence.consequence_type} ({consequence.magnitude})"
        )


class ScheduleManager:
    """Decides SD presentation and consequence delivery for a fixed FR1."""

    def __init__(self, schedules: list[Schedule]) -> None:
        self._schedules = {s.schedule_id: s for s in schedules}
        # Seed last-presentation time with construction-time so the very first
        # ``check_sd_presentation`` call returns ``None`` until the configured
        # delay has elapsed; matches the v0 test expectations.
        now = datetime.now()
        self._last_sd_presentation_time: dict[str, datetime] = {
            sid: now for sid in self._schedules
        }
        self._response_counts: dict[str, int] = {}

    def check_sd_presentation(self, current_time: datetime) -> Stimulus | None:
        fr1_schedule_id = "FR1_schedule"
        if fr1_schedule_id in self._schedules:
            last = self._last_sd_presentation_time.get(fr1_schedule_id, current_time)
            if (current_time - last).total_seconds() > 5:
                self._last_sd_presentation_time[fr1_schedule_id] = current_time
                return Stimulus(
                    stimulus_id="sd-light",
                    stimulus_type="visual",
                    content="Please respond to the light.",
                )
        return None

    def check_consequence_delivery(
        self, response: Response, current_time: datetime
    ) -> Consequence | None:
        fr1_schedule_id = "FR1_schedule"
        if fr1_schedule_id in self._schedules and response:
            self._response_counts[fr1_schedule_id] = (
                self._response_counts.get(fr1_schedule_id, 0) + 1
            )
            if self._response_counts[fr1_schedule_id] >= self._schedules[
                fr1_schedule_id
            ].parameters.get("ratio", 1):
                self._response_counts[fr1_schedule_id] = 0
                return Consequence(
                    consequence_id=f"rein-{current_time.timestamp()}",
                    consequence_type="reinforcer",
                    delivery_type="text_feedback",
                    magnitude=1.0,
                    schedule_id=fr1_schedule_id,
                )
        return None


__all__ = ["LLMSubject", "OperantEnvironment", "ScheduleManager"]
