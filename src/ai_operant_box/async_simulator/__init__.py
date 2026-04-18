"""Asyncio-based experiment loop and DTOs (parallel to the SD-R-SR loop).

This subpackage carries the v0 ``src/ai_operant_box`` design (an async
:class:`OperantEnvironment` + :class:`LLMSubject` + :class:`ScheduleManager`
driven by :func:`run_operant_session`). It is preserved for tests and for
callers that want event-stream DTOs (:class:`BehavioralEvent`, etc.).

The discrete-time SD-R-SR API in :mod:`ai_operant_box.chamber` and
:mod:`ai_operant_box.agent` is the recommended entry point for new code.
"""

from ai_operant_box.async_simulator.core import (
    LLMSubject as AsyncLLMSubject,
)
from ai_operant_box.async_simulator.core import OperantEnvironment, ScheduleManager
from ai_operant_box.async_simulator.logger import DataLogger
from ai_operant_box.async_simulator.models import (
    BehavioralEvent,
    Consequence,
    Response,
    Schedule,
    Session,
    Stimulus,
)
from ai_operant_box.async_simulator.simulator import run_operant_session

__all__ = [
    "AsyncLLMSubject",
    "BehavioralEvent",
    "Consequence",
    "DataLogger",
    "OperantEnvironment",
    "Response",
    "Schedule",
    "ScheduleManager",
    "Session",
    "Stimulus",
    "run_operant_session",
]
