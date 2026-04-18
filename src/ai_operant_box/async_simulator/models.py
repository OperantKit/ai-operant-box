"""Event-stream DTOs used by the asyncio simulator."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class BehavioralEvent:
    session_id: str
    subject_id: str
    event_type: str  # e.g. "SD_ON", "RESPONSE", "REINFORCER_ON", ...
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    sd_id: str | None = None
    response_id: str | None = None
    reinforcer_id: str | None = None
    punisher_id: str | None = None
    magnitude: int | float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Stimulus:
    stimulus_type: str
    content: str
    stimulus_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schedule_id: str | None = None
    duration: float | None = None


@dataclass
class Response:
    response_type: str
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conditions: dict[str, Any] = field(default_factory=dict)


@dataclass
class Consequence:
    consequence_type: str
    delivery_type: str
    magnitude: int | float
    consequence_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schedule_id: str | None = None
    delay: float = 0.0


@dataclass
class Schedule:
    schedule_type: str
    schedule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    subject_id: str
    program_name: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    schedules_applied: dict[str, str] = field(default_factory=dict)
    status: str = "running"


__all__ = [
    "BehavioralEvent",
    "Consequence",
    "Response",
    "Schedule",
    "Session",
    "Stimulus",
]
