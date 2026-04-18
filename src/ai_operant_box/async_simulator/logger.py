"""In-memory event logger for the asyncio simulator."""

from __future__ import annotations

from ai_operant_box.async_simulator.models import BehavioralEvent, Session


class DataLogger:
    """Append-only event logger backed by an in-process list."""

    def __init__(self) -> None:
        self._events: list[BehavioralEvent] = []
        self._sessions: list[Session] = []

    def log_event(self, event: BehavioralEvent) -> None:
        self._events.append(event)

    def get_session_events(self, session_id: str) -> list[BehavioralEvent]:
        return [event for event in self._events if event.session_id == session_id]

    def log_session(self, session: Session) -> None:
        self._sessions.append(session)

    def get_all_events(self) -> list[BehavioralEvent]:
        return self._events


__all__ = ["DataLogger"]
