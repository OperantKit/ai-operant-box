"""Abstract base class for virtual subjects (rat / pigeon / LLM)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Subject(ABC):
    """Abstract base class for a virtual subject.

    Implementations follow the sense-act-learn cycle of the three-term
    contingency: ``sense(SD) → act() -> R → learn(SR)``.
    """

    @abstractmethod
    def sense(self, discriminative_stimulus: Any) -> None:
        """Receive the discriminative stimulus (SD) from the environment."""

    @abstractmethod
    def act(self) -> str | None:
        """Decide and emit a response (R). Return ``None`` for no response."""

    @abstractmethod
    def learn(self, reinforcer: Any | None) -> None:
        """Receive the consequence (SR) and update internal state."""

    @property
    @abstractmethod
    def history(self) -> list[dict]:
        """The behavioral history of this subject."""
