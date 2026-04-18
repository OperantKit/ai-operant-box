"""Dyadic interdependent reinforcement schedule + leader-follower chamber.

Models social behaviour as a *response unit* in which a follower's response
within a defined time-window after a leader's response produces reinforcement
for both subjects. Formally equivalent to a dyadic DRL applied at the pair
level.

Reference
---------
Benvenuti, M. F., et al. (2024). The role of schedules of reinforcement in the
experimental analysis of social behavior. *Perspectives on Behavior Science*.
https://doi.org/10.1007/s40614-024-00421-5
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_operant_box.agent.base import Subject


@dataclass
class SocialEvent:
    """Per-step event from :class:`LeaderFollowerBox`."""
    time_step: int
    leader_responded: bool
    follower_responded: bool
    leader_follower_irt: float | None
    reinforced: bool
    leader_reinforcer: Any | None
    follower_reinforcer: Any | None


class InterdependentSchedule:
    """Leader-follower interdependent reinforcement schedule.

    Reinforcement is delivered when a follower response occurs within
    ``window`` seconds after the most recent leader response, and the
    leader-follower IRT is at least ``min_irt`` seconds.

    Parameters
    ----------
    window : float
        Maximum leader-follower IRT in seconds. Default ``5.0``.
    min_irt : float
        Minimum leader-follower IRT in seconds. Default ``0.0``.
    reinforcer : Any
        The reinforcer delivered to both subjects on a successful pair.
    """

    def __init__(
        self,
        window: float = 5.0,
        min_irt: float = 0.0,
        reinforcer: Any = "food",
    ) -> None:
        if window <= 0:
            raise ValueError(f"window must be > 0, got {window}")
        if min_irt < 0:
            raise ValueError(f"min_irt must be >= 0, got {min_irt}")
        self.window = window
        self.min_irt = min_irt
        self.reinforcer = reinforcer

        self._last_leader_time: float | None = None

    def note_leader_response(self, current_time: float) -> None:
        self._last_leader_time = current_time

    def evaluate_follower_response(
        self, current_time: float
    ) -> tuple[bool, float | None]:
        if self._last_leader_time is None:
            return False, None
        irt = current_time - self._last_leader_time
        if self.min_irt <= irt <= self.window:
            self._last_leader_time = None
            return True, irt
        return False, irt

    def reset(self) -> None:
        self._last_leader_time = None

    @property
    def pending_leader(self) -> bool:
        return self._last_leader_time is not None


class LeaderFollowerBox:
    """Discrete-time dyadic chamber for interdependent reinforcement.

    Each step: leader senses + acts → follower senses + acts → schedule
    evaluates → both subjects learn. The leader's response opens the window;
    the follower's response within the window yields joint reinforcement.

    Parameters
    ----------
    leader, follower : Subject
        The two subjects, with ``leader`` always acting first per step.
    schedule : InterdependentSchedule
        Joint reinforcement criterion.
    step_duration : float
        Seconds per simulated discrete step. Default ``1.0``.
    leader_sd, follower_sd : Any
        Per-step discriminative stimuli for each subject.
    """

    def __init__(
        self,
        leader: Subject,
        follower: Subject,
        schedule: InterdependentSchedule,
        step_duration: float = 1.0,
        leader_sd: Any = None,
        follower_sd: Any = None,
    ) -> None:
        self.leader = leader
        self.follower = follower
        self.schedule = schedule
        self.step_duration = step_duration
        self.leader_sd = leader_sd
        self.follower_sd = follower_sd

        self._current_time: float = 0.0
        self._log: list[SocialEvent] = []

    def run(self, n_steps: int) -> list[SocialEvent]:
        for step in range(n_steps):
            self.leader.sense(self.leader_sd)
            self.follower.sense(self.follower_sd)

            leader_response = self.leader.act()
            leader_responded = leader_response is not None
            if leader_responded:
                self.schedule.note_leader_response(self._current_time)

            follower_response = self.follower.act()
            follower_responded = follower_response is not None

            reinforced = False
            irt: float | None = None
            leader_sr: Any | None = None
            follower_sr: Any | None = None

            if follower_responded:
                reinforced, irt = self.schedule.evaluate_follower_response(
                    self._current_time
                )
                if reinforced:
                    leader_sr = self.schedule.reinforcer
                    follower_sr = self.schedule.reinforcer

            self.leader.learn(leader_sr)
            self.follower.learn(follower_sr)

            self._log.append(
                SocialEvent(
                    time_step=step,
                    leader_responded=leader_responded,
                    follower_responded=follower_responded,
                    leader_follower_irt=irt,
                    reinforced=reinforced,
                    leader_reinforcer=leader_sr,
                    follower_reinforcer=follower_sr,
                )
            )

            self._current_time += self.step_duration

        return self._log

    def reset(self) -> None:
        self._current_time = 0.0
        self._log = []
        self.schedule.reset()

    @property
    def log(self) -> list[SocialEvent]:
        return self._log

    def dyadic_response_rate(self) -> tuple[float, float]:
        if not self._log:
            return 0.0, 0.0
        n = len(self._log)
        leader_rate = sum(1 for e in self._log if e.leader_responded) / n
        follower_rate = sum(1 for e in self._log if e.follower_responded) / n
        return leader_rate, follower_rate

    def reinforcement_rate(self) -> float:
        if not self._log:
            return 0.0
        return sum(1 for e in self._log if e.reinforced) / len(self._log)

    def mean_irt(self) -> float | None:
        reinforced_irts = [
            e.leader_follower_irt
            for e in self._log
            if e.reinforced and e.leader_follower_irt is not None
        ]
        if not reinforced_irts:
            return None
        return sum(reinforced_irts) / len(reinforced_irts)

    def synchrony_ratio(self) -> float:
        if not self._log:
            return 0.0
        both = sum(1 for e in self._log if e.leader_responded and e.follower_responded)
        return both / len(self._log)


__all__ = ["InterdependentSchedule", "LeaderFollowerBox", "SocialEvent"]
