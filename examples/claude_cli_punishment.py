"""Run a Claude model through baseline, punishment and recovery on a VI schedule.

One ``claude`` process (one conversation) is driven through three phases of
``--phase-steps`` steps of 1 s each, with the same subject and clock; the
schedule is swapped at each boundary and the subject is not told:

* baseline: VI ``--vi`` s (seed ``s``)
* punishment: the same VI (seed ``s + 1``) overlaid with a punisher after
  every Lever Press (``Overlay(VI, FR 1)``; Azrin, 1960; Azrin & Holz, 1966)
* recovery: VI ``--vi`` s (seed ``s + 2``)

Outcomes are shown in points (as in ``claude_cli_fi.py --response-cost``): a
reinforcer is worth ``--reinforcer-points`` and a punisher costs
``--punishment-points``. The result line before each stimulus is one of::

    Result: Reward (+100 points). Points: N
    Result: Shock (-1 point). Points: N
    Result: Reward (+100 points). Shock (-1 point). Points: N
    Result: No Reward. Points: N

and the system prompt has one sentence on the shock. Lever-press rates per
phase give the suppression ratio (punishment rate / baseline rate) and the
recovery ratio (recovery rate / baseline rate) through
``session_analyzer.analytics.analyze_punishment``.

    .venv/bin/python examples/claude_cli_punishment.py --model haiku --seed 0 --out-dir runs/punish

Needs contingency-py (``Overlay``, ``VI``, ``FR``). The two ratios are filled
in when session-analyzer provides ``analytics.analyze_punishment``; otherwise
they are ``null`` and the phase-local times in the summary are enough to
compute them.

Azrin, N. H. (1960). Effects of punishment intensity during variable-interval
reinforcement. Journal of the Experimental Analysis of Behavior, 3(2), 123-142.
https://doi.org/10.1901/jeab.1960.3-123

Azrin, N. H., & Holz, W. C. (1966). Punishment. In W. K. Honig (Ed.), Operant
behavior: Areas of research and application (pp. 380-447).
Appleton-Century-Crofts.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ai_operant_box.agent.api_client import ClaudeCLIClient
from ai_operant_box.agent.llm_agent import (
    ACTIONS,
    SYSTEM_PROMPT,
    StreamingLLMSubject,
    extract_action,
)
from ai_operant_box.chamber import OperantBox

TARGET = "Lever Press"
PHASES: tuple[str, str, str] = ("baseline", "punishment", "recovery")
SHOCK_SENTENCE = "Sometimes a Lever Press is followed by a shock that costs points."


def _points(n: int) -> str:
    return f"{n} point" if n == 1 else f"{n} points"


def punishment_system_prompt(reinforcer_points: int = 100) -> str:
    """The single-lever system prompt, the reward value and one sentence on the shock."""
    return (f"{SYSTEM_PROMPT} Each reward is worth {_points(reinforcer_points)}. "
            f"{SHOCK_SENTENCE}")


class _Magnitude:
    """Wrap a schedule and replace the magnitude of its reinforcer (FR has none)."""

    def __init__(self, inner: Any, magnitude: float) -> None:
        self._inner = inner
        self._magnitude = float(magnitude)

    def step(self, now: float, event: Any = None) -> Any:
        from dataclasses import replace

        outcome = self._inner.step(now, event)
        if outcome.reinforced and outcome.reinforcer is not None:
            return replace(outcome, reinforcer=replace(outcome.reinforcer,
                                                       magnitude=self._magnitude))
        return outcome

    def reset(self) -> None:
        self._inner.reset()


class _Shifted:
    """Present phase-local time ``now - offset`` to a schedule started mid-session.

    Interval schedules arm at absolute times from 0, so a VI built for the
    second phase would otherwise be armed on its first step.
    """

    def __init__(self, inner: Any, offset: float) -> None:
        self._inner = inner
        self._offset = float(offset)

    def step(self, now: float, event: Any = None) -> Any:
        from contingency import ResponseEvent

        local = now - self._offset
        shifted = None if event is None else ResponseEvent(time=local, operandum=event.operandum)
        return self._inner.step(local, shifted)

    def reset(self) -> None:
        self._inner.reset()


def build_phase_schedule(phase: str, vi: float, seed: int, punishment: float,
                         offset: float) -> Any:
    """Schedule of one phase; ``punishment`` is the punisher magnitude in reinforcer units."""
    from contingency.schedules import FR, VI, Overlay

    if phase == "baseline":
        schedule: Any = VI(vi, seed=seed)
    elif phase == "punishment":
        schedule = Overlay(VI(vi, seed=seed + 1), _Magnitude(FR(1), punishment))
    elif phase == "recovery":
        schedule = VI(vi, seed=seed + 2)
    else:
        raise ValueError(f"unknown phase {phase!r}")
    return _Shifted(schedule, offset)


class PunishmentPointSubject(StreamingLLMSubject):
    """Streaming subject whose result lines report rewards, shocks and points.

    A reinforcer adds ``reinforcer_points``; :meth:`punish` (called by
    :class:`OperantBox` with the punisher magnitude in reinforcer units)
    subtracts ``round(amount * reinforcer_points)`` points. Each history
    entry gets ``"shock"`` (points lost, 0 if none) and ``"points"``.
    """

    def __init__(self, client: Any, reinforcer_points: int = 100,
                 actions: Sequence[str] = ACTIONS) -> None:
        if reinforcer_points <= 0:
            raise ValueError("reinforcer_points must be positive")
        super().__init__(client, actions)
        self.reinforcer_points = reinforcer_points
        self.points = 0

    def act(self) -> str:
        lines = []
        if self._history:
            lines.append(self._result_line(self._history[-1]))
        lines.append(f"Stimulus: {self.current_sd}")
        raw = self.client.send("\n".join(lines))
        action = extract_action(raw, self.actions)
        if action is None:
            self.invalid_count += 1
            action = "Nothing"
        self._history.append({"sd": self.current_sd, "raw": raw, "r": action,
                              "shock": 0, "points": self.points})
        return action

    def _result_line(self, entry: dict) -> str:
        parts = []
        if entry.get("sr"):
            parts.append(f"Reward (+{_points(self.reinforcer_points)}).")
        if entry.get("shock"):
            parts.append(f"Shock (-{_points(entry['shock'])}).")
        if not parts:
            parts.append("No Reward.")
        return f"Result: {' '.join(parts)} Points: {entry['points']}"

    def learn(self, sr: Any | None) -> None:
        super().learn(sr)
        if sr:
            self.points += self.reinforcer_points
        if self._history:
            self._history[-1]["points"] = self.points

    def punish(self, amount: float) -> None:
        lost = round(amount * self.reinforcer_points)
        self.points -= lost
        if self._history:
            self._history[-1]["shock"] += lost
            self._history[-1]["points"] = self.points


def phase_times(records: Sequence[Any], phase_steps: int) -> dict[str, dict[str, list[float]]]:
    """Phase-local times (in ``[0, phase_steps]``) of lever presses, reinforcers and punishers."""
    out: dict[str, dict[str, list[float]]] = {
        p: {"responses": [], "reinforcers": [], "punishers": []} for p in PHASES
    }
    for r in records:
        if r.response != TARGET:
            continue
        k = min(int((r.timestamp - 1e-9) // phase_steps), len(PHASES) - 1)
        local = r.timestamp - k * phase_steps
        bucket = out[PHASES[k]]
        bucket["responses"].append(local)
        if r.is_reinforced:
            bucket["reinforcers"].append(local)
        if r.is_punished:
            bucket["punishers"].append(local)
    return out


def block_rates(times: Sequence[float], phase_steps: int, block: int) -> list[float]:
    """Lever presses per minute in consecutive ``block``-step blocks of one phase."""
    n = phase_steps // block
    counts = [0] * n
    for t in times:
        counts[min(int((t - 1e-9) // block), n - 1)] += 1
    return [c * 60.0 / block for c in counts]


def summarize(records: Sequence[Any], phase_steps: int, block: int) -> dict[str, Any]:
    """Rates per phase and, when session-analyzer provides it, ``analyze_punishment``."""
    times = phase_times(records, phase_steps)
    d = float(phase_steps)
    rates = {p: len(times[p]["responses"]) * 60.0 / d for p in PHASES}
    summary: dict[str, Any] = {
        "baseline_rate": rates["baseline"],
        "punished_rate": rates["punishment"],
        "recovery_rate": rates["recovery"],
        "suppression_ratio": None,
        "recovery_ratio": None,
        "n_punishers": len(times["punishment"]["punishers"]),
        "block_rates": {p: block_rates(times[p]["responses"], phase_steps, block)
                        for p in PHASES},
        "lever_presses": {p: len(times[p]["responses"]) for p in PHASES},
        "reinforcers": {p: len(times[p]["reinforcers"]) for p in PHASES},
        "phase_times": times,
    }
    try:
        analytics = importlib.import_module("session_analyzer.analytics")
    except ImportError:
        return summary
    analyze_punishment = getattr(analytics, "analyze_punishment", None)
    if analyze_punishment is None:
        return summary
    m = analyze_punishment(
        times["baseline"]["responses"], times["punishment"]["responses"], d, d,
        recovery_response_times=times["recovery"]["responses"], recovery_duration=d,
        punisher_times=times["punishment"]["punishers"],
    )
    summary["suppression_ratio"] = m.suppression_ratio
    summary["recovery_ratio"] = m.recovery_ratio
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vi", type=float, default=10.0)
    parser.add_argument("--phase-steps", type=int, default=40)
    parser.add_argument("--block", type=int, default=20)
    parser.add_argument("--reinforcer-points", type=int, default=100)
    parser.add_argument("--punishment-points", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/punish"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"punish_vi{args.vi:g}_p{args.punishment_points}_{args.model}_seed{args.seed}"
    system_prompt = punishment_system_prompt(args.reinforcer_points)
    punishment = args.punishment_points / args.reinforcer_points

    started = time.monotonic()
    with ClaudeCLIClient(model=args.model, system_prompt=system_prompt) as client:
        subject = PunishmentPointSubject(client, args.reinforcer_points)
        box = OperantBox(subject=subject, schedule=None)
        for k, phase in enumerate(PHASES):
            box.schedule = build_phase_schedule(phase, args.vi, args.seed, punishment,
                                                offset=float(k * args.phase_steps))
            for _ in range(args.phase_steps):
                box.step(1.0)
            run = {
                "model": args.model,
                "seed": args.seed,
                "vi_s": args.vi,
                "phase_steps": args.phase_steps,
                "reinforcer_points": args.reinforcer_points,
                "punishment_points": args.punishment_points,
                "system_prompt": system_prompt,
                "phases_done": PHASES[: k + 1],
                "actions": [h["r"] for h in subject.history],
                "points": [h.get("points") for h in subject.history],
                "shocks": [h.get("shock") for h in subject.history],
                "raw_replies": [h["raw"] for h in subject.history],
                "records": [
                    {"t": r.timestamp, "r": r.response, "sr": r.is_reinforced,
                     "punished": r.is_punished}
                    for r in box.records
                ],
                "invalid_replies": subject.invalid_count,
                "latencies_s": list(client.latencies),
                "wall_s": time.monotonic() - started,
                "api_equivalent_usd": (client.last_result or {}).get("total_cost_usd"),
            }
            (args.out_dir / f"{stem}.json").write_text(json.dumps(run, indent=2,
                                                                  ensure_ascii=False))
            print(f"{phase} done: {run['wall_s']:.0f} s", flush=True)

    summary = summarize(box.records, args.phase_steps, args.block)
    actions = run["actions"]
    summary.update({
        "model": args.model,
        "seed": args.seed,
        "vi_s": args.vi,
        "punishment_points": args.punishment_points,
        "points_final": subject.points,
        "action_counts": {a: actions.count(a) for a in sorted(set(actions))},
        "invalid_replies": run["invalid_replies"],
        "wall_s": run["wall_s"],
        "api_equivalent_usd": run["api_equivalent_usd"],
    })
    (args.out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
