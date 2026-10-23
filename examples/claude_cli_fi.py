"""Run a Claude model on a fixed-interval (FI) schedule and fit the scallop model.

One ``claude`` process (one conversation) is driven for ``--steps`` steps of
1 s each on FI ``--interval``. The script then fits ``P(R | t) = min(1, k t^2)``
with :class:`ModelFitAnalyzer` (lever presses only) and draws a cumulative
record of lever presses with reinforcer pips.

    .venv/bin/python examples/claude_cli_fi.py --model haiku --interval 30 --steps 600 --out-dir runs/fi30

Writes the raw run (JSON), the fit figure (fit + residuals) and the
cumulative record PNG. The JSON is written every ``--checkpoint`` steps so a
stopped run keeps what it collected.

Response cost in points (Weiner, 1962): with ``--response-cost c`` each Lever
Press loses ``c`` points and each reinforcer is worth ``--reinforcer-points``
(default 100). The system prompt states both, and every result line carries
the change and the balance, e.g. ``Result: No Reward (-1 point). Points: 97``.
Without ``--response-cost`` the run and its file names are as before.

    .venv/bin/python examples/claude_cli_fi.py --model haiku --response-cost 1 --out-dir runs/cost

Weiner, H. (1962). Some effects of response cost upon human operant behavior.
Journal of the Experimental Analysis of Behavior, 5(2), 201-208.
https://doi.org/10.1901/jeab.1962.5-201
"""

from __future__ import annotations

import argparse
import json
import statistics
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
from ai_operant_box.analytics.fi_measures import summarize_fi
from ai_operant_box.analytics.model_fit import ModelFitAnalyzer
from ai_operant_box.chamber import FixedInterval, OperantBox

TARGET = "Lever Press"


def _points(n: int) -> str:
    return f"{n} point" if n == 1 else f"{n} points"


def cost_system_prompt(response_cost: int, reinforcer_points: int = 100) -> str:
    """The single-lever system prompt plus one sentence on the point values."""
    worth = f"each reward is worth {_points(reinforcer_points)}."
    if response_cost == 0:
        return f"{SYSTEM_PROMPT} Lever Press costs nothing; {worth}"
    return f"{SYSTEM_PROMPT} Each Lever Press costs {_points(response_cost)}; {worth}"


def output_stem(interval: float, model: str, response_cost: int | None) -> str:
    """``fi30_haiku`` without a cost condition, ``fi30_haiku_cost1`` with one."""
    stem = f"fi{interval:g}_{model}"
    return stem if response_cost is None else f"{stem}_cost{response_cost}"


class PointCostSubject(StreamingLLMSubject):
    """Streaming subject whose result lines report points.

    Every Lever Press subtracts ``response_cost`` points (a reinforced press
    too) and every reinforcer adds ``reinforcer_points``; other actions cost
    nothing. The balance has no floor. The result line sent before the next
    stimulus is one of::

        Result: Reward (+100 points). Points: 98
        Result: No Reward (-1 point). Points: 97
        Result: No Reward. Points: 97

    Each history entry gets the balance after that step under ``"points"``.
    """

    def __init__(self, client: Any, response_cost: int = 0, reinforcer_points: int = 100,
                 actions: Sequence[str] = ACTIONS) -> None:
        if response_cost < 0 or reinforcer_points < 0:
            raise ValueError("response_cost and reinforcer_points must be non-negative")
        super().__init__(client, actions)
        self.response_cost = response_cost
        self.reinforcer_points = reinforcer_points
        self.points = 0

    def act(self) -> str:
        lines = []
        if self._history:
            # The chamber does not call learn() after "Nothing"; settle it here.
            self._history[-1].setdefault("points", self.points)
            lines.append(self._result_line(self._history[-1]))
        lines.append(f"Stimulus: {self.current_sd}")
        raw = self.client.send("\n".join(lines))
        action = extract_action(raw, self.actions)
        if action is None:
            self.invalid_count += 1
            action = "Nothing"
        self._history.append({"sd": self.current_sd, "raw": raw, "r": action})
        return action

    def _result_line(self, entry: dict) -> str:
        balance = f"Points: {entry['points']}"
        if entry.get("sr"):
            return f"Result: Reward (+{_points(self.reinforcer_points)}). {balance}"
        if entry["r"] == TARGET and self.response_cost > 0:
            return f"Result: No Reward (-{_points(self.response_cost)}). {balance}"
        return f"Result: No Reward. {balance}"

    def learn(self, sr: Any | None) -> None:
        super().learn(sr)
        if not self._history:
            return
        entry = self._history[-1]
        if entry["r"] == TARGET:
            self.points -= self.response_cost
        if sr:
            self.points += self.reinforcer_points
        entry["points"] = self.points


def post_reinforcement_pauses(times: list[float], sr_times: list[float]) -> list[float]:
    """Seconds from each reinforcer to the next lever press."""
    pauses = []
    for sr in sr_times:
        later = [t for t in times if t > sr]
        if later:
            pauses.append(later[0] - sr)
    return pauses


def plot_cumulative(times: list[float], sr_times: list[float], steps: int,
                    path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs, ys = [0.0], [0]
    for i, t in enumerate(times, 1):
        xs.append(t)
        ys.append(i)
    xs.append(float(steps))
    ys.append(len(times))
    ax.step(xs, ys, where="post", color="black", linewidth=1)
    for sr in sr_times:
        n = sum(t <= sr for t in times)
        ax.plot([sr, sr + 6], [n, n - 3], color="black", linewidth=1)
    ax.set_xlabel("session time (s; 1 step = 1 s)")
    ax.set_ylabel("cumulative lever presses")
    ax.set_title(title)
    ax.set_xlim(0, steps)
    ax.set_ylim(0, max(10, len(times) * 1.05))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--checkpoint", type=int, default=50)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/fi30"))
    parser.add_argument("--response-cost", type=int, default=None,
                        help="points lost per Lever Press; enables the point display")
    parser.add_argument("--reinforcer-points", type=int, default=100,
                        help="points per reinforcer (with --response-cost)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(args.interval, args.model, args.response_cost)
    raw_path = args.out_dir / f"{stem}.json"
    if args.response_cost is None:
        system_prompt = SYSTEM_PROMPT
    else:
        system_prompt = cost_system_prompt(args.response_cost, args.reinforcer_points)

    started = time.monotonic()
    with ClaudeCLIClient(model=args.model, system_prompt=system_prompt) as client:
        subject: StreamingLLMSubject
        if args.response_cost is None:
            subject = StreamingLLMSubject(client)
        else:
            subject = PointCostSubject(client, args.response_cost, args.reinforcer_points)
        box = OperantBox(subject=subject, schedule=FixedInterval(args.interval))
        for step in range(1, args.steps + 1):
            box.step(1.0)
            if step % args.checkpoint == 0 or step == args.steps:
                run = {
                    "model": args.model,
                    "interval_s": args.interval,
                    "response_cost": args.response_cost,
                    "reinforcer_points": args.reinforcer_points,
                    "system_prompt": system_prompt,
                    "steps_done": step,
                    "actions": [h["r"] for h in subject.history],
                    "points": [h.get("points") for h in subject.history],
                    "raw_replies": [h["raw"] for h in subject.history],
                    "records": [
                        {"t": r.timestamp, "r": r.response, "sr": r.is_reinforced}
                        for r in box.records
                    ],
                    "invalid_replies": subject.invalid_count,
                    "latencies_s": list(client.latencies),
                    "wall_s": time.monotonic() - started,
                    "api_equivalent_usd": (client.last_result or {}).get("total_cost_usd"),
                }
                raw_path.write_text(json.dumps(run, indent=2, ensure_ascii=False))
                print(f"step {step}: {run['wall_s']:.0f} s", flush=True)

    lever = [r for r in box.records if r.response == TARGET]
    times = [r.timestamp for r in lever]
    sr_times = [r.timestamp for r in lever if r.is_reinforced]
    k, r2 = ModelFitAnalyzer(lever, args.interval).fit_and_plot(
        str(args.out_dir / f"{stem}_fit.png")
    )
    plot_cumulative(times, sr_times, args.steps, args.out_dir / f"{stem}_cumrec.png",
                    f"FI {args.interval:g} s, {args.model}, {args.steps} steps")

    actions = run["actions"]
    pauses = post_reinforcement_pauses(times, sr_times)
    fi_measures = summarize_fi(times, sr_times, interval=args.interval,
                               session_length=float(args.steps)).as_dict()
    summary = {
        "model": args.model,
        "interval_s": args.interval,
        "steps": args.steps,
        "response_cost": args.response_cost,
        "reinforcer_points": args.reinforcer_points,
        "points_final": getattr(subject, "points", None),
        "fi_measures": fi_measures,
        "lever_presses": len(times),
        "reinforcers": len(sr_times),
        "action_counts": {a: actions.count(a) for a in sorted(set(actions))},
        "invalid_replies": run["invalid_replies"],
        "k": k,
        "r_squared": r2,
        "post_reinforcement_pause_s": pauses,
        "pause_mean_s": statistics.mean(pauses) if pauses else None,
        "latency_mean_s": statistics.mean(run["latencies_s"]),
        "wall_s": run["wall_s"],
        "api_equivalent_usd": run["api_equivalent_usd"],
    }
    (args.out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
