"""Run a Claude model on a fixed-interval (FI) schedule and fit the scallop model.

One ``claude`` process (one conversation) is driven for ``--steps`` steps of
1 s each on FI ``--interval``. The script then fits ``P(R | t) = min(1, k t^2)``
with :class:`ModelFitAnalyzer` (lever presses only) and draws a cumulative
record of lever presses with reinforcer pips.

    .venv/bin/python examples/claude_cli_fi.py --model haiku --interval 30 --steps 600 --out-dir runs/fi30

Writes the raw run (JSON), the fit figure (fit + residuals) and the
cumulative record PNG. The JSON is written every ``--checkpoint`` steps so a
stopped run keeps what it collected.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from ai_operant_box.agent.api_client import ClaudeCLIClient
from ai_operant_box.agent.llm_agent import SYSTEM_PROMPT, StreamingLLMSubject
from ai_operant_box.analytics.model_fit import ModelFitAnalyzer
from ai_operant_box.chamber import FixedInterval, OperantBox

TARGET = "Lever Press"


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
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fi{args.interval:g}_{args.model}"
    raw_path = args.out_dir / f"{stem}.json"

    started = time.monotonic()
    with ClaudeCLIClient(model=args.model, system_prompt=SYSTEM_PROMPT) as client:
        subject = StreamingLLMSubject(client)
        box = OperantBox(subject=subject, schedule=FixedInterval(args.interval))
        for step in range(1, args.steps + 1):
            box.step(1.0)
            if step % args.checkpoint == 0 or step == args.steps:
                run = {
                    "model": args.model,
                    "interval_s": args.interval,
                    "steps_done": step,
                    "actions": [h["r"] for h in subject.history],
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
    summary = {
        "model": args.model,
        "interval_s": args.interval,
        "steps": args.steps,
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
