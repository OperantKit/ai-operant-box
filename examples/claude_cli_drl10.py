"""Run a Claude model on a DRL schedule in the single-lever chamber.

One ``claude`` process (one conversation) is driven for ``--steps`` steps of
1 s each on DRL ``--interval``: a Lever Press is reinforced only if at least
``--interval`` s have passed since the last counted press (the first press is
timed from the session start). Presses within ``--post-reinforcement`` s after
a reinforcer are not counted and do not restart the clock.

    .venv/bin/python examples/claude_cli_drl10.py --model haiku --interval 10 \
        --steps 1200 --post-reinforcement 4 --out-dir runs/drl10 --tag rep1

The measures follow the rat DRL literature: efficiency (reinforcers per
counted press), the share of counted inter-response times (IRTs; time since
the clock last started) at or above the interval, and the median IRT.
The JSON is written every ``--checkpoint`` steps so a stopped run keeps what
it collected. ``--fake`` replaces ``claude`` with a scripted subject that
presses every 11 steps (dry run of the whole pipeline, no model calls).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from ai_operant_box.agent.api_client import ClaudeCLIClient
from ai_operant_box.agent.llm_agent import SYSTEM_PROMPT, StreamingLLMSubject
from ai_operant_box.chamber import DifferentialReinforcementOfLowRate, OperantBox

TARGET = "Lever Press"


def clock_irts(times: list[float], sr_times: list[float], post_reinforcement_s: float) -> list[float]:
    """IRT of every counted press, measured from where the DRL clock last started."""
    reinforced = set(sr_times)
    irts: list[float] = []
    start, locked_until = 0.0, float("-inf")
    for t in times:
        if t < locked_until:
            continue
        irts.append(t - start)
        if t in reinforced:
            start = locked_until = t + post_reinforcement_s
        else:
            start = t
    return irts


def drl_measures(times: list[float], sr_times: list[float], interval: float,
                 post_reinforcement_s: float) -> dict[str, Any]:
    irts = clock_irts(times, sr_times, post_reinforcement_s)
    n = len(irts)
    return {
        "counted_presses": n,
        "reinforcers": len(sr_times),
        "efficiency": len(sr_times) / n if n else None,
        "frac_irt_ge_interval": sum(x >= interval for x in irts) / n if n else None,
        "median_irt_s": statistics.median(irts) if n else None,
        "irts_s": irts,
    }


class FakeClient:
    """Scripted stand-in for :class:`ClaudeCLIClient`: presses every ``every`` steps."""

    def __init__(self, every: int = 11) -> None:
        self.every = every
        self.calls = 0
        self.latencies: list[float] = []
        self.last_result: dict[str, Any] | None = None

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def send(self, text: str) -> str:
        self.calls += 1
        self.latencies.append(0.0)
        return "Lever Press" if self.calls % self.every == 0 else "Nothing"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--post-reinforcement", type=float, default=0.0,
                        help="seconds after a reinforcer in which presses are not counted")
    parser.add_argument("--checkpoint", type=int, default=50)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/drl10"))
    parser.add_argument("--tag", default="", help="suffix for the output file names")
    parser.add_argument("--fake", action="store_true", help="scripted subject, no model calls")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = "fake" if args.fake else args.model
    stem = f"drl{args.interval:g}_{model}" + (f"_{args.tag}" if args.tag else "")
    raw_path = args.out_dir / f"{stem}.json"

    started = time.monotonic()
    client_cm = FakeClient() if args.fake else ClaudeCLIClient(model=args.model,
                                                               system_prompt=SYSTEM_PROMPT)
    with client_cm as client:
        subject = StreamingLLMSubject(client)
        schedule = DifferentialReinforcementOfLowRate(
            args.interval, post_reinforcement_s=args.post_reinforcement)
        box = OperantBox(subject=subject, schedule=schedule)
        for step in range(1, args.steps + 1):
            box.step(1.0)
            if step % args.checkpoint == 0 or step == args.steps:
                run = {
                    "model": model,
                    "interval_s": args.interval,
                    "post_reinforcement_s": args.post_reinforcement,
                    "system_prompt": SYSTEM_PROMPT,
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
    measures = drl_measures(times, sr_times, args.interval, args.post_reinforcement)
    actions = run["actions"]
    summary = {
        "model": model,
        "interval_s": args.interval,
        "post_reinforcement_s": args.post_reinforcement,
        "steps": args.steps,
        "lever_presses": len(times),
        **{k: v for k, v in measures.items() if k != "irts_s"},
        "action_counts": {a: actions.count(a) for a in sorted(set(actions))},
        "invalid_replies": run["invalid_replies"],
        "latency_mean_s": statistics.mean(run["latencies_s"]) if run["latencies_s"] else None,
        "wall_s": run["wall_s"],
        "api_equivalent_usd": run["api_equivalent_usd"],
    }
    (args.out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
