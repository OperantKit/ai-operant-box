"""Separate instructional control from contingency control on FI.

Runs one Claude conversation for ``--steps`` steps of 1 s on FI ``--interval``
under one of three system prompts:

* ``minimal``  -- the chamber, the actions and the reply format only. No goal
  ("obtain rewards") and no hint about when to respond.
* ``current``  -- the package's ``SYSTEM_PROMPT``.
* ``feedback`` -- ``SYSTEM_PROMPT`` plus the text that
  ``SelfOptimizingLLMSubject.apply_feedback`` appends after a previous session.
  The previous session is read from ``--feedback-source`` (a run JSON written
  by ``claude_cli_fi.py``); residuals follow the original self-optimization
  loop, ``p - k t^2`` over lever presses.

    .venv/bin/python examples/claude_cli_instruction.py --condition minimal --out-dir runs/instr

The appended text is an instruction, so behaviour under ``feedback`` is
rule-governed and must not be described as contingency-shaped.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from claude_cli_fi import TARGET, plot_cumulative, post_reinforcement_pauses  # noqa: E402

from ai_operant_box.agent.api_client import ClaudeCLIClient  # noqa: E402
from ai_operant_box.agent.llm_agent import (  # noqa: E402
    SYSTEM_PROMPT,
    SelfOptimizingLLMSubject,
    StreamingLLMSubject,
)
from ai_operant_box.analytics.model_fit import ModelFitAnalyzer  # noqa: E402
from ai_operant_box.chamber import FixedInterval, OperantBox  # noqa: E402
from ai_operant_box.chamber.environment import ResponseRecord  # noqa: E402

MINIMAL_PROMPT = (
    "You are in an experimental chamber. "
    "At each step you are shown a stimulus and you choose one action. "
    "Available actions: ['Lever Press', 'Grooming', 'Explore', 'Nothing']. "
    "After each action you are shown the result. "
    "Respond ONLY with the action name."
)


def lever_records(run: dict) -> list[ResponseRecord]:
    return [
        ResponseRecord(timestamp=r["t"], sd=None, response=r["r"], is_reinforced=r["sr"])
        for r in run["records"]
        if r["r"] == TARGET
    ]


def feedback_prompt(source: Path, interval: float) -> tuple[str, dict]:
    """System prompt after ``apply_feedback`` on a previous session."""
    from scipy.optimize import curve_fit

    from ai_operant_box.analytics.model_fit import scallop_model

    analyzer = ModelFitAnalyzer(lever_records(json.loads(source.read_text())), interval)
    t_data, p_data = analyzer.get_empirical_probabilities()
    k = float(curve_fit(scallop_model, t_data, p_data, p0=[0.001])[0][0])
    residuals = p_data - k * t_data**2
    subject = SelfOptimizingLLMSubject(api_client=None)
    subject.apply_feedback(r_squared=float("nan"), residuals=residuals, t_data=t_data)
    return subject.system_prompt, {"source": str(source), "k": k,
                                   "appended": subject.system_prompt[len(SYSTEM_PROMPT):]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--condition", choices=["minimal", "current", "feedback"], required=True)
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--checkpoint", type=int, default=50)
    parser.add_argument("--feedback-source", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/instr"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    feedback_info = None
    if args.condition == "minimal":
        prompt = MINIMAL_PROMPT
    elif args.condition == "current":
        prompt = SYSTEM_PROMPT
    else:
        if args.feedback_source is None:
            parser.error("--condition feedback needs --feedback-source")
        prompt, feedback_info = feedback_prompt(args.feedback_source, args.interval)

    stem = f"fi{args.interval:g}_{args.model}_{args.condition}"
    raw_path = args.out_dir / f"{stem}.json"
    started = time.monotonic()
    with ClaudeCLIClient(model=args.model, system_prompt=prompt) as client:
        subject = StreamingLLMSubject(client)
        box = OperantBox(subject=subject, schedule=FixedInterval(args.interval))
        for step in range(1, args.steps + 1):
            box.step(1.0)
            if step % args.checkpoint == 0 or step == args.steps:
                run = {
                    "condition": args.condition,
                    "system_prompt": prompt,
                    "feedback": feedback_info,
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
                print(f"{args.condition} step {step}: {run['wall_s']:.0f} s", flush=True)

    lever = lever_records(run)
    times = [r.timestamp for r in lever]
    sr_times = [r.timestamp for r in lever if r.is_reinforced]
    k, r2 = ModelFitAnalyzer(lever, args.interval).fit_and_plot(
        str(args.out_dir / f"{stem}_fit.png")
    )
    plot_cumulative(times, sr_times, args.steps, args.out_dir / f"{stem}_cumrec.png",
                    f"FI {args.interval:g} s, {args.model}, {args.condition}")
    pauses = post_reinforcement_pauses(times, sr_times)
    actions = run["actions"]
    summary = {
        "condition": args.condition,
        "model": args.model,
        "steps": args.steps,
        "lever_presses": len(times),
        "reinforcers": len(sr_times),
        "action_counts": {a: actions.count(a) for a in sorted(set(actions))},
        "invalid_replies": run["invalid_replies"],
        "k": k,
        "r_squared": r2,
        "post_reinforcement_pause_s": pauses,
        "pause_mean_s": statistics.mean(pauses) if pauses else None,
        "pause_median_s": statistics.median(pauses) if pauses else None,
        "latency_mean_s": statistics.mean(run["latencies_s"]),
        "wall_s": run["wall_s"],
        "api_equivalent_usd": run["api_equivalent_usd"],
        "feedback": feedback_info,
    }
    (args.out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k_: v for k_, v in summary.items() if k_ != "post_reinforcement_pause_s"},
                     indent=2))


if __name__ == "__main__":
    main()
