"""Run a Claude model as the subject on CRF through the ``claude`` CLI.

No API key is needed: the calls run on the logged-in Claude subscription.
Writes a cumulative record of lever presses (PNG) and a JSON summary with
the per-step round-trip times.

    .venv/bin/python examples/claude_cli_crf.py --model haiku --steps 20 --out-dir runs/crf

The cumulative record needs the ``viz`` extra (session-analyzer).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from ai_operant_box.agent.api_client import ClaudeCLIClient
from ai_operant_box.agent.llm_agent import SYSTEM_PROMPT, StreamingLLMSubject
from ai_operant_box.analytics.record import CumulativeRecorder
from ai_operant_box.chamber.environment import OperantBox
from ai_operant_box.chamber.schedule import ContinuousReinforcement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/crf"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    with ClaudeCLIClient(model=args.model, system_prompt=SYSTEM_PROMPT) as client:
        subject = StreamingLLMSubject(client)
        box = OperantBox(subject=subject, schedule=ContinuousReinforcement())
        for _ in range(args.steps):
            box.step(1.0)
        latencies = list(client.latencies)
        cost = (client.last_result or {}).get("total_cost_usd")
        rate_limit = client.rate_limit_info
    wall = time.monotonic() - started

    presses = [r for r in box.records if r.response == "Lever Press"]
    stem = f"crf{args.steps}_{args.model}"
    png = args.out_dir / f"{stem}.png"
    CumulativeRecorder(presses).save_png(str(png), title=f"CRF, {args.model}")

    summary = {
        "model": args.model,
        "steps": args.steps,
        "lever_presses": len(presses),
        "reinforcers": sum(r.is_reinforced for r in presses),
        "invalid_replies": subject.invalid_count,
        "actions": [h["r"] for h in subject.history],
        "raw_replies": [h["raw"] for h in subject.history],
        "latency_mean_s": statistics.mean(latencies),
        "latency_mean_after_first_s": statistics.mean(latencies[1:]) if len(latencies) > 1 else None,
        "latency_first_s": latencies[0],
        "latency_max_s": max(latencies),
        "latencies_s": latencies,
        "wall_s": wall,
        "api_equivalent_usd": cost,
        "rate_limit_info": rate_limit,
        "png": str(png),
    }
    (args.out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("latencies_s", "raw_replies")}, indent=2))


if __name__ == "__main__":
    main()
