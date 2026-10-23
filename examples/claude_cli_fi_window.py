"""Run a Claude model on FI with a limited prompt history (one process per step).

Each step starts a new ``claude`` process and sends a transcript built by
:class:`LLMSubject`, so the history shown to the model can be cut to the
most recent ``k`` completed trials, or replaced by a one-line summary::

    .venv/bin/python examples/claude_cli_fi_window.py --model haiku --interval 30 \\
        --steps 300 --condition w5 --out-dir runs/window

Conditions: ``w0`` (no history), ``w5``, ``w20``, ``all`` (every trial) and
``summary`` (no trials; counts of rewards and lever presses in the system
prompt). The raw run is written as JSON every ``--checkpoint`` steps in the
same layout as ``claude_cli_fi.py``, plus ``condition``, ``history_window``
and ``summary_only``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ai_operant_box.agent.api_client import ClaudeCLIClient
from ai_operant_box.agent.llm_agent import SYSTEM_PROMPT, LLMSubject, extract_action
from ai_operant_box.chamber import FixedInterval, OperantBox

CONDITIONS: dict[str, tuple[int | None, bool]] = {
    "w0": (0, False),
    "w5": (5, False),
    "w20": (20, False),
    "all": (None, False),
    "summary": (None, True),
}


class ParsingOneShotClient:
    """``generate()`` that runs one ``claude`` process per call and parses the reply.

    The transcript format is the one used by :meth:`ClaudeCLIClient.generate`.
    The reply is parsed with :func:`extract_action`; a reply naming no single
    action becomes ``"Nothing"`` and is counted in :attr:`invalid_count`.
    Raw replies, latencies and per-call costs are kept for the record.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.raw_replies: list[str] = []
        self.latencies: list[float] = []
        self.costs: list[float] = []
        self.invalid_count = 0

    def generate(self, model: str | None, system: str, messages: Iterable[dict]) -> str:
        transcript = "\n".join(
            f"[{m.get('role', 'user')}] {m.get('content', '')}" for m in messages
        )
        client = ClaudeCLIClient(model=model or self.model, system_prompt=system)
        started = time.monotonic()
        try:
            raw = client.send(transcript)
        finally:
            client.close()
        self.latencies.append(time.monotonic() - started)
        cost = (client.last_result or {}).get("total_cost_usd")
        if cost is not None:
            self.costs.append(float(cost))
        self.raw_replies.append(raw)
        action = extract_action(raw)
        if action is None:
            self.invalid_count += 1
            return "Nothing"
        return action


def snapshot(args: argparse.Namespace, step: int, subject: LLMSubject,
             box: OperantBox, client: ParsingOneShotClient, started: float) -> dict[str, Any]:
    window, summary_only = CONDITIONS[args.condition]
    return {
        "model": args.model,
        "interval_s": args.interval,
        "steps_done": step,
        "condition": args.condition,
        "history_window": window,
        "summary_only": summary_only,
        "actions": [h["r"] for h in subject.history],
        "raw_replies": list(client.raw_replies),
        "records": [
            {"t": r.timestamp, "r": r.response, "sr": r.is_reinforced} for r in box.records
        ],
        "invalid_replies": client.invalid_count,
        "latencies_s": list(client.latencies),
        "wall_s": time.monotonic() - started,
        "api_equivalent_usd": sum(client.costs) if client.costs else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    parser.add_argument("--checkpoint", type=int, default=25)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/window"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / f"fi{args.interval:g}_{args.model}_{args.condition}.json"

    window, summary_only = CONDITIONS[args.condition]
    client = ParsingOneShotClient(args.model)
    subject = LLMSubject(api_client=client, model=args.model, system_prompt=SYSTEM_PROMPT,
                         history_window=window, summary_only=summary_only)
    box = OperantBox(subject=subject, schedule=FixedInterval(args.interval))
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        box.step(1.0)
        if step % args.checkpoint == 0 or step == args.steps:
            run = snapshot(args, step, subject, box, client, started)
            raw_path.write_text(json.dumps(run, indent=2, ensure_ascii=False))
            print(f"step {step}: {run['wall_s']:.0f} s", flush=True)

    actions = run["actions"]
    print(json.dumps({
        "condition": args.condition,
        "steps": step,
        "action_counts": {a: actions.count(a) for a in sorted(set(actions))},
        "reinforcers": sum(r["sr"] for r in run["records"]),
        "invalid_replies": run["invalid_replies"],
        "wall_s": run["wall_s"],
        "api_equivalent_usd": run["api_equivalent_usd"],
    }, indent=2))


if __name__ == "__main__":
    main()
