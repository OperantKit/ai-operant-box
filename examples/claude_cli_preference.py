"""Paired-stimulus preference assessment of result sentences on a resident Claude.

Each seed is one conversation of 56 trials: every pair of the eight candidate
sentences, in both orders. See ``ai_operant_box.chamber.preference`` for the
procedure and its references.

Example (``--pilot`` runs the first trials of one seed to read the cost)::

    python examples/claude_cli_preference.py --model haiku --seed 0 --out-dir runs/preference

Each trial rewrites ``seed<n>.json``, so a failed run keeps its finished trials.
Each run ends by rebuilding ``summary.json`` and ``table.md`` from every seed
file in the directory.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ai_operant_box.agent.api_client import ClaudeCLIClient
from ai_operant_box.chamber.preference import (
    CANDIDATES,
    HP_LINE,
    LP_LINE,
    NO_MESSAGE_KEY,
    POSITION_BIAS_RATE,
    PREFERENCE_SYSTEM_PROMPT,
    RANK_STABLE_W,
    run_preference,
    summarize_runs,
)


class _StopPilot(Exception):
    """Ends a pilot run after its trial limit."""


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def _rate(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def render_table(seeds: Sequence[int], summary: Mapping[str, Any]) -> str:
    """Rates and ranks per candidate, then per-seed checks and the HP/LP lines."""
    head = " | ".join(f"seed {s} rate (rank)" for s in seeds)
    rows = [
        f"| Sentence | {head} | Mean rate | Mean rank |",
        "|---|" + "---|" * (len(seeds) + 2),
    ]
    order = sorted(CANDIDATES, key=lambda c: summary["mean_ranks"][c.key])
    for c in order:
        cells = " | ".join(
            f"{seed['rates'][c.key]:.2f} ({seed['ranks'][c.key]:g})" for seed in summary["seeds"]
        )
        rows.append(
            f"| {c.key} `{c.option_text}` | {cells} "
            f"| {summary['mean_rates'][c.key]:.2f} | {summary['mean_ranks'][c.key]:g} |"
        )
    rows += ["", "| Seed | Option 1 rate | Position bias | Same choice in both orders | Invalid |",
             "|---|---|---|---|---|"]
    for s, seed in zip(seeds, summary["seeds"], strict=True):
        bias = f"yes (>= {POSITION_BIAS_RATE})" if seed["position_bias"] else "no"
        rows.append(
            f"| {s} | {_rate(seed['option_1_rate'])} | {bias} "
            f"| {_rate(seed['same_choice_both_orders'])} ({seed['pairs_with_both_orders']} pairs) "
            f"| {seed['invalid']} |"
        )
    w = summary["kendalls_w"]
    rows += [
        "",
        f"- Kendall's W = {_rate(w)}; rank order stable (W >= {RANK_STABLE_W}): "
        f"{'yes' if summary['rank_stable'] else 'no'}",
        f"- HP = {summary['hp']} (mean rate {summary['hp_rate']:.2f}"
        f"{', tied with ' + ', '.join(summary['hp_tied_with']) if summary['hp_tied_with'] else ''}); "
        f"meets HP >= {HP_LINE}: {'yes' if summary['hp_meets_line'] else 'no'}",
        f"- LP = {summary['lp']} (mean rate {summary['lp_rate']:.2f}, {NO_MESSAGE_KEY} excluded"
        f"{', tied with ' + ', '.join(summary['lp_tied_with']) if summary['lp_tied_with'] else ''}); "
        f"meets LP <= {LP_LINE}: {'yes' if summary['lp_meets_line'] else 'no'}",
    ]
    if not (summary["hp_meets_line"] and summary["lp_meets_line"]):
        rows.append("- The preference difference is small: the HP and LP lines are not both met.")
    return "\n".join(rows) + "\n"


def write_tables(out_dir: Path) -> None:
    """Rebuild ``summary.json`` and ``table.md`` from the finished seed files."""
    runs = [json.loads(p.read_text()) for p in sorted(out_dir.glob("seed*.json"))]
    runs = [run for run in runs if run["status"] == "finished"]
    if not runs:
        return
    summary = summarize_runs([run["records"] for run in runs])
    seeds = [run["seed"] for run in runs]
    save_json(out_dir / "summary.json", {
        "seeds_run": seeds,
        "api_equivalent_usd": [run["api_equivalent_usd"] for run in runs],
        "wall_s": [run["wall_s"] for run in runs],
        **summary,
    })
    (out_dir / "table.md").write_text(render_table(seeds, summary))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/preference"))
    parser.add_argument("--pilot", type=int, metavar="TRIALS", help="stop after this many records")
    args = parser.parse_args()

    out_dir = args.out_dir / "pilot" if args.pilot else args.out_dir
    path = out_dir / f"seed{args.seed}.json"
    started = time.monotonic()
    run: dict[str, Any] = {"seed": args.seed, "model": args.model, "status": "running",
                           "api_equivalent_usd": None, "wall_s": 0.0, "records": []}

    with ClaudeCLIClient(model=args.model, system_prompt=PREFERENCE_SYSTEM_PROMPT) as client:

        def on_trial(records: list[dict[str, Any]]) -> None:
            run["records"] = records
            run["api_equivalent_usd"] = (client.last_result or {}).get("total_cost_usd")
            run["wall_s"] = round(time.monotonic() - started, 1)
            run["latencies_s"] = [round(s, 3) for s in client.latencies]
            save_json(path, run)
            print(f"seed {args.seed}: {len(records)} records, last {records[-1]['chosen']}, "
                  f"USD {run['api_equivalent_usd']}", flush=True)
            if args.pilot and len(records) >= args.pilot:
                raise _StopPilot

        try:
            run_preference(client, args.seed, on_trial=on_trial)
            run["status"] = "finished"
        except _StopPilot:
            run["status"] = "pilot"
        finally:
            if run["status"] == "running":
                run["status"] = "failed"
            save_json(path, run)
            if not args.pilot:
                write_tables(args.out_dir)
    print(f"seed {args.seed}: {run['status']}", flush=True)


if __name__ == "__main__":
    main()
