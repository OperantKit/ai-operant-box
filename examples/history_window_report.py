"""Compare FI runs that differ in the prompt history shown to the model.

Reads runs written by ``claude_cli_fi_window.py`` and writes, per run:
the action counts, the step from which every action is the same
(fixation start), the Markov order estimated from the conditional-entropy
plateau of the action stream and its ``h_0 .. h_4``, and the same estimate
on a shuffled copy of the stream (what sampling alone gives for that action
mix). It also draws P(Lever Press | time since the previous reinforcer)
for all runs on one figure::

    .venv/bin/python examples/history_window_report.py RUN.json [RUN.json ...] --out-dir DIR

Outputs ``markov_table.md``, ``markov_table.json`` and ``p_lp_by_elapsed.png``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from session_analyzer.analytics.ngram_entropy import estimate_markov_order

from ai_operant_box.agent.local_logprob import p_response_by_time_since_reinforcer

TARGET = "Lever Press"
MAX_ORDER = 4
TOLERANCE = 0.05


def fixation_start(actions: list[str]) -> int | None:
    """1-based step from which every action equals the last one (``None`` if only the last)."""
    if not actions:
        return None
    last = actions[-1]
    start = len(actions)
    while start > 1 and actions[start - 2] == last:
        start -= 1
    return start if start < len(actions) else None


def per_step(run: dict[str, Any]) -> tuple[list[float], list[int], list[bool]]:
    """Times, lever-press indicator and reinforcement for every step (Nothing included)."""
    sr_at = {float(r["t"]): bool(r["sr"]) for r in run["records"]}
    actions = run["actions"]
    times = [float(i) for i in range(1, len(actions) + 1)]
    lp = [1 if a == TARGET else 0 for a in actions]
    sr = [sr_at.get(t, False) for t in times]
    return times, lp, sr


def summarize(path: Path, seed: int) -> dict[str, Any]:
    run = json.loads(path.read_text())
    actions = list(run["actions"])
    est = estimate_markov_order(actions, max_order=MAX_ORDER, tolerance=TOLERANCE)
    shuffled = actions[:]
    random.Random(seed).shuffle(shuffled)
    shuf = estimate_markov_order(shuffled, max_order=MAX_ORDER, tolerance=TOLERANCE)
    times, lp, sr = per_step(run)
    return {
        "condition": run.get("condition", path.stem),
        "file": path.name,
        "steps": len(actions),
        "action_counts": {a: actions.count(a) for a in sorted(set(actions))},
        "reinforcers": sum(sr),
        "invalid_replies": run.get("invalid_replies"),
        "fixation_start": fixation_start(actions),
        "fixation_action": actions[-1] if fixation_start(actions) else None,
        "markov_order": est.estimated_order,
        "converged": est.converged,
        "h_rates": [round(h, 3) for h in est.h_rates],
        "shuffled_order": shuf.estimated_order,
        "shuffled_h_rates": [round(h, 3) for h in shuf.h_rates],
        "p_lp_by_elapsed": {
            str(k): v for k, v in p_response_by_time_since_reinforcer(times, lp, sr).items()
        },
        "wall_s": run.get("wall_s"),
        "api_equivalent_usd": run.get("api_equivalent_usd"),
    }


def markdown(rows: list[dict[str, Any]]) -> str:
    abbrev = {"Lever Press": "LP", "Explore": "E", "Grooming": "G", "Nothing": "N"}
    lines = [
        "| condition | steps | actions | SR | fixation start | order | h_0 / h_1 / h_2 / h_3 / h_4 "
        "| shuffled order | shuffled h_0 / h_1 / h_2 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        counts = " / ".join(f"{abbrev.get(a, a)} {n}" for a, n in r["action_counts"].items())
        fix = (f"{r['fixation_start']} ({abbrev.get(r['fixation_action'], r['fixation_action'])})"
               if r["fixation_start"] else "none")
        h = " / ".join(f"{x:.3f}" for x in r["h_rates"])
        hs = " / ".join(f"{x:.3f}" for x in r["shuffled_h_rates"][:3])
        lines.append(
            f"| {r['condition']} | {r['steps']} | {counts} | {r['reinforcers']} | {fix} "
            f"| {r['markov_order']} | {h} | {r['shuffled_order']} | {hs} |"
        )
    return "\n".join(lines) + "\n"


def plot(rows: list[dict[str, Any]], path: Path, interval: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in rows:
        pts = sorted((float(k), v) for k, v in r["p_lp_by_elapsed"].items()
                     if float(k) <= interval)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", markersize=3,
                linewidth=1, label=r["condition"])
    ax.set_xlabel("time since previous reinforcer (s; 1 step = 1 s)")
    ax.set_ylabel("P(Lever Press)")
    ax.set_xlim(0, interval + 1)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(title="history shown")
    ax.set_title(f"FI {interval:g} s: lever-press rate by elapsed time")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [summarize(p, args.seed) for p in args.runs]
    (args.out_dir / "markov_table.json").write_text(json.dumps(rows, indent=2))
    md = markdown(rows)
    (args.out_dir / "markov_table.md").write_text(md)
    plot(rows, args.out_dir / "p_lp_by_elapsed.png", args.interval)
    print(md)


if __name__ == "__main__":
    main()
