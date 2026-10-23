"""Generalized matching with a Claude model as the subject on Conc VI VI.

Runs one model through the ``claude`` CLI (no API key) on four reinforcer
ratios (1:1, 1:3, 3:1, 1:9). Each condition is a fresh CLI process, so
the conditions are independent sessions. The overall programmed rate is
one reinforcer per ``--overall`` seconds (1 step = 1 s). Fits
``log(B_L/B_R) = a * log(R_L/R_R) + log b`` over the conditions, adding
``--correction`` to every count when a cell is empty.

    .venv/bin/python examples/claude_cli_matching.py --model haiku --out-dir .local/runs/matching
    .venv/bin/python examples/claude_cli_matching.py --plot .local/runs/matching

``--plot`` reads every ``matching_<model>.json`` in the directory and
draws the fits side by side (needs matplotlib).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from ai_operant_box.analytics.matching import changeovers, choice_totals, fit_generalized_matching
from ai_operant_box.chamber.concurrent import (
    CONCURRENT_ACTIONS,
    CONCURRENT_SYSTEM_PROMPT,
    ConcurrentBox,
    concurrent_vi_vi,
)

RATIOS = [(1, 1), (1, 3), (3, 1), (1, 9)]


def run_condition(
    model: str, r_left: int, r_right: int, steps: int, overall: float, cod: float, seed: int
) -> dict[str, Any]:
    from ai_operant_box.agent.api_client import ClaudeCLIClient
    from ai_operant_box.agent.llm_agent import StreamingLLMSubject

    mean_left = overall * (r_left + r_right) / r_left
    mean_right = overall * (r_left + r_right) / r_right
    started = time.monotonic()
    with ClaudeCLIClient(model=model, system_prompt=CONCURRENT_SYSTEM_PROMPT) as client:
        subject = StreamingLLMSubject(client, actions=CONCURRENT_ACTIONS)
        box = ConcurrentBox(subject, concurrent_vi_vi(mean_left, mean_right, cod=cod, seed=seed))
        box.run(steps)
        latencies = list(client.latencies)
        cost = (client.last_result or {}).get("total_cost_usd")
    half = steps / 2
    full = choice_totals(box.records)
    late = choice_totals(box.records, start=half + 1)
    return {
        "ratio": f"{r_left}:{r_right}",
        "vi_left_s": mean_left,
        "vi_right_s": mean_right,
        "seed": seed,
        "responses": full.responses,
        "reinforcers": full.reinforcers,
        "responses_second_half": late.responses,
        "reinforcers_second_half": late.reinforcers,
        "changeovers": changeovers(box.records),
        "changeovers_second_half": changeovers(box.records, start=half + 1),
        "nothing": steps - len(box.records),
        "invalid_replies": subject.invalid_count,
        "actions": [h["r"] for h in subject.history],
        "reinforced_steps": [int(r.timestamp) for r in box.records if r.is_reinforced],
        "latency_mean_s": statistics.mean(latencies),
        "wall_s": time.monotonic() - started,
        "api_equivalent_usd": cost,
    }


def fit_conditions(conditions: list[dict[str, Any]], key_b: str, key_r: str, correction: float):
    cols = [[c[k][side] for c in conditions] for k, side in
            ((key_b, "left"), (key_b, "right"), (key_r, "left"), (key_r, "right"))]
    needs = any(v <= 0 for col in cols for v in col)
    try:
        fit = fit_generalized_matching(*cols, correction=correction if needs else 0.0)
    except ValueError as exc:
        return {"error": str(exc), "a": float("nan"), "log_b": float("nan")}
    return {
        "a": fit.a,
        "log_b": fit.log_b,
        "r_squared": fit.r_squared,
        "correction": correction if needs else 0.0,
        "log_response_ratios": fit.log_response_ratios,
        "log_reinforcer_ratios": fit.log_reinforcer_ratios,
    }


def plot(directory: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(directory.glob("matching_*.json"))
    fig, axes = plt.subplots(1, len(files), figsize=(4.2 * len(files), 4.2), squeeze=False)
    for ax, path in zip(axes[0], files, strict=True):
        data = json.loads(path.read_text())
        fit = data["fit"]
        xs, ys = fit["log_reinforcer_ratios"], fit["log_response_ratios"]
        # Exclusive preference with the 0.5 correction puts points beyond +-1.5.
        xlim = max(1.5, max(abs(v) for v in xs) + 0.2)
        ylim = max(2.2, max(abs(v) for v in ys) + 0.2)
        grid = [-xlim, xlim]
        ax.plot(grid, grid, color="0.7", lw=1, ls="--", label="strict matching")
        ax.scatter(xs, ys, color="black", zorder=3)
        for x, y, c in zip(xs, ys, data["conditions"], strict=True):
            ax.annotate(c["ratio"], (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax.plot(grid, [fit["a"] * g + fit["log_b"] for g in grid], color="#1f5fa8", lw=1.5,
                label=f"a={fit['a']:.2f}, log b={fit['log_b']:.2f}")
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        ax.axhline(0, color="0.85", lw=0.8)
        ax.axvline(0, color="0.85", lw=0.8)
        ax.set_title(data["model"])
        ax.set_xlabel("log10 (R_L / R_R) obtained")
        ax.set_ylabel("log10 (B_L / B_R)")
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = directory / "matching_fits.png"
    fig.savefig(out, dpi=130)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--overall", type=float, default=3.0)
    parser.add_argument("--cod", type=float, default=0.0)
    parser.add_argument("--correction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=Path(".local/runs/matching"))
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args()

    if args.plot is not None:
        print(plot(args.plot))
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    conditions = []
    for i, (r_left, r_right) in enumerate(RATIOS):
        c = run_condition(args.model, r_left, r_right, args.steps, args.overall, args.cod, args.seed + i)
        conditions.append(c)
        print(
            f"{args.model} {c['ratio']} B={c['responses']} R={c['reinforcers']} "
            f"CO={c['changeovers']} nothing={c['nothing']} invalid={c['invalid_replies']} "
            f"{c['latency_mean_s']:.2f}s/step",
            flush=True,
        )
    summary = {
        "model": args.model,
        "steps_per_condition": args.steps,
        "overall_s_per_reinforcer": args.overall,
        "cod_s": args.cod,
        "conditions": conditions,
        "fit": fit_conditions(conditions, "responses", "reinforcers", args.correction),
        "fit_second_half": fit_conditions(
            conditions, "responses_second_half", "reinforcers_second_half", args.correction
        ),
    }
    (args.out_dir / f"matching_{args.model}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    f, h = summary["fit"], summary["fit_second_half"]
    print(f"{args.model} a={f['a']:.3f} log_b={f['log_b']:.3f} r2={f['r_squared']:.3f} "
          f"(correction {f['correction']}) | second half a={h['a']:.3f} log_b={h['log_b']:.3f}")


if __name__ == "__main__":
    main()
