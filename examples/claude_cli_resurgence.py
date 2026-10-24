"""Run a Claude model through the three phases of a resurgence procedure.

One ``claude`` process (one conversation) is the subject in the two-lever
chamber (:class:`ConcurrentBox`) for 220 steps of 1 s. The schedule is a
contingency-py ``PhasedConcurrent``; the subject gets no stimulus change
at a boundary, only the contingencies change (Craig & Shahan, 2016):

* phase 1 (120 steps): left (target) VI ``--target-vi`` s, right EXT
* phase 2 (80 steps): left EXT, right (alternative) VI ``--alt-vi`` s
  (``--alt-vi 0`` puts the alternative on EXT too)
* phase 3 (20 steps): both EXT

Rates are counted in 20-step blocks (6, 4 and 1 blocks) as responses per
minute, and summarised with session-analyzer ``analyze_resurgence``
(``baseline_window=2, test_window=1``): the target rate in the first
phase-3 block minus the mean of the last two phase-2 blocks.

    .venv/bin/python examples/claude_cli_resurgence.py --model haiku --seed 0 --out-dir runs/resurgence
    .venv/bin/python examples/claude_cli_resurgence.py --plot runs/resurgence

The JSON is written every ``--checkpoint`` steps so a stopped run keeps
what it collected. ``--plot`` reads every ``resurgence_<model>_*.json`` in
the directory and draws the target and alternative rates per block.

Needs a contingency-py that provides ``contingency.schedules.phased`` and
session-analyzer.

Craig, A. R., & Shahan, T. A. (2016). Behavioral momentum theory fails to
account for the effects of reinforcement rate on resurgence. Journal of the
Experimental Analysis of Behavior, 105(3), 375-392.
https://doi.org/10.1002/jeab.207
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ai_operant_box.chamber.concurrent import (
    CONCURRENT_ACTIONS,
    CONCURRENT_SYSTEM_PROMPT,
    ChoiceRecord,
    ConcurrentBox,
)

PHASE_STEPS: tuple[int, int, int] = (120, 80, 20)
PHASE_KEYS: tuple[str, str, str] = ("phase1", "phase2", "phase3")
BLOCK_STEPS = 20
BASELINE_WINDOW = 2
TEST_WINDOW = 1


def build_schedule(
    alt_vi: float | None,
    target_vi: float = 15.0,
    cod: float = 2.0,
    seed: int | None = None,
    phase_steps: tuple[int, int, int] = PHASE_STEPS,
) -> Any:
    """``PhasedConcurrent`` for the three phases (1 step = 1 s).

    Phase ``i`` draws its left VI from ``seed + 2 * i`` and its right VI
    from ``seed + 2 * i + 1``, as ``contingency/ConcPhases-v0`` does.
    """
    from contingency.schedules import EXT, VI, Concurrent
    from contingency.schedules.phased import PhasedConcurrent

    specs = ((target_vi, None), (None, alt_vi), (None, None))
    phases = []
    for i, ((left, right), steps) in enumerate(zip(specs, phase_steps, strict=True)):
        seeds = (None, None) if seed is None else (seed + 2 * i, seed + 2 * i + 1)
        components = {
            key: EXT() if mean is None else VI(mean, seed=s)
            for key, mean, s in (("left", left, seeds[0]), ("right", right, seeds[1]))
        }
        phases.append((Concurrent(components, cod=cod), float(steps)))
    return PhasedConcurrent(phases)


def block_rates(
    records: Iterable[ChoiceRecord],
    phase_steps: tuple[int, int, int] = PHASE_STEPS,
    block: int = BLOCK_STEPS,
) -> dict[str, dict[str, list[float]]]:
    """Responses per minute on each operandum in ``block``-step blocks per phase.

    A response at time ``t`` (1-based steps) falls in block ``ceil(t / block) - 1``
    counted from the start of the session; phases are whole blocks.
    """
    n_blocks = sum(phase_steps) // block
    counts = {"left": [0] * n_blocks, "right": [0] * n_blocks}
    for r in records:
        index = (int(round(r.timestamp)) - 1) // block
        if 0 <= index < n_blocks and r.operandum in counts:
            counts[r.operandum][index] += 1
    per_minute = 60.0 / block
    out: dict[str, dict[str, list[float]]] = {}
    for side, values in counts.items():
        out[side] = {}
        start = 0
        for key, steps in zip(PHASE_KEYS, phase_steps, strict=True):
            n = steps // block
            out[side][key] = [c * per_minute for c in values[start : start + n]]
            start += n
    return out


def resurgence_metrics(rates: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    """session-analyzer ``analyze_resurgence`` on the block rates (left = target)."""
    from dataclasses import asdict

    from session_analyzer.analytics.resurgence import analyze_resurgence

    left, right = rates["left"], rates["right"]
    metrics = analyze_resurgence(
        target_rates_phase1=left["phase1"],
        target_rates_phase2=left["phase2"],
        target_rates_phase3=left["phase3"],
        alt_rates_phase2=right["phase2"],
        alt_rates_phase3=right["phase3"],
        baseline_window=BASELINE_WINDOW,
        test_window=TEST_WINDOW,
    )
    return asdict(metrics)


def reinforcers_per_phase(records: Iterable[ChoiceRecord]) -> dict[str, dict[str, int]]:
    bounds = [sum(PHASE_STEPS[: i + 1]) for i in range(len(PHASE_STEPS))]
    out = {key: {"left": 0, "right": 0} for key in PHASE_KEYS}
    for r in records:
        if not r.is_reinforced:
            continue
        phase = next(i for i, b in enumerate(bounds) if r.timestamp <= b or i == len(bounds) - 1)
        out[PHASE_KEYS[phase]][r.operandum] += 1
    return out


def snapshot(
    args: argparse.Namespace,
    box: ConcurrentBox,
    subject: Any,
    client: Any,
    step: int,
    started: float,
) -> dict[str, Any]:
    return {
        "model": args.model,
        "seed": args.seed,
        "target_vi_s": args.target_vi,
        "alt_vi_s": args.alt_vi or None,
        "cod_s": args.cod,
        "phase_steps": list(PHASE_STEPS),
        "block_steps": BLOCK_STEPS,
        "system_prompt": CONCURRENT_SYSTEM_PROMPT,
        "steps_done": step,
        "actions": [h["r"] for h in subject.history],
        "raw_replies": [h["raw"] for h in subject.history],
        "records": [
            {
                "t": r.timestamp,
                "operandum": r.operandum,
                "sr": r.is_reinforced,
                "cod_suppressed": r.cod_suppressed,
            }
            for r in box.records
        ],
        "invalid_replies": subject.invalid_count,
        "latencies_s": list(client.latencies),
        "wall_s": time.monotonic() - started,
        "api_equivalent_usd": (client.last_result or {}).get("total_cost_usd"),
    }


def run(args: argparse.Namespace) -> Path:
    from ai_operant_box.agent.api_client import ClaudeCLIClient
    from ai_operant_box.agent.llm_agent import StreamingLLMSubject

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"resurgence_{args.model}_seed{args.seed}.json"
    total = sum(PHASE_STEPS)
    alt_vi = args.alt_vi or None
    started = time.monotonic()
    with ClaudeCLIClient(model=args.model, system_prompt=CONCURRENT_SYSTEM_PROMPT) as client:
        subject = StreamingLLMSubject(client, actions=CONCURRENT_ACTIONS)
        schedule = build_schedule(alt_vi, args.target_vi, args.cod, args.seed)
        box = ConcurrentBox(subject, schedule)
        for step in range(1, total + 1):
            box.step(1.0)
            if step % args.checkpoint == 0 or step == total:
                data = snapshot(args, box, subject, client, step, started)
                path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
                print(f"seed {args.seed} step {step}: {data['wall_s']:.0f} s", flush=True)
    rates = block_rates(box.records)
    actions = data["actions"]
    data.update(
        rates=rates,
        reinforcers=reinforcers_per_phase(box.records),
        action_counts={a: actions.count(a) for a in sorted(set(actions))},
        metrics=resurgence_metrics(rates),
        latency_mean_s=statistics.mean(data["latencies_s"]),
    )
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {"seed": args.seed, "metrics": data["metrics"], "reinforcers": data["reinforcers"]},
            indent=2,
        )
    )
    return path


def _block_x(n_blocks: tuple[int, ...]) -> list[float]:
    """Each block's x position: phase index + position within the phase."""
    return [p + (i + 0.5) / n for p, n in enumerate(n_blocks) for i in range(n)]


def plot(directory: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [json.loads(p.read_text()) for p in sorted(directory.glob("resurgence_*_seed*.json"))]
    runs = [r for r in runs if "rates" in r]
    if not runs:
        raise SystemExit(f"no finished runs in {directory}")
    n_blocks = tuple(len(runs[0]["rates"]["left"][k]) for k in PHASE_KEYS)
    x = _block_x(n_blocks)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for side, color, ax in (("left", "#1f5fa8", axes[0]), ("right", "#c05a1a", axes[1])):
        for r in runs:
            ys = [v for k in PHASE_KEYS for v in r["rates"][side][k]]
            ax.plot(x, ys, color=color, lw=0.8, alpha=0.5, marker="o", ms=3)
        med = [
            statistics.median(v)
            for v in zip(
                *[[v for k in PHASE_KEYS for v in r["rates"][side][k]] for r in runs], strict=True
            )
        ]
        ax.plot(x, med, color=color, lw=2, label=f"median of {len(runs)} runs")
        for b in (1, 2):
            ax.axvline(b, color="k", lw=0.6)
        ax.set_xticks([0.5, 1.5, 2.5], ["phase 1", "phase 2", "phase 3"])
        ax.set_ylim(0, 62)
        ax.set_ylabel("responses / min")
        ax.set_title(f"{side} ({'target' if side == 'left' else 'alternative'})")
        ax.legend(fontsize=8)
    fig.suptitle(f"Resurgence, {runs[0]['model']}, alt VI {runs[0]['alt_vi_s']} s")
    fig.tight_layout()
    out = directory / f"resurgence_{runs[0]['model']}.png"
    fig.savefig(out, dpi=120)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-vi", type=float, default=15.0)
    parser.add_argument(
        "--alt-vi", type=float, default=15.0, help="phase-2 alternative VI mean in s; 0 = EXT"
    )
    parser.add_argument("--cod", type=float, default=2.0)
    parser.add_argument("--checkpoint", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/resurgence"))
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args()
    if args.plot is not None:
        print(plot(args.plot))
        return
    run(args)


if __name__ == "__main__":
    main()
