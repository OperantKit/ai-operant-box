"""Read P(Lever Press | time since reinforcer) on FI from a local model, without sampling.

At every step the probabilities of the four actions are computed from the
model's logits (see :mod:`ai_operant_box.agent.local_logprob`). The emitted
action, needed to run the schedule, is a seeded draw from that distribution
(``--policy sample``) or the most probable action (``--policy greedy``).

    .venv/bin/python -m pip install -e ".[local,viz]"
    .venv/bin/python examples/local_logprob_fi.py \\
        --model path/to/qwen2.5-1.5b-instruct-q4_k_m.gguf \\
        --interval 30 --steps 600 --out-dir runs/fi30

Writes ``<stem>.png`` (P per step over the session, and mean P by elapsed
time since the previous reinforcer) and ``<stem>.json`` (every step).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from ai_operant_box.agent.local_logprob import (
    LlamaCppBackend,
    LocalLogprobClient,
    LocalLogprobSubject,
    chatml_prompt,
    p_response_by_time_since_reinforcer,
)
from ai_operant_box.chamber.environment import OperantBox
from ai_operant_box.chamber.schedule import FixedInterval

TARGET = "Lever Press"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True, help="GGUF file")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--policy", choices=("sample", "greedy"), default="sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-ctx", type=int, default=32768)
    parser.add_argument(
        "--history-window",
        type=int,
        default=None,
        help="render only the last N (user, assistant) pairs (default: all)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/fi_local"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    backend = LlamaCppBackend(str(args.model), n_ctx=args.n_ctx)
    client = LocalLogprobClient(
        backend, temperature=args.temperature, history_window=args.history_window
    )
    subject = LocalLogprobSubject(client, policy=args.policy, seed=args.seed)
    box = OperantBox(subject=subject, schedule=FixedInterval(args.interval))

    step_s: list[float] = []
    started = time.monotonic()
    for _ in range(args.steps):
        t0 = time.monotonic()
        box.step(1.0)
        step_s.append(time.monotonic() - t0)
    wall = time.monotonic() - started

    times = [float(i + 1) for i in range(args.steps)]
    p_lp = [h["p"][TARGET] for h in subject.history]
    reinforced = [h.get("sr") is not None for h in subject.history]
    curve = p_response_by_time_since_reinforcer(times, p_lp, reinforced)

    stem = (
        f"fi{int(args.interval)}_{args.model.stem}_T{args.temperature:g}_{args.policy}{args.seed}"
    )
    if args.history_window is not None:
        stem += f"_w{args.history_window}"
    png = args.out_dir / f"{stem}.png"
    _plot(png, times, p_lp, reinforced, curve, args)

    actions = [h["r"] for h in subject.history]
    summary = {
        "model": args.model.name,
        "schedule": f"FI {args.interval:g} s (1 step = 1 s)",
        "steps": args.steps,
        "temperature": args.temperature,
        "policy": args.policy,
        "seed": args.seed,
        "history_window": args.history_window,
        "reinforcers": sum(reinforced),
        "action_counts": {a: actions.count(a) for a in client.actions},
        "p_lever_press_mean": statistics.mean(p_lp),
        "p_lever_press_min": min(p_lp),
        "p_lever_press_max": max(p_lp),
        "p_by_elapsed_s": {str(k): v for k, v in curve.items()},
        "step_s_mean": statistics.mean(step_s),
        "step_s_max": max(step_s),
        "wall_s": wall,
        "prompt_tokens_final": len(
            backend.tokenize(chatml_prompt(client.system_prompt, client.messages))
        ),
        "png": str(png),
        "steps_detail": [
            {
                "t": t,
                "action": h["r"],
                "reinforced": r,
                "p": h["p"],
                "sequence_logprob": h["sequence_logprob"],
                "first_token_logprob": h["first_token_logprob"],
            }
            for t, h, r in zip(times, subject.history, reinforced, strict=True)
        ],
    }
    (args.out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("steps_detail", "p_by_elapsed_s")},
            indent=2,
        )
    )


def _plot(png: Path, times, p_lp, reinforced, curve, args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    ax1.plot(times, p_lp, lw=0.8, color="black")
    sr_times = [t for t, r in zip(times, reinforced, strict=True) if r]
    ax1.plot(
        sr_times, [1.04] * len(sr_times), "v", color="tab:red", clip_on=False, label="reinforcer"
    )
    ax1.set_xlabel("session time (s)")
    ax1.set_ylabel("P(Lever Press)")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="center right")
    ax1.set_title(
        f"{args.model.stem}, FI {args.interval:g} s, T={args.temperature:g}, {args.policy}",
        pad=14,
    )

    origin = 0.0
    xs, ys = [], []
    for t, p, r in zip(times, p_lp, reinforced, strict=True):
        xs.append(t - origin)
        ys.append(p)
        if r:
            origin = t
    ax2.scatter(xs, ys, s=6, color="0.6", label="each step")
    ax2.plot(list(curve), list(curve.values()), color="black", lw=1.5, label="mean")
    ax2.set_xlabel("time since previous reinforcer (s)")
    ax2.set_ylabel("P(Lever Press | t)")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="best")
    fig.savefig(png, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
