"""Run a Claude model through CRF then extinction (EXT) in one session.

Each replication is a fresh ``claude`` process (independent conversation).
The schedule switches from CRF to EXT without any signal to the subject;
the only change it sees is ``Result: No Reward`` after lever presses.

    .venv/bin/python examples/claude_cli_crf_ext.py --model haiku --crf 50 --ext 100 --reps 3 --out-dir runs/crf_ext

Writes per-replication JSON (actions, raw replies, latencies), a summary
table (JSON + Markdown) and a cumulative record PNG over all steps.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from ai_operant_box.agent.api_client import ClaudeCLIClient
from ai_operant_box.agent.llm_agent import SYSTEM_PROMPT, StreamingLLMSubject
from ai_operant_box.chamber import ContinuousReinforcement, Extinction, OperantBox

TARGET = "Lever Press"


def rate(actions: list[str], start: int, stop: int) -> float:
    window = actions[start:stop]
    return sum(a == TARGET for a in window) / len(window)


def run_once(model: str, crf: int, ext: int) -> dict:
    started = time.monotonic()
    with ClaudeCLIClient(model=model, system_prompt=SYSTEM_PROMPT) as client:
        subject = StreamingLLMSubject(client)
        box = OperantBox(subject=subject, schedule=ContinuousReinforcement())
        for _ in range(crf):
            box.step(1.0)
        box.schedule = Extinction()
        for _ in range(ext):
            box.step(1.0)
        latencies = list(client.latencies)
        cost = (client.last_result or {}).get("total_cost_usd")
    actions = [h["r"] for h in subject.history]
    return {
        "model": model,
        "crf_steps": crf,
        "ext_steps": ext,
        "actions": actions,
        "raw_replies": [h["raw"] for h in subject.history],
        "reinforced": [bool(h.get("sr")) for h in subject.history],
        "invalid_replies": subject.invalid_count,
        "latency_mean_s": statistics.mean(latencies),
        "latencies_s": latencies,
        "wall_s": time.monotonic() - started,
        "api_equivalent_usd": cost,
        "crf_last20_rate": rate(actions, crf - 20, crf),
        "ext_first10_rate": rate(actions, crf, crf + 10),
        "ext_last20_rate": rate(actions, crf + ext - 20, crf + ext),
    }


def plot(results: list[dict], path: Path, crf: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, res in enumerate(results, 1):
        cum, total = [], 0
        for a in res["actions"]:
            total += a == TARGET
            cum.append(total)
        ax.step(range(1, len(cum) + 1), cum, where="post", label=f"rep {i}")
    ax.axvline(crf + 0.5, color="black", linestyle="--", linewidth=1)
    ax.text(crf + 2, 2, "CRF → EXT", fontsize=9)
    ax.set_xlabel("step (1 step = 1 s)")
    ax.set_ylabel("cumulative lever presses")
    ax.set_title(f"CRF {crf} → EXT, {results[0]['model']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--crf", type=int, default=50)
    parser.add_argument("--ext", type=int, default=100)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/crf_ext"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i in range(1, args.reps + 1):
        res = run_once(args.model, args.crf, args.ext)
        (args.out_dir / f"rep{i}_{args.model}.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False)
        )
        results.append(res)
        print(f"rep {i}: done in {res['wall_s']:.0f} s", flush=True)

    keys = ["crf_last20_rate", "ext_first10_rate", "ext_last20_rate",
            "invalid_replies", "latency_mean_s", "api_equivalent_usd"]
    table = [{"rep": i, **{k: r[k] for k in keys}} for i, r in enumerate(results, 1)]
    (args.out_dir / f"summary_{args.model}.json").write_text(json.dumps(table, indent=2))
    lines = ["| rep | CRF last 20 | EXT first 10 | EXT last 20 | invalid | s/step |",
             "|---|---|---|---|---|---|"]
    for row in table:
        lines.append(
            f"| {row['rep']} | {row['crf_last20_rate']:.2f} | {row['ext_first10_rate']:.2f} | "
            f"{row['ext_last20_rate']:.2f} | {row['invalid_replies']} | {row['latency_mean_s']:.2f} |"
        )
    md = "\n".join(lines)
    (args.out_dir / f"summary_{args.model}.md").write_text(md + "\n")
    plot(results, args.out_dir / f"crf_ext_{args.model}.png", args.crf)
    print(md)


if __name__ == "__main__":
    main()
