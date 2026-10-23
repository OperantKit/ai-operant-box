"""Contingent vs yoked (non-contingent) reinforcement with a Claude subject.

Each replication is a master-yoked pair of fresh ``claude`` processes:

1. **contingent** (master): random-interval (RI) schedule. Each 1 s step a
   reinforcer is set up with probability ``1 / --ri``; a set-up reinforcer is
   held until the next lever press, which produces it.
2. **yoked**: a new subject receives a reinforcer at exactly the steps where
   the master was reinforced, whatever it did at that step (the reinforcer is
   independent of its behavior).

Both see the same feedback lines (``Result: Reward`` / ``No Reward``). The
numerical control ``InductionSubject`` (Baum & Grace, 2020) receives the same
yoked reinforcer steps as non-contingent reinforcers and is averaged over
``--induction-sims`` simulated sessions.

    .venv/bin/python examples/claude_cli_yoked.py --model haiku --steps 150 --reps 3 --out-dir runs/yoked

``--fake`` replaces Claude with a seeded random subject to check the
mechanics without using the Max quota.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ai_operant_box.agent.api_client import ClaudeCLIClient
from ai_operant_box.agent.induction_agent import InductionSubject
from ai_operant_box.agent.llm_agent import ACTIONS, SYSTEM_PROMPT, StreamingLLMSubject

TARGET = "Lever Press"
SD = "Light ON"


class RandomInterval:
    """RI schedule on a 1 s step clock with a held set-up (seeded)."""

    def __init__(self, mean_s: float, seed: int) -> None:
        self.p = 1.0 / mean_s
        self.rng = random.Random(seed)
        self.armed = False

    def step(self, action: str) -> bool:
        if not self.armed and self.rng.random() < self.p:
            self.armed = True
        if self.armed and action == TARGET:
            self.armed = False
            return True
        return False


class Yoked:
    """Deliver reinforcers at fixed steps regardless of the response."""

    def __init__(self, sr_steps: list[int]) -> None:
        self.sr_steps = set(sr_steps)
        self.i = 0

    def step(self, action: str) -> bool:
        self.i += 1
        return self.i in self.sr_steps


class FakeClient:
    """Seeded stand-in for ``ClaudeCLIClient`` (mechanics check only)."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.latencies: list[float] = []
        self.last_result: dict | None = None

    def send(self, text: str) -> str:
        self.latencies.append(0.0)
        return self.rng.choice(ACTIONS)

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def run_session(model: str, steps: int, schedule, fake_seed: int | None) -> dict:
    started = time.monotonic()
    client_cm = (FakeClient(fake_seed) if fake_seed is not None
                 else ClaudeCLIClient(model=model, system_prompt=SYSTEM_PROMPT))
    with client_cm as client:
        subject = StreamingLLMSubject(client)
        sr_steps = []
        for i in range(1, steps + 1):
            subject.sense(SD)
            action = subject.act()
            reinforced = schedule.step(action)
            subject.learn("Food Pellet" if reinforced else None)
            if reinforced:
                sr_steps.append(i)
        latencies = list(client.latencies)
        cost = (client.last_result or {}).get("total_cost_usd")
    actions = [h["r"] for h in subject.history]
    return {
        "actions": actions,
        "raw_replies": [h["raw"] for h in subject.history],
        "sr_steps": sr_steps,
        "invalid_replies": subject.invalid_count,
        "latency_mean_s": statistics.mean(latencies) if latencies else None,
        "wall_s": time.monotonic() - started,
        "api_equivalent_usd": cost,
    }


def run_pair(rep: int, args: argparse.Namespace) -> dict:
    fake = args.fake
    master = run_session(args.model, args.steps, RandomInterval(args.ri, seed=rep),
                         fake_seed=1000 + rep if fake else None)
    yoked = run_session(args.model, args.steps, Yoked(master["sr_steps"]),
                        fake_seed=2000 + rep if fake else None)
    assert yoked["sr_steps"] == master["sr_steps"], "yoked schedule drifted"
    pair = {"rep": rep, "model": "fake" if fake else args.model, "ri_s": args.ri,
            "steps": args.steps, "contingent": master, "yoked": yoked}
    (args.out_dir / f"rep{rep}_{pair['model']}.json").write_text(
        json.dumps(pair, indent=2, ensure_ascii=False))
    print(f"rep {rep}: contingent {master['wall_s']:.0f} s, yoked {yoked['wall_s']:.0f} s",
          flush=True)
    return pair


def induction_rates(sr_steps: list[int], steps: int, sims: int, seed: int) -> list[float]:
    """Mean per-step P(lever press) of InductionSubject given yoked NCRs."""
    random.seed(seed)
    srs = set(sr_steps)
    totals = [0] * steps
    for _ in range(sims):
        subject = InductionSubject()
        for i in range(steps):
            subject.update_time(1.0)
            totals[i] += subject.act() == TARGET
            if i + 1 in srs:
                subject.deliver_ncr()
    return [t / sims for t in totals]


def block_rates(actions: list[str], block: int) -> list[float]:
    return [sum(a == TARGET for a in actions[s:s + block]) / len(actions[s:s + block])
            for s in range(0, len(actions), block)]


def rate(actions: list[str], start: int, stop: int) -> float:
    w = actions[start:stop]
    return sum(a == TARGET for a in w) / len(w)


def summarize(pairs: list[dict], args: argparse.Namespace) -> tuple[list[dict], str]:
    rows = []
    n = args.steps
    for p in pairs:
        ind = induction_rates(p["contingent"]["sr_steps"], n, args.induction_sims, p["rep"])
        for group in ("contingent", "yoked"):
            s = p[group]
            acts = s["actions"]
            sr = s["sr_steps"]
            rows.append({
                "rep": p["rep"], "group": group,
                "rate_all": rate(acts, 0, n), "rate_first50": rate(acts, 0, 50),
                "rate_last50": rate(acts, n - 50, n), "reinforcers": len(sr),
                "sr_on_press": sum(acts[i - 1] == TARGET for i in sr),
                "counts": {a: acts.count(a) for a in ACTIONS},
                "invalid": s["invalid_replies"], "s_per_step": s["latency_mean_s"],
                "usd": s["api_equivalent_usd"],
            })
        rows.append({
            "rep": p["rep"], "group": "induction (sim)",
            "rate_all": statistics.mean(ind), "rate_first50": statistics.mean(ind[:50]),
            "rate_last50": statistics.mean(ind[n - 50:]),
            "reinforcers": len(p["contingent"]["sr_steps"]), "sr_on_press": None,
            "counts": None, "invalid": None, "s_per_step": None, "usd": None,
        })
    lines = ["| rep | group | LP rate all | first 50 | last 50 | SR | SR on LP | "
             "LP / G / E / N | invalid | s/step |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c = r["counts"]
        counts = (f"{c['Lever Press']} / {c['Grooming']} / {c['Explore']} / {c['Nothing']}"
                  if c else "—")
        sps = f"{r['s_per_step']:.2f}" if r["s_per_step"] is not None else "—"
        lines.append(
            f"| {r['rep']} | {r['group']} | {r['rate_all']:.2f} | {r['rate_first50']:.2f} | "
            f"{r['rate_last50']:.2f} | {r['reinforcers']} | "
            f"{r['sr_on_press'] if r['sr_on_press'] is not None else '—'} | {counts} | "
            f"{r['invalid'] if r['invalid'] is not None else '—'} | {sps} |")
    return rows, "\n".join(lines)


def plot(pairs: list[dict], args: argparse.Namespace, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    block = args.block
    n = args.steps
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"contingent": "tab:blue", "yoked": "tab:orange", "induction": "tab:gray"}
    for p in pairs:
        for group in ("contingent", "yoked"):
            acts = p[group]["actions"]
            cum, total = [], 0
            for a in acts:
                total += a == TARGET
                cum.append(total)
            axes[0].step(range(1, n + 1), cum, where="post", color=colors[group],
                         alpha=0.8, label=f"{group}" if p["rep"] == 1 else None)
            br = block_rates(acts, block)
            xs = [block * (i + 0.5) for i in range(len(br))]
            axes[1].plot(xs, br, marker="o", color=colors[group], alpha=0.6,
                         label=f"{group}" if p["rep"] == 1 else None)
        ind = induction_rates(p["contingent"]["sr_steps"], n, args.induction_sims, p["rep"])
        cum_ind, tot = [], 0.0
        for v in ind:
            tot += v
            cum_ind.append(tot)
        axes[0].plot(range(1, n + 1), cum_ind, color=colors["induction"], linestyle="--",
                     label="InductionSubject (yoked NCR, mean)" if p["rep"] == 1 else None)
        ib = [statistics.mean(ind[s:s + block]) for s in range(0, n, block)]
        axes[1].plot([block * (i + 0.5) for i in range(len(ib))], ib,
                     color=colors["induction"], linestyle="--",
                     label="InductionSubject" if p["rep"] == 1 else None)
    axes[0].set_xlabel("step (1 step = 1 s)")
    axes[0].set_ylabel("cumulative lever presses")
    axes[0].set_title(f"RI {args.ri:g} s master vs yoked, {pairs[0]['model']}, "
                      f"{len(pairs)} pairs")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel(f"P(Lever Press) per {block}-step block")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--ri", type=float, default=5.0, help="RI mean in steps (s)")
    parser.add_argument("--block", type=int, default=25)
    parser.add_argument("--induction-sims", type=int, default=500)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/yoked"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.reps) as pool:
        pairs = list(pool.map(lambda r: run_pair(r, args), range(1, args.reps + 1)))

    model = pairs[0]["model"]
    rows, md = summarize(pairs, args)
    (args.out_dir / f"summary_{model}.json").write_text(json.dumps(rows, indent=2))
    (args.out_dir / f"summary_{model}.md").write_text(md + "\n")
    plot(pairs, args, args.out_dir / f"yoked_{model}.png")
    print(md)


if __name__ == "__main__":
    main()
