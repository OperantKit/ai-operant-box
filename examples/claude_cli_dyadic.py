"""Two Claude subjects on a leader-follower interdependent schedule.

A leader and a follower (two separate ``claude -p`` processes) act once per
1 s step in :class:`LeaderFollowerBox`. A lever press by the follower within
``--window`` s after the leader's most recent press reinforces both
(``InterdependentSchedule``; a press at the same step counts, IRT 0).
Neither subject is told about the partner: both get the usual chamber prompt
and see only ``Stimulus: Light ON`` and their own ``Result: Reward / No Reward``.

The numerical control is a pair of ``BernoulliAgent`` subjects whose press
probabilities are the LLM leader's and follower's overall press rates
(contingency-insensitive, rate-matched), averaged over ``--control-sims``
simulated sessions on the same box.

    .venv/bin/python examples/claude_cli_dyadic.py --model haiku --steps 300 --out-dir runs/dyadic

``--fake`` replaces Claude with seeded random subjects to check the mechanics
without using the Max quota.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ai_operant_box.agent.api_client import ClaudeCLIClient
from ai_operant_box.agent.base import Subject
from ai_operant_box.agent.bernoulli_agent import BernoulliAgent
from ai_operant_box.agent.llm_agent import ACTIONS, SYSTEM_PROMPT, StreamingLLMSubject
from ai_operant_box.chamber.interdependent import InterdependentSchedule, LeaderFollowerBox

TARGET = "Lever Press"
SD = "Light ON"


class PressOnly(Subject):
    """Expose an LLM subject to the box: ``Lever Press`` is a response, else ``None``."""

    def __init__(self, inner: StreamingLLMSubject) -> None:
        self.inner = inner

    def sense(self, discriminative_stimulus: Any) -> None:
        self.inner.sense(discriminative_stimulus)

    def act(self) -> str | None:
        return TARGET if self.inner.act() == TARGET else None

    def learn(self, reinforcer: Any | None) -> None:
        self.inner.learn(reinforcer)

    @property
    def history(self) -> list[dict]:
        return self.inner.history


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


def make_client(model: str, fake_seed: int | None):
    if fake_seed is not None:
        return FakeClient(fake_seed)
    return ClaudeCLIClient(model=model, system_prompt=SYSTEM_PROMPT)


def run_llm_dyad(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    with make_client(args.model, 1 if args.fake else None) as lc, \
            make_client(args.model, 2 if args.fake else None) as fc:
        leader = StreamingLLMSubject(lc)
        follower = StreamingLLMSubject(fc)
        box = LeaderFollowerBox(
            PressOnly(leader), PressOnly(follower),
            InterdependentSchedule(window=args.window), step_duration=1.0,
            leader_sd=SD, follower_sd=SD,
        )
        # Each step sends one turn per subject; run them concurrently so a
        # step costs one round trip instead of two.
        pool = ThreadPoolExecutor(max_workers=2)
        orig_l, orig_f = box.leader.act, box.follower.act
        pending: dict[str, Any] = {}

        def leader_act() -> str | None:
            pending["f"] = pool.submit(orig_f)
            return orig_l()

        def follower_act() -> str | None:
            return pending.pop("f").result()

        box.leader.act = leader_act  # type: ignore[method-assign]
        box.follower.act = follower_act  # type: ignore[method-assign]
        events = box.run(args.steps)
        pool.shutdown()
        lat = {"leader": list(lc.latencies), "follower": list(fc.latencies)}
        cost = [(c.last_result or {}).get("total_cost_usd") for c in (lc, fc)]
    return {
        "events": [e.__dict__ for e in events],
        "leader_actions": [h["r"] for h in leader.history],
        "follower_actions": [h["r"] for h in follower.history],
        "leader_raw": [h["raw"] for h in leader.history],
        "follower_raw": [h["raw"] for h in follower.history],
        "invalid": {"leader": leader.invalid_count, "follower": follower.invalid_count},
        "latency_mean_s": {k: statistics.mean(v) if v else None for k, v in lat.items()},
        "wall_s": time.monotonic() - started,
        "api_equivalent_usd": cost,
    }


def run_control(p_leader: float, p_follower: float, steps: int, window: float,
                seed: int) -> list[dict]:
    rng = random.Random(seed)
    box = LeaderFollowerBox(
        BernoulliAgent(p_leader, rng), BernoulliAgent(p_follower, rng),
        InterdependentSchedule(window=window), step_duration=1.0,
    )
    return [e.__dict__ for e in box.run(steps)]


def block_measures(events: list[dict], block: int) -> list[dict]:
    rows = []
    last_l = last_f = None
    own_irts: dict[str, list[float]] = {"leader": [], "follower": []}
    for start in range(0, len(events), block):
        w = events[start:start + block]
        irts_l: list[float] = []
        irts_f: list[float] = []
        for e in w:
            t = e["time_step"]
            if e["leader_responded"]:
                if last_l is not None:
                    irts_l.append(t - last_l)
                last_l = t
            if e["follower_responded"]:
                if last_f is not None:
                    irts_f.append(t - last_f)
                last_f = t
        own_irts["leader"] += irts_l
        own_irts["follower"] += irts_f
        pair_irts = [e["leader_follower_irt"] for e in w if e["reinforced"]]
        n = len(w)
        rows.append({
            "start": start, "n": n,
            "sr_rate": sum(e["reinforced"] for e in w) / n,
            "sync": sum(e["leader_responded"] and e["follower_responded"] for e in w) / n,
            "leader_rate": sum(e["leader_responded"] for e in w) / n,
            "follower_rate": sum(e["follower_responded"] for e in w) / n,
            "pair_irt_mean": statistics.mean(pair_irts) if pair_irts else None,
            "leader_irt_mean": statistics.mean(irts_l) if irts_l else None,
            "follower_irt_mean": statistics.mean(irts_f) if irts_f else None,
        })
    return rows


def overall(events: list[dict]) -> dict:
    n = len(events)
    pl = sum(e["leader_responded"] for e in events) / n
    pf = sum(e["follower_responded"] for e in events) / n
    pair = [e["leader_follower_irt"] for e in events if e["reinforced"]]
    return {
        "sr_rate": sum(e["reinforced"] for e in events) / n,
        "reinforcers": sum(e["reinforced"] for e in events),
        "sync": sum(e["leader_responded"] and e["follower_responded"] for e in events) / n,
        "sync_expected_independent": pl * pf,
        "leader_rate": pl, "follower_rate": pf,
        "pair_irt_mean": statistics.mean(pair) if pair else None,
        "pair_irt_counts": {str(k): pair.count(k) for k in sorted(set(pair))},
    }


def fmt(v: float | None, nd: int = 2) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def plot(llm_blocks: list[dict], ctrl_blocks: list[dict], args: argparse.Namespace,
         path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [b["start"] + b["n"] / 2 for b in llm_blocks]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [
        ("sr_rate", "joint reinforcers / step"),
        ("sync", "both pressed / step (synchrony)"),
        ("pair_irt_mean", "mean leader→follower IRT of reinforced pairs (s)"),
    ]
    for ax, (key, label) in zip(axes, panels, strict=True):
        ax.plot(xs, [b[key] for b in llm_blocks], marker="o", color="tab:blue",
                label=f"{args.model} dyad")
        ax.plot(xs, [b[key] for b in ctrl_blocks], linestyle="--", color="tab:gray",
                label="rate-matched Bernoulli pair (mean)")
        ax.set_xlabel("step (1 step = 1 s)")
        ax.set_ylabel(label)
        if key != "pair_irt_mean":
            ax.set_ylim(-0.05, 1.05)
        else:
            ax.set_ylim(-0.2, args.window + 0.2)
        ax.legend(fontsize=8)
    axes[1].plot(xs, [b["leader_rate"] for b in llm_blocks], color="tab:green", alpha=0.6,
                 label="leader press rate")
    axes[1].plot(xs, [b["follower_rate"] for b in llm_blocks], color="tab:orange",
                 alpha=0.6, label="follower press rate")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Leader-follower interdependent schedule, window {args.window:g} s, "
                 f"{args.steps} steps, {args.block}-step blocks")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def average_blocks(runs: list[list[dict]]) -> list[dict]:
    out = []
    for i, b in enumerate(runs[0]):
        row = {"start": b["start"], "n": b["n"]}
        for k in ("sr_rate", "sync", "leader_rate", "follower_rate", "pair_irt_mean",
                  "leader_irt_mean", "follower_irt_mean"):
            vals = [r[i][k] for r in runs if r[i][k] is not None]
            row[k] = statistics.mean(vals) if vals else None
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--window", type=float, default=5.0)
    parser.add_argument("--block", type=int, default=25)
    parser.add_argument("--control-sims", type=int, default=500)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/dyadic"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = "fake" if args.fake else args.model

    dyad = run_llm_dyad(args)
    (args.out_dir / f"dyad_{name}.json").write_text(json.dumps(dyad, indent=2, ensure_ascii=False))
    llm_all = overall(dyad["events"])
    llm_blocks = block_measures(dyad["events"], args.block)

    ctrl_runs = [run_control(llm_all["leader_rate"], llm_all["follower_rate"], args.steps,
                             args.window, seed) for seed in range(args.control_sims)]
    ctrl_blocks = average_blocks([block_measures(r, args.block) for r in ctrl_runs])
    ctrl_all = {k: statistics.mean(overall(r)[k] for r in ctrl_runs)
                for k in ("sr_rate", "sync", "leader_rate", "follower_rate")}
    ctrl_sr = sorted(overall(r)["reinforcers"] for r in ctrl_runs)
    ctrl_all["reinforcers_mean"] = statistics.mean(ctrl_sr)
    ctrl_all["reinforcers_p_ge_llm"] = (
        sum(v >= llm_all["reinforcers"] for v in ctrl_sr) / len(ctrl_sr))

    lines = [
        "| block start | SR/step | sync | leader rate | follower rate | pair IRT | "
        "control SR/step | control sync |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for b, c in zip(llm_blocks, ctrl_blocks, strict=True):
        lines.append(
            f"| {b['start']} | {fmt(b['sr_rate'])} | {fmt(b['sync'])} | "
            f"{fmt(b['leader_rate'])} | {fmt(b['follower_rate'])} | "
            f"{fmt(b['pair_irt_mean'])} | {fmt(c['sr_rate'])} | {fmt(c['sync'])} |")
    md = "\n".join(lines)
    summary = {"llm": llm_all, "control": ctrl_all, "llm_blocks": llm_blocks,
               "control_blocks": ctrl_blocks, "invalid": dyad["invalid"],
               "latency_mean_s": dyad["latency_mean_s"], "wall_s": dyad["wall_s"],
               "api_equivalent_usd": dyad["api_equivalent_usd"],
               "action_counts": {
                   role: {a: dyad[f"{role}_actions"].count(a) for a in ACTIONS}
                   for role in ("leader", "follower")}}
    (args.out_dir / f"summary_{name}.json").write_text(json.dumps(summary, indent=2))
    (args.out_dir / f"summary_{name}.md").write_text(md + "\n")
    plot(llm_blocks, ctrl_blocks, args, args.out_dir / f"dyadic_{name}.png")
    print(md)
    print(json.dumps({k: summary[k] for k in ("llm", "control", "invalid", "latency_mean_s",
                                              "wall_s", "action_counts")}, indent=1))


if __name__ == "__main__":
    main()
