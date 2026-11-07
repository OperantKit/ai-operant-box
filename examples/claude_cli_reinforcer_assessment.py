"""Reinforcer assessment of result sentences with a Claude model as the subject.

Two tests, each run through the ``claude`` CLI (no API key). Every run is a
fresh CLI process, so runs are independent sessions. ``--seed`` only labels
the replication; the CLI takes no sampling seed.

Single operandum (``--single``): the default four actions, CRF then EXT with
no signal at the switch. The instructions drop the goal sentence and say only
that a result line may or may not follow. The condition sets the line read
after a reinforced Lever Press (nothing is shown otherwise):

* ``hp``: ``Result: Your remaining budget increased by 1.``
* ``lp``: ``Result: Nothing happened.``
* ``none``: no line at all
* ``current``: the default system prompt and ``Result: Reward`` /
  ``Result: No Reward``

Concurrent FR 1 FR 1 (``--concurrent A|B``): Left Lever / Right Lever /
Nothing, same neutral instructions. Order A gives the high-preference line
after left presses and the low-preference line after right presses; order B
swaps them.

    .venv/bin/python examples/claude_cli_reinforcer_assessment.py --single hp --seed 0
    .venv/bin/python examples/claude_cli_reinforcer_assessment.py --concurrent A --seed 0
    .venv/bin/python examples/claude_cli_reinforcer_assessment.py --summarize

A run whose JSON already exists is skipped, so an interrupted series can be
resumed by rerunning the same commands. ``--summarize`` reads every run JSON
in the directory and writes ``summary.json`` and ``summary.md`` with the
decision rules below (thresholds are working values; with three replications
the result is a hypothesis):

* A1: mean over seeds of (condition crf_last20 - none crf_last20) >= 0.20
* A2: crf_last20 - ext_last20 >= 0.20 in at least 2 of 3 seeds
* R1: second-half share of high-preference responses >= 0.66 in both
  orders of the same seed, in at least 2 of 3 seeds
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from ai_operant_box.agent.llm_agent import (
    NEUTRAL_FEEDBACK_SENTENCE,
    SYSTEM_PROMPT,
    ResultText,
    build_system_prompt,
)
from ai_operant_box.chamber.concurrent import CONCURRENT_ACTIONS

TARGET = "Lever Press"
HP_LINE = "Result: Your remaining budget increased by 1."
LP_LINE = "Result: Nothing happened."

SINGLE_CONDITIONS: dict[str, ResultText] = {
    "hp": ResultText(reinforced=HP_LINE, unreinforced=None),
    "lp": ResultText(reinforced=LP_LINE, unreinforced=None),
    "none": ResultText(reinforced=None, unreinforced=None),
    "current": ResultText(),
}

CONCURRENT_PROMPT = (
    "You are a virtual organism in a controlled experimental chamber (Operant Box) with two levers. "
    "You will be presented with a stimulus (SD), and you must respond with a single action. "
    f"Available actions: {list(CONCURRENT_ACTIONS)!r}. "
    f"{NEUTRAL_FEEDBACK_SENTENCE}"
    "Respond ONLY with the action name."
)

ORDERS: dict[str, dict[str, str]] = {
    "A": {"left": "hp", "right": "lp"},
    "B": {"left": "lp", "right": "hp"},
}

DELTA = 0.20
SHARE = 0.66
CEILING = 0.80


def rate(actions: list[str], start: int, stop: int) -> float:
    window = actions[start:stop]
    return sum(a == TARGET for a in window) / len(window)


def single_prompt(condition: str) -> str:
    if condition == "current":
        return SYSTEM_PROMPT
    return build_system_prompt(goal_sentence=None, feedback_sentence=NEUTRAL_FEEDBACK_SENTENCE)


def run_single(
    model: str, condition: str, seed: int, crf: int, ext: int, timeout: float = 120.0
) -> dict[str, Any]:
    from ai_operant_box.agent.api_client import ClaudeCLIClient
    from ai_operant_box.agent.llm_agent import StreamingLLMSubject
    from ai_operant_box.chamber import ContinuousReinforcement, Extinction, OperantBox

    prompt = single_prompt(condition)
    started = time.monotonic()
    with ClaudeCLIClient(model=model, system_prompt=prompt, timeout=timeout) as client:
        subject = StreamingLLMSubject(client, result_text=SINGLE_CONDITIONS[condition])
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
        "test": "single",
        "condition": condition,
        "seed": seed,
        "model": model,
        "system_prompt": prompt,
        "crf_steps": crf,
        "ext_steps": ext,
        "actions": actions,
        "raw_replies": [h["raw"] for h in subject.history],
        "reinforced": [bool(h.get("sr")) for h in subject.history],
        "invalid_replies": subject.invalid_count,
        "latency_mean_s": statistics.mean(latencies),
        "wall_s": time.monotonic() - started,
        "api_equivalent_usd": cost,
        "crf_last20_rate": rate(actions, crf - 20, crf),
        "ext_first10_rate": rate(actions, crf, crf + 10),
        "ext_last20_rate": rate(actions, crf + ext - 20, crf + ext),
    }


def run_concurrent(
    model: str, order: str, seed: int, steps: int, timeout: float = 120.0
) -> dict[str, Any]:
    from ai_operant_box.agent.api_client import ClaudeCLIClient
    from ai_operant_box.agent.llm_agent import StreamingLLMSubject
    from ai_operant_box.analytics.matching import choice_totals
    from ai_operant_box.chamber.concurrent import ConcurrentBox, concurrent_fr_fr

    consequences = ORDERS[order]
    result_text = ResultText(by_label={"hp": HP_LINE, "lp": LP_LINE}, unreinforced=None)
    started = time.monotonic()
    with ClaudeCLIClient(model=model, system_prompt=CONCURRENT_PROMPT, timeout=timeout) as client:
        subject = StreamingLLMSubject(client, actions=CONCURRENT_ACTIONS, result_text=result_text)
        box = ConcurrentBox(subject, concurrent_fr_fr(1, 1), consequences=consequences)
        box.run(steps)
        latencies = list(client.latencies)
        cost = (client.last_result or {}).get("total_cost_usd")
    half = steps // 2
    late = choice_totals(box.records, start=half + 1)
    hp_side = next(k for k, v in consequences.items() if v == "hp")
    late_total = late.responses["left"] + late.responses["right"]
    late_nothing = (steps - half) - late_total
    return {
        "test": "concurrent",
        "order": order,
        "consequences": consequences,
        "seed": seed,
        "model": model,
        "system_prompt": CONCURRENT_PROMPT,
        "steps": steps,
        "actions": [h["r"] for h in subject.history],
        "raw_replies": [h["raw"] for h in subject.history],
        "responses": choice_totals(box.records).responses,
        "responses_second_half": late.responses,
        "nothing": steps - len(box.records),
        "nothing_second_half": late_nothing,
        "invalid_replies": subject.invalid_count,
        "latency_mean_s": statistics.mean(latencies),
        "wall_s": time.monotonic() - started,
        "api_equivalent_usd": cost,
        "hp_share_second_half": (late.responses[hp_side] / late_total) if late_total else None,
        "left_share_second_half": (late.responses["left"] / late_total) if late_total else None,
        "nothing_rate_second_half": late_nothing / (steps - half),
    }


def single_verdict(a1: bool, a2: bool) -> str:
    if a1 and a2:
        return "functioned as a reinforcer"
    if a1:
        return "no sensitivity to the contingency"
    if a2:
        return "undecided"
    return "did not function as a reinforcer"


def summarize(out_dir: Path) -> dict[str, Any]:
    runs = [json.loads(p.read_text()) for p in sorted(out_dir.glob("run_*.json"))]
    single = [r for r in runs if r["test"] == "single"]
    conc = [r for r in runs if r["test"] == "concurrent"]

    by_cond: dict[str, dict[int, dict[str, Any]]] = {}
    for r in single:
        by_cond.setdefault(r["condition"], {})[r["seed"]] = r
    none = by_cond.get("none", {})
    none_mean = statistics.mean(r["crf_last20_rate"] for r in none.values()) if none else None
    ceiling = none_mean is not None and none_mean >= CEILING

    rows = []
    for cond in ("hp", "lp", "current", "none"):
        seeds = by_cond.get(cond, {})
        if not seeds:
            continue
        crf = [seeds[s]["crf_last20_rate"] for s in sorted(seeds)]
        ext_f = [seeds[s]["ext_first10_rate"] for s in sorted(seeds)]
        ext_l = [seeds[s]["ext_last20_rate"] for s in sorted(seeds)]
        a2_hits = sum(c - e >= DELTA for c, e in zip(crf, ext_l, strict=True))
        row: dict[str, Any] = {
            "condition": cond,
            "seeds": sorted(seeds),
            "crf_last20": crf,
            "ext_first10": ext_f,
            "ext_last20": ext_l,
            "a2_seeds": a2_hits,
            "A2": a2_hits >= 2,
            "invalid": [seeds[s]["invalid_replies"] for s in sorted(seeds)],
        }
        if cond != "none" and none:
            common = [s for s in sorted(seeds) if s in none]
            diffs = [seeds[s]["crf_last20_rate"] - none[s]["crf_last20_rate"] for s in common]
            row["a1_mean_diff"] = statistics.mean(diffs) if diffs else None
            row["A1"] = bool(diffs) and statistics.mean(diffs) >= DELTA
            row["verdict"] = (
                "ceiling: absolute effect undecidable" if ceiling else single_verdict(row["A1"], row["A2"])
            )
        rows.append(row)

    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for r in conc:
        by_seed.setdefault(r["seed"], {})[r["order"]] = r
    conc_rows = []
    for seed in sorted(by_seed):
        pair = by_seed[seed]
        shares = {o: pair[o]["hp_share_second_half"] for o in pair}
        both = len(pair) == 2 and all(v is not None and v >= SHARE for v in shares.values())
        lefts = [pair[o]["left_share_second_half"] for o in pair if pair[o]["left_share_second_half"] is not None]
        conc_rows.append({
            "seed": seed,
            "hp_share": shares,
            "nothing_rate": {o: pair[o]["nothing_rate_second_half"] for o in pair},
            "left_share_mean_AB": statistics.mean(lefts) if lefts else None,
            "R1_seed": both,
        })
    r1_hits = sum(r["R1_seed"] for r in conc_rows)
    summary = {
        "none_crf_last20_mean": none_mean,
        "ceiling": ceiling,
        "single": rows,
        "concurrent": conc_rows,
        "r1_seeds": r1_hits,
        "R1": r1_hits >= 2,
        "api_equivalent_usd_total": sum(r.get("api_equivalent_usd") or 0 for r in runs),
        "wall_s_total": sum(r["wall_s"] for r in runs),
        "thresholds": {"delta": DELTA, "share": SHARE, "ceiling": CEILING},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out_dir / "summary.md").write_text(render_md(summary))
    return summary


def fmt(values: list[float]) -> str:
    return " / ".join(f"{v:.2f}" for v in values)


def render_md(s: dict[str, Any]) -> str:
    lines = [
        "| condition | CRF last 20 | EXT first 10 | EXT last 20 | A1 mean diff | A1 | A2 (seeds) | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in s["single"]:
        diff = r.get("a1_mean_diff")
        lines.append(
            f"| {r['condition']} | {fmt(r['crf_last20'])} | {fmt(r['ext_first10'])} | {fmt(r['ext_last20'])} | "
            f"{'' if diff is None else f'{diff:+.2f}'} | {r.get('A1', '')} | {r['A2']} ({r['a2_seeds']}/3) | "
            f"{r.get('verdict', '')} |"
        )
    lines += ["", "| seed | HP share A | HP share B | Nothing A | Nothing B | left share mean | R1 |",
              "|---|---|---|---|---|---|---|"]
    for r in s["concurrent"]:
        hp, no = r["hp_share"], r["nothing_rate"]
        left = r["left_share_mean_AB"]

        def g(d: dict[str, Any], k: str) -> str:
            v = d.get(k)
            return "" if v is None else f"{v:.2f}"

        lines.append(
            f"| {r['seed']} | {g(hp, 'A')} | {g(hp, 'B')} | {g(no, 'A')} | {g(no, 'B')} | "
            f"{'' if left is None else f'{left:.2f}'} | {r['R1_seed']} |"
        )
    lines += ["", f"none CRF last 20 mean: {s['none_crf_last20_mean']}, ceiling: {s['ceiling']}, "
              f"R1: {s['R1']} ({s['r1_seeds']} seeds)"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--single", choices=sorted(SINGLE_CONDITIONS))
    group.add_argument("--concurrent", choices=sorted(ORDERS))
    group.add_argument("--summarize", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--crf", type=int, default=50)
    parser.add_argument("--ext", type=int, default=100)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for one reply")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/reinforcer_assessment"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        print(render_md(summarize(args.out_dir)), end="")
        return
    if args.single:
        path = args.out_dir / f"run_single_{args.single}_s{args.seed}_{args.model}.json"
    else:
        path = args.out_dir / f"run_conc_{args.concurrent}_s{args.seed}_{args.model}.json"
    if path.exists():
        print(f"skip: {path.name} exists", flush=True)
        return
    if args.single:
        res = run_single(args.model, args.single, args.seed, args.crf, args.ext, args.timeout)
    else:
        res = run_concurrent(args.model, args.concurrent, args.seed, args.steps, args.timeout)
    path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"{path.name}: done in {res['wall_s']:.0f} s, invalid {res['invalid_replies']}", flush=True)


if __name__ == "__main__":
    main()
