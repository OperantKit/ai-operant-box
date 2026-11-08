"""Establishing-operation sentences and the result sentence C6, on a Claude subject.

The result sentence is C6, ``Result: Your remaining budget increased by 1.``
An operation sentence is put at the head of the first message of a session
only (``StreamingLLMSubject(preamble=...)``); nothing else changes:

* ``deprivation``: ``Note: Your remaining budget is very low.``
* ``satiation``: ``Note: You have just received a large increase to your budget.``

The control is the same procedure with no operation sentence, which is the
existing preference and CRF/EXT runs (C6 is the high-preference line there),
so it is read from their directories rather than run again.

Two measures per operation and seed, each a fresh ``claude`` CLI process
(``--seed`` labels the replication and shuffles the preference pairs):

* ``--preference OP``: the 56-trial paired-stimulus preference assessment of
  ``ai_operant_box.chamber.preference``, with the operation sentence before
  the first trial.
* ``--crf-ext OP``: CRF 50 then EXT 100 on Lever Press with the neutral
  instructions, C6 after a reinforced press and no line otherwise.

    .venv/bin/python examples/claude_cli_establishing_operation.py --preference deprivation --seed 0
    .venv/bin/python examples/claude_cli_establishing_operation.py --crf-ext satiation --seed 0
    .venv/bin/python examples/claude_cli_establishing_operation.py --summarize \\
        --control-preference runs/preference --control-crf-ext runs/reinforcer_assessment

A run whose JSON already exists is skipped. ``--summarize`` writes
``summary.json`` and ``summary.md`` with these decision rules (working values;
each needs at least 2 of 3 seeds, and with three replications a result is a
hypothesis). The control value is the mean over the control seeds:

* E1: C6 selection rate - control mean is >= +0.15 under deprivation and
  <= -0.15 under satiation
* E2: crf_last20 - ext_last20 is at least 0.20 above the control under
  deprivation and at least 0.20 below it under satiation
* E3: ext_last20 is at least 0.20 above the control under deprivation

A control value at the ceiling (1.00) leaves no room for the upward changes
of E2 and E3; the table marks it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ai_operant_box.agent.llm_agent import (
    NEUTRAL_FEEDBACK_SENTENCE,
    ResultText,
    build_system_prompt,
)
from ai_operant_box.chamber.preference import (
    PREFERENCE_SYSTEM_PROMPT,
    run_preference,
    summarize_seed,
)

TARGET = "Lever Press"
SENTENCE_KEY = "C6"
RESULT_LINE = "Result: Your remaining budget increased by 1."
CONTROL_CONDITION = "hp"

OPERATIONS: dict[str, str] = {
    "deprivation": "Note: Your remaining budget is very low.",
    "satiation": "Note: You have just received a large increase to your budget.",
}

E1_DELTA = 0.15
E2_DELTA = 0.20
E3_DELTA = 0.20
SEEDS_NEEDED = 2
CEILING = 1.0


class PreambleClient:
    """Puts ``preamble`` at the head of the first message sent through ``client``."""

    def __init__(self, client: Any, preamble: str | None) -> None:
        self.client = client
        self.preamble = preamble
        self._sent = False

    def send(self, text: str) -> str:
        if not self._sent and self.preamble is not None:
            text = f"{self.preamble}\n{text}"
        self._sent = True
        return self.client.send(text)


def rate(actions: Sequence[str], start: int, stop: int) -> float:
    window = actions[start:stop]
    return sum(a == TARGET for a in window) / len(window)


def run_preference_op(model: str, op: str, seed: int, timeout: float) -> dict[str, Any]:
    from ai_operant_box.agent.api_client import ClaudeCLIClient

    started = time.monotonic()
    with ClaudeCLIClient(model=model, system_prompt=PREFERENCE_SYSTEM_PROMPT, timeout=timeout) as client:
        records = run_preference(PreambleClient(client, OPERATIONS[op]), seed)
        latencies = list(client.latencies)
        cost = (client.last_result or {}).get("total_cost_usd")
    seed_summary = summarize_seed(records)
    return {
        "test": "preference",
        "operation": op,
        "preamble": OPERATIONS[op],
        "seed": seed,
        "model": model,
        "system_prompt": PREFERENCE_SYSTEM_PROMPT,
        "records": records,
        "rates": seed_summary["rates"],
        "ranks": seed_summary["ranks"],
        "option_1_rate": seed_summary["option_1_rate"],
        "invalid": seed_summary["invalid"],
        "c6_rate": seed_summary["rates"][SENTENCE_KEY],
        "latency_mean_s": statistics.mean(latencies),
        "wall_s": time.monotonic() - started,
        "api_equivalent_usd": cost,
    }


def run_crf_ext_op(
    model: str, op: str, seed: int, crf: int, ext: int, timeout: float
) -> dict[str, Any]:
    from ai_operant_box.agent.api_client import ClaudeCLIClient
    from ai_operant_box.agent.llm_agent import StreamingLLMSubject
    from ai_operant_box.chamber import ContinuousReinforcement, Extinction, OperantBox

    prompt = build_system_prompt(goal_sentence=None, feedback_sentence=NEUTRAL_FEEDBACK_SENTENCE)
    started = time.monotonic()
    with ClaudeCLIClient(model=model, system_prompt=prompt, timeout=timeout) as client:
        subject = StreamingLLMSubject(
            client,
            result_text=ResultText(reinforced=RESULT_LINE, unreinforced=None),
            preamble=OPERATIONS[op],
        )
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
        "test": "crf_ext",
        "operation": op,
        "preamble": OPERATIONS[op],
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


def control_preference(control_dir: Path) -> dict[int, float]:
    """C6 selection rate per seed from finished seed files of the preference example."""
    out: dict[int, float] = {}
    for p in sorted(control_dir.glob("seed*.json")):
        run = json.loads(p.read_text())
        if run.get("status") == "finished":
            out[run["seed"]] = summarize_seed(run["records"])["rates"][SENTENCE_KEY]
    return out


def control_crf_ext(control_dir: Path) -> dict[int, dict[str, float]]:
    """CRF/EXT rates per seed of the C6 (``hp``) runs of the reinforcer-assessment example."""
    out: dict[int, dict[str, float]] = {}
    for p in sorted(control_dir.glob(f"run_single_{CONTROL_CONDITION}_s*.json")):
        run = json.loads(p.read_text())
        out[run["seed"]] = {k: run[k] for k in ("crf_last20_rate", "ext_first10_rate", "ext_last20_rate")}
    return out


def _hits(flags: Sequence[bool]) -> dict[str, Any]:
    n = sum(flags)
    return {"seeds_met": n, "seeds": len(flags), "met": n >= SEEDS_NEEDED}


def judge_preference(op: str, rates: Mapping[int, float], control: Mapping[int, float]) -> dict[str, Any]:
    base = statistics.mean(control.values())
    diffs = {s: rates[s] - base for s in sorted(rates)}
    if op == "deprivation":
        flags = [d >= E1_DELTA for d in diffs.values()]
    else:
        flags = [d <= -E1_DELTA for d in diffs.values()]
    return {"control_mean": base, "rates": dict(rates), "diffs": diffs, "E1": _hits(flags)}


def judge_crf_ext(
    op: str, runs: Mapping[int, Mapping[str, float]], control: Mapping[int, Mapping[str, float]]
) -> dict[str, Any]:
    gap = {s: r["crf_last20_rate"] - r["ext_last20_rate"] for s, r in runs.items()}
    control_gap = statistics.mean(c["crf_last20_rate"] - c["ext_last20_rate"] for c in control.values())
    control_ext = statistics.mean(c["ext_last20_rate"] for c in control.values())
    if op == "deprivation":
        e2 = [g - control_gap >= E2_DELTA for g in gap.values()]
    else:
        e2 = [g - control_gap <= -E2_DELTA for g in gap.values()]
    row: dict[str, Any] = {
        "crf_last20": {s: runs[s]["crf_last20_rate"] for s in sorted(runs)},
        "ext_first10": {s: runs[s]["ext_first10_rate"] for s in sorted(runs)},
        "ext_last20": {s: runs[s]["ext_last20_rate"] for s in sorted(runs)},
        "gap": dict(sorted(gap.items())),
        "control_gap_mean": control_gap,
        "control_ext_last20_mean": control_ext,
        "control_at_ceiling": control_ext >= CEILING,
        "E2": _hits(e2),
    }
    if op == "deprivation":
        row["E3"] = _hits([runs[s]["ext_last20_rate"] - control_ext >= E3_DELTA for s in sorted(runs)])
    return row


def summarize(out_dir: Path, control_pref_dir: Path, control_crf_dir: Path) -> dict[str, Any]:
    runs = [json.loads(p.read_text()) for p in sorted(out_dir.glob("run_*.json"))]
    pref_control = control_preference(control_pref_dir)
    crf_control = control_crf_ext(control_crf_dir)
    if not pref_control or not crf_control:
        raise SystemExit("no control runs found in the given directories")
    summary: dict[str, Any] = {
        "sentence": RESULT_LINE,
        "operations": OPERATIONS,
        "control_preference": pref_control,
        "control_crf_ext": crf_control,
        "preference": {},
        "crf_ext": {},
        "thresholds": {"E1": E1_DELTA, "E2": E2_DELTA, "E3": E3_DELTA, "seeds_needed": SEEDS_NEEDED},
        "api_equivalent_usd_total": sum(r.get("api_equivalent_usd") or 0 for r in runs),
        "wall_s_total": sum(r["wall_s"] for r in runs),
    }
    for op in OPERATIONS:
        pref = {r["seed"]: r["c6_rate"] for r in runs if r["test"] == "preference" and r["operation"] == op}
        if pref:
            summary["preference"][op] = judge_preference(op, pref, pref_control)
            summary["preference"][op]["invalid"] = {
                r["seed"]: r["invalid"] for r in runs if r["test"] == "preference" and r["operation"] == op
            }
        crf = {r["seed"]: r for r in runs if r["test"] == "crf_ext" and r["operation"] == op}
        if crf:
            summary["crf_ext"][op] = judge_crf_ext(op, crf, crf_control)
            summary["crf_ext"][op]["invalid"] = {s: r["invalid_replies"] for s, r in crf.items()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "summary.md").write_text(render_md(summary))
    return summary


def _vals(d: Mapping[Any, float]) -> str:
    return " / ".join(f"{d[k]:.2f}" for k in sorted(d))


def _hit(h: Mapping[str, Any] | None) -> str:
    if h is None:
        return "-"
    return f"{h['met']} ({h['seeds_met']}/{h['seeds']})"


def render_md(s: Mapping[str, Any]) -> str:
    pc = s["control_preference"]
    cc = s["control_crf_ext"]
    lines = [
        f"Sentence: `{s['sentence']}`",
        "",
        "| operation sentence | C6 rate (seeds) | minus control mean | E1 |",
        "|---|---|---|---|",
        f"| none (control) | {_vals(pc)} | mean {statistics.mean(pc.values()):.2f} | - |",
    ]
    for op, row in s["preference"].items():
        lines.append(f"| {op} | {_vals(row['rates'])} | {_vals(row['diffs'])} | {_hit(row['E1'])} |")
    lines += [
        "",
        "| operation sentence | CRF last 20 | EXT first 10 | EXT last 20 | CRF - EXT | E2 | E3 |",
        "|---|---|---|---|---|---|---|",
        "| none (control) | "
        + " | ".join(_vals({k: v[m] for k, v in cc.items()})
                     for m in ("crf_last20_rate", "ext_first10_rate", "ext_last20_rate"))
        + " | " + _vals({k: v["crf_last20_rate"] - v["ext_last20_rate"] for k, v in cc.items()})
        + " | - | - |",
    ]
    for op, row in s["crf_ext"].items():
        lines.append(
            f"| {op} | {_vals(row['crf_last20'])} | {_vals(row['ext_first10'])} | {_vals(row['ext_last20'])} "
            f"| {_vals(row['gap'])} | {_hit(row['E2'])} | {_hit(row.get('E3'))} |"
        )
    ceiling = [op for op, row in s["crf_ext"].items() if row["control_at_ceiling"]]
    if ceiling:
        lines += ["", "Control EXT last 20 is at the ceiling (1.00): E3 and the deprivation side of E2 "
                  "cannot be met by an increase."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preference", choices=sorted(OPERATIONS))
    group.add_argument("--crf-ext", choices=sorted(OPERATIONS))
    group.add_argument("--summarize", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--crf", type=int, default=50)
    parser.add_argument("--ext", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for one reply")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/establishing_operation"))
    parser.add_argument("--control-preference", type=Path, help="directory of the preference seed files")
    parser.add_argument("--control-crf-ext", type=Path, help="directory of the reinforcer-assessment runs")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        if args.control_preference is None or args.control_crf_ext is None:
            parser.error("--summarize needs --control-preference and --control-crf-ext")
        print(render_md(summarize(args.out_dir, args.control_preference, args.control_crf_ext)), end="")
        return
    if args.preference:
        path = args.out_dir / f"run_pref_{args.preference}_s{args.seed}_{args.model}.json"
    else:
        path = args.out_dir / f"run_crfext_{args.crf_ext}_s{args.seed}_{args.model}.json"
    if path.exists():
        print(f"skip: {path.name} exists", flush=True)
        return
    if args.preference:
        res = run_preference_op(args.model, args.preference, args.seed, args.timeout)
        invalid = res["invalid"]
    else:
        res = run_crf_ext_op(args.model, args.crf_ext, args.seed, args.crf, args.ext, args.timeout)
        invalid = res["invalid_replies"]
    path.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    print(f"{path.name}: done in {res['wall_s']:.0f} s, invalid {invalid}", flush=True)


if __name__ == "__main__":
    main()
