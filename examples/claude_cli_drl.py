"""Reduce excessive tool calls with DRL: a chain-following task under CRF vs DRL.

Each run gives a fresh ``claude -p`` session (Bash only) a directory of
record files: 12 form a chain (``amount: <n>`` / ``next: <record>``, the
last one ``next: END``) and 6 are decoys. The subject reports the chain's
sum. A ``PreToolUse`` command hook (:mod:`ai_operant_box.agent.tool_contingency`)
decides every attempted Bash call: under CRF every attempt executes; under
DRL t an attempt executes only if t s have passed since the last executed
one, otherwise it is denied with "The command was not executed." (no rule).

    .venv/bin/python examples/claude_cli_drl.py --schedule crf --reps 3 --out-dir .local/runs/s51
    .venv/bin/python examples/claude_cli_drl.py --schedule drl --t 5 --reps 3 --out-dir .local/runs/s51
    .venv/bin/python examples/claude_cli_drl.py --summarize .local/runs/s51

``--fake`` swaps ``claude`` for a scripted subject that reads one file per
call and retries a denied call after 1 s (dry run of the whole pipeline).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

N_CHAIN = 12
N_DECOY = 6
BLOCK_HOPS = 4

TASK = (
    "The directory ./data contains record files. Start at data/{start}.txt. "
    "Each record has a line `amount: <n>` and a line `next: <record>`; "
    "follow `next` from record to record until `next: END`. "
    "Add up the amounts of every record on that chain, including the first and the last. "
    "Use the Bash tool to read the files. "
    "End your answer with a line of the form `SUM = <n>`."
)

FAKE_CLAUDE = r'''
import json, re, subprocess, sys, time, uuid
args = sys.argv[1:]
settings = json.loads(args[args.index("--settings") + 1])
pre = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
post = settings["hooks"]["PostToolUse"][0]["hooks"][0]["command"]
start = re.search(r"data/(record_\d+)\.txt", args[-1]).group(1)
session = uuid.uuid4().hex
def hook(cmd, payload):
    out = subprocess.run(cmd, shell=True, input=json.dumps(payload), capture_output=True, text=True, check=True).stdout
    return json.loads(out) if out.strip() else {}
total, rec, n = 0, start, 0
while rec != "END":
    n += 1
    command = f"cat data/{rec}.txt"
    payload = {"session_id": session, "hook_event_name": "PreToolUse", "tool_name": "Bash",
               "tool_input": {"command": command}, "tool_use_id": f"t{n}"}
    decision = hook(pre, payload)["hookSpecificOutput"]["permissionDecision"]
    time.sleep(0.3)
    if decision != "allow":
        time.sleep(1.0)
        continue
    text = open(f"data/{rec}.txt").read()
    hook(post, dict(payload, hook_event_name="PostToolUse", tool_response={"stdout": text, "stderr": ""}))
    total += int(re.search(r"amount: (\d+)", text).group(1))
    rec = re.search(r"next: (\S+)", text).group(1)
print(json.dumps({"type": "result", "subtype": "success", "result": f"SUM = {total}",
                  "total_cost_usd": 0.0, "num_turns": n, "duration_ms": 0}))
'''


def make_chain(data_dir: Path, seed: int) -> tuple[str, int, list[str]]:
    """Write the chain and decoy records; return (start, true sum, chain order)."""
    rng = random.Random(seed)
    names = [f"record_{i:02d}" for i in rng.sample(range(10, 100), N_CHAIN + N_DECOY)]
    chain, decoys = names[:N_CHAIN], names[N_CHAIN:]
    data_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for i, name in enumerate(chain):
        amount = rng.randint(100, 999)
        total += amount
        nxt = chain[i + 1] if i + 1 < N_CHAIN else "END"
        (data_dir / f"{name}.txt").write_text(f"amount: {amount}\nnext: {nxt}\n")
    for i, name in enumerate(decoys):
        nxt = decoys[i + 1] if i + 1 < N_DECOY else "END"
        (data_dir / f"{name}.txt").write_text(f"amount: {rng.randint(100, 999)}\nnext: {nxt}\n")
    return chain[0], total, chain


def run_once(
    schedule: str, t: float, seed: int, model: str, out_dir: Path, fake: bool, timeout: float
) -> dict[str, Any]:
    tag = f"{schedule}{'' if schedule == 'crf' else f'{t:g}'}_{model}_seed{seed}"
    hook_dir = (out_dir / "hook" / tag).resolve()
    if hook_dir.exists():
        shutil.rmtree(hook_dir)
    hook_dir.mkdir(parents=True)
    work = Path(tempfile.mkdtemp(prefix="aob-drl-"))
    start, true_sum, chain = make_chain(work / "data", seed)

    hook_cmd = (
        f"{sys.executable} -m ai_operant_box.agent.tool_contingency "
        f"--schedule {schedule} --t {t} --state-dir {hook_dir}"
    )
    hook = [{"matcher": "Bash", "hooks": [{"type": "command", "command": hook_cmd}]}]
    settings = json.dumps({"hooks": {"PreToolUse": hook, "PostToolUse": hook}})
    prompt = TASK.format(start=start)
    if fake:
        fake_path = work / "fake_claude.py"
        fake_path.write_text(FAKE_CLAUDE)
        cmd = [sys.executable, str(fake_path), "--settings", settings, prompt]
    else:
        cmd = [
            "claude", "-p", "--model", model,
            "--tools", "Bash", "--allowedTools", "Bash", "--permission-prompts", "none",
            "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence",
            "--settings", settings,
            "--output-format", "stream-json", "--verbose", "--include-hook-events",
            prompt,
        ]
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=timeout)
        stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr, code = "timeout", -1
    wall = time.time() - started
    (out_dir / f"{tag}.stream.jsonl").write_text(stdout)
    if stderr.strip():
        (out_dir / f"{tag}.stderr.txt").write_text(stderr)
    shutil.rmtree(work, ignore_errors=True)

    result: dict[str, Any] = {}
    for line in stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "result":
            result = ev
    trials: list[dict[str, Any]] = []
    for path in sorted(hook_dir.glob("trials_*.jsonl")):
        trials += [json.loads(x) for x in path.read_text().splitlines() if x]
    trials.sort(key=lambda r: r["t"])
    answer = re.findall(r"SUM\s*=\s*(-?\d+)", result.get("result") or "")
    run = {
        "tag": tag, "schedule": schedule, "t": t, "model": model, "seed": seed, "fake": fake,
        "start": start, "chain": chain, "true_sum": true_sum,
        "reported_sum": int(answer[-1]) if answer else None,
        "correct": bool(answer) and int(answer[-1]) == true_sum,
        "exit_code": code, "wall_s": wall, "t0": started,
        "total_cost_usd": result.get("total_cost_usd"), "num_turns": result.get("num_turns"),
        "final_text": result.get("result"), "trials": trials,
    }
    run["measures"] = measures(run)
    (out_dir / f"{tag}.json").write_text(json.dumps(run, ensure_ascii=False, indent=1))
    return run


LOOP = re.compile(r"\b(while|for|xargs|until)\b")
SLEEP = re.compile(r"\bsleep\b")


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def measures(run: dict[str, Any]) -> dict[str, Any]:
    trials, t = run["trials"], run["t"]
    chain_pos = {name: i for i, name in enumerate(run["chain"])}
    executed = [r for r in trials if r["executed"]]
    irt_attempt = [r["irt_attempt"] for r in trials if r["irt_attempt"] is not None]
    times_exec = [r["t"] for r in executed]
    irt_exec = [b - a for a, b in zip(times_exec, times_exec[1:], strict=False)]
    blocks: list[dict[str, Any]] = []
    reached = -1
    per_block: dict[int, list[dict[str, Any]]] = {}
    for r in trials:
        block = min(N_CHAIN // BLOCK_HOPS - 1, (reached + 1) // BLOCK_HOPS)
        per_block.setdefault(block, []).append(r)
        if r["executed"]:
            for name in re.findall(r"record_\d+", r.get("command") or ""):
                if name in chain_pos:
                    reached = max(reached, chain_pos[name])
    for b in range(N_CHAIN // BLOCK_HOPS):
        rows = per_block.get(b, [])
        irts = [r["irt_attempt"] for r in rows if r["irt_attempt"] is not None]
        blocks.append({
            "block": b + 1, "attempts": len(rows), "executed": sum(r["executed"] for r in rows),
            "reinf_rate": (sum(r["executed"] for r in rows) / len(rows)) if rows else None,
            "irt_median": _median(irts),
        })
    files = [len(re.findall(r"record_\d+", r.get("command") or "")) for r in trials]
    loops = [i for i, r in enumerate(trials) if LOOP.search(r.get("command") or "")]
    first_loop = loops[0] if loops else None
    return {
        "first_loop_attempt": None if first_loop is None else first_loop + 1,
        "denials_before_first_loop": (
            None if first_loop is None else sum(not r["executed"] for r in trials[:first_loop])
        ),
        "attempts": len(trials), "executed": len(executed), "denied": len(trials) - len(executed),
        "reinf_rate": len(executed) / len(trials) if trials else None,
        "irt_attempt_median": _median(irt_attempt),
        "irt_exec_median": _median(irt_exec),
        "share_irt_ge_t": (sum(x >= t for x in irt_attempt) / len(irt_attempt)) if irt_attempt and t else None,
        "files_per_attempt_mean": statistics.mean(files) if files else None,
        "loop_attempts": sum(bool(LOOP.search(r.get("command") or "")) for r in trials),
        "sleep_attempts": sum(bool(SLEEP.search(r.get("command") or "")) for r in trials),
        "chain_reached": reached + 1,
        "blocks": blocks,
        "irt_attempt": irt_attempt, "irt_exec": irt_exec,
    }


def _fmt(x: Any, nd: int = 2) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "○" if x else "×"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def summarize(out_dir: Path) -> None:
    runs = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*_seed*.json"))]
    runs = [r for r in runs if "trials" in r]
    for r in runs:
        r["measures"] = measures(r)  # recomputed from the raw trials
    head = ("| 条件 | 種 | 試行 | 実行 | 不実行 | 強化率 | 試行 IRT 中央 s | 実行 IRT 中央 s | IRT ≥ t の割合 "
            "| ループ試行 | 最初のループ試行 | その前の不実行 | sleep | 正答 | 所要 s | USD |")
    lines = [head, "|" + "---|" * 16]
    for r in runs:
        m = r["measures"]
        lines.append("| " + " | ".join([
            r["tag"].split("_seed")[0], str(r["seed"]), str(m["attempts"]), str(m["executed"]), str(m["denied"]),
            _fmt(m["reinf_rate"]), _fmt(m["irt_attempt_median"]), _fmt(m["irt_exec_median"]),
            _fmt(m["share_irt_ge_t"]), str(m["loop_attempts"]), _fmt(m["first_loop_attempt"]),
            _fmt(m["denials_before_first_loop"]), str(m["sleep_attempts"]), _fmt(r["correct"]),
            _fmt(r["wall_s"], 1), _fmt(r["total_cost_usd"], 3),
        ]) + " |")
    lines += ["", "ブロック（鎖 4 ホップごと）: 試行 / 実行 / 試行 IRT 中央 s", "",
              "| 条件 | 種 | ブロック 1 | ブロック 2 | ブロック 3 |", "|---|---|---|---|---|"]
    for r in runs:
        cells = [f"{b['attempts']} / {b['executed']} / {_fmt(b['irt_median'])}" for b in r["measures"]["blocks"]]
        lines.append(f"| {r['tag'].split('_seed')[0]} | {r['seed']} | " + " | ".join(cells) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    _plot(runs, out_dir / "irt_by_condition.png")


def _plot(runs: list[dict[str, Any]], png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = sorted({r["tag"].split("_seed")[0] for r in runs})
    fig, axes = plt.subplots(2, len(conds), figsize=(5 * len(conds), 6.5), squeeze=False)
    bins = [i * 0.5 for i in range(0, 41)]
    for j, cond in enumerate(conds):
        rs = [r for r in runs if r["tag"].startswith(cond + "_seed")]
        t = rs[0]["t"]
        for i, key in enumerate(("irt_attempt", "irt_exec")):
            ax = axes[i][j]
            for r in rs:
                ax.hist([min(x, 19.9) for x in r["measures"][key]], bins=bins, histtype="step",
                        label=f"seed {r['seed']}")
            if t:
                ax.axvline(t, color="k", ls="--", lw=1)
            ax.set_title(f"{cond}: {'attempt' if i == 0 else 'executed'} IRT")
            ax.set_xlabel("s (≥ 20 s pooled at the right edge)")
            ax.set_ylabel("count")
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--schedule", choices=("crf", "ext", "drl", "drh"), default="crf")
    parser.add_argument("--t", type=float, default=0.0)
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed0", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out-dir", type=Path, default=Path(".local/runs/s51"))
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--summarize", type=Path)
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for k in range(args.reps):
        run = run_once(args.schedule, args.t, args.seed0 + k, args.model, args.out_dir, args.fake, args.timeout)
        m = run["measures"]
        print(f"{run['tag']}: attempts={m['attempts']} executed={m['executed']} "
              f"irt_med={_fmt(m['irt_attempt_median'])} correct={run['correct']} "
              f"wall={run['wall_s']:.1f}s exit={run['exit_code']} usd={run['total_cost_usd']}", flush=True)


if __name__ == "__main__":
    main()
