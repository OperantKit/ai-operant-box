"""Stimulus-equivalence test of a resident Claude conversation.

Procedure (simultaneous matching to sample, three classes of three nonsense
syllables, after Sidman & Tailby, 1982):

1. T-AB and T-BC: train each relation on continuous reinforcement.
2. T-mix: both relations mixed, continuous reinforcement.
3. T-p: both relations mixed, a correct choice earns a point with p = .2. The
   stage opens with a one-line notice that points get rarer.
4. X-sym: baseline at p = .2 with unreinforced probes of symmetry (BA, CB).
5. X-tr: baseline at p = .2 with unreinforced probes of transitivity (AC) and
   equivalence (CA).

A stage ends at its criterion or after its block limit; a subject that misses
the limit stops there and takes no test. Each relation gets 30 probes, scored
as derived (>= 90% correct), above chance (one-sided binomial against 1/3,
p < .01) or chance. A test whose baseline falls below 90% correct is invalid.

Example (one process per seed; ``--pilot`` runs one T-AB block to read the cost)::

    python examples/claude_cli_equivalence.py --model haiku --seed 0 --out-dir runs/equivalence

Each block rewrites ``seed<n>.json``, so a failed run keeps its finished blocks.
Each run ends by rebuilding ``summary.json`` and ``table.md`` from every seed
file in the directory.

Controls:

- ``--stimuli words`` swaps the syllables for meaningful words (red-apple-rouge,
  blue-sky-bleu, green-leaf-vert) and opens with a pretest: 5 presentations of
  each BA, CB, AC and CA trial type (60 trials, no baseline, no result lines,
  no reinforcement) measure what the names alone get right before training.
- ``--skip-training`` runs X-sym and X-tr only (180 trials). The baseline is
  untrained too, so these trials estimate the floor of the test itself. Its
  baseline still earns points at p = .2, so AB and BC can be learned during it.

- ``--history-window 30 --history-from <seed file>`` runs X-tr alone with only the
  last 30 exchanges in context, each trial in a new process, starting from the
  end of X-sym in the seed file. Training trials are out of context, so correct
  probes cannot come from reading them.

Stage seeds come from a stage's place in the full procedure, so a control shows
the same trial order as the same seed of the full procedure.

Reference:
    Sidman, M., & Tailby, W. (1982). Conditional discrimination vs. matching to
    sample: An expansion of the testing paradigm. *Journal of the Experimental
    Analysis of Behavior, 37*(1), 5-22. https://doi.org/10.1901/jeab.1982.37-5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from contingency.schedules import EXT, RR, MatchingToSample
from contingency.schedules.trial_based import TrialType, trial_types_from_relations

from ai_operant_box.agent.api_client import ClaudeCLIClient
from ai_operant_box.chamber.mts import (
    MatchingToSampleBox,
    MatchingToSampleSubject,
    TrialOutcome,
    points_feedback,
)

SYSTEM_PROMPT = (
    "You are a subject in a matching-to-sample experiment. On each trial you see a sample "
    "and three comparison stimuli. Choose the comparison that goes with the sample by "
    "replying with its number only (1, 2 or 3). Correct choices sometimes earn a point."
)
LEAN_INSTRUCTION = (
    "You won't always get a point from now on, but you will have a chance to get more later."
)
STIMULUS_SETS = {
    "nonsense": {
        "A": ("ZIB", "VOK", "MEF"),
        "B": ("DAQ", "TUJ", "PEV"),
        "C": ("LOJ", "HUV", "WEZ"),
    },
    "words": {
        "A": ("red", "blue", "green"),
        "B": ("apple", "sky", "leaf"),
        "C": ("rouge", "bleu", "vert"),
    },
}
STIMULI = STIMULUS_SETS["nonsense"]
RELATIONS = ("BA", "CB", "AC", "CA")
LEAN_P = 0.2


def _locate(stimulus: str) -> tuple[str, int]:
    """The set name and class index of ``stimulus``; the stimulus sets share no member."""
    for stimuli in STIMULUS_SETS.values():
        for name, members in stimuli.items():
            if stimulus in members:
                return name, members.index(stimulus)
    raise KeyError(stimulus)


def set_of(stimulus: str) -> str:
    return _locate(stimulus)[0]


def class_of(stimulus: str) -> int:
    return _locate(stimulus)[1]


def relation(label: str, stimuli: dict[str, tuple[str, ...]] = STIMULI) -> tuple[TrialType, ...]:
    """The three trial types of ``label``: sample set then comparison set, e.g. ``"BA"``."""
    pairs = dict(zip(stimuli[label[0]], stimuli[label[1]], strict=True))
    return tuple(trial_types_from_relations(pairs))


@dataclass(frozen=True)
class Stage:
    name: str
    baseline: tuple[TrialType, ...]
    probes: tuple[TrialType, ...] = ()
    criterion: int | None = None  # correct baseline trials per block; None for a test
    max_blocks: int = 1
    p: float | None = None  # None = continuous reinforcement
    seed_offset: int = 0  # the stage's place in the full procedure
    pretest: bool = False  # probes only: no baseline, no result lines, no reinforcement

    @property
    def block_length(self) -> int:
        return len(self.baseline) + len(self.probes)


_PRETEST_SEED_OFFSET = 6


def build_stages(
    stimuli: dict[str, tuple[str, ...]],
    *,
    skip_training: bool = False,
    pretest: bool = False,
) -> tuple[Stage, ...]:
    """The stages over ``stimuli``, optionally without training or after a pretest."""

    def rel(label: str) -> tuple[TrialType, ...]:
        return relation(label, stimuli)

    mixed = rel("AB") * 5 + rel("BC") * 5
    training = (
        Stage("T-AB", rel("AB") * 10, criterion=29, max_blocks=5, seed_offset=0),
        Stage("T-BC", rel("BC") * 10, criterion=29, max_blocks=5, seed_offset=1),
        Stage("T-mix", mixed, criterion=29, max_blocks=3, seed_offset=2),
        Stage("T-p", mixed, criterion=27, max_blocks=2, p=LEAN_P, seed_offset=3),
    )
    tests = (
        Stage("X-sym", mixed, rel("BA") * 10 + rel("CB") * 10, p=LEAN_P, seed_offset=4),
        Stage("X-tr", mixed, rel("AC") * 10 + rel("CA") * 10, p=LEAN_P, seed_offset=5),
    )
    probes = tuple(t for label in RELATIONS for t in rel(label) * 5)
    opening = (Stage("Pre", (), probes, seed_offset=_PRETEST_SEED_OFFSET, pretest=True),) if pretest else ()
    return (*opening, *(() if skip_training else training), *tests)


def stages_for(stimuli: str, *, skip_training: bool) -> tuple[Stage, ...]:
    """The stages of the command-line flags: only a trained word run opens with the pretest."""
    return build_stages(
        STIMULUS_SETS[stimuli],
        skip_training=skip_training,
        pretest=stimuli == "words" and not skip_training,
    )


STAGES = build_stages(STIMULI)


def verdict(correct: int, n: int) -> str:
    """``derived`` at 90%, ``above_chance`` at one-sided binomial p < .01 against 1/3, else ``chance``."""
    if correct * 10 >= n * 9:
        return "derived"
    tail = sum(math.comb(n, k) * (1 / 3) ** k * (2 / 3) ** (n - k) for k in range(correct, n + 1))
    return "above_chance" if tail < 0.01 else "chance"


def baseline_valid(correct: int, n: int) -> bool:
    return correct * 10 >= n * 9


def _score_test(name: str, trials: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = [t for t in trials if not t["probe"]]
    by_relation: dict[str, list[dict[str, Any]]] = {}
    for t in trials:
        if t["probe"]:
            label = set_of(t["sample"]) + set_of(t["correct"])
            by_relation.setdefault(label, []).append(t)
    relations = []
    for label, probes in sorted(by_relation.items(), key=lambda kv: RELATIONS.index(kv[0])):
        correct = sum(t["is_correct"] for t in probes)
        errors = Counter(t["sample"] for t in probes if not t["is_correct"])
        samples = sorted({t["sample"] for t in probes})
        relations.append({
            "relation": label,
            "n": len(probes),
            "correct": correct,
            "verdict": verdict(correct, len(probes)),
            "errors_by_sample": {s: errors[s] for s in samples},
            "each_sample_within_one_error": all(errors[s] <= 1 for s in samples),
        })
    baseline_correct = sum(t["is_correct"] for t in baseline)
    return {
        "stage": name,
        "relations": relations,
        "baseline_correct": baseline_correct,
        "baseline_n": len(baseline),
        "valid": baseline_valid(baseline_correct, len(baseline)),
    }


class WindowedClient:
    """``send`` over the last ``window`` exchanges, each trial in a new process.

    The transcript is the one :meth:`ClaudeCLIClient.generate` sends; ``factory``
    builds the client here so that the cost of every process can be added up.
    """

    def __init__(
        self,
        factory: Callable[[], Any],
        *,
        window: int,
        history: Iterable[tuple[str, str]] = (),
    ) -> None:
        self.factory = factory
        self.window = window
        self.exchanges = list(history)
        self.latencies: list[float] = []
        self.last_result: dict[str, float] = {"total_cost_usd": 0.0}

    def send(self, text: str) -> str:
        lines = [
            line
            for user, assistant in self.exchanges[-self.window:]
            for line in (f"[user] {user}", f"[assistant] {assistant}")
        ]
        lines.append(f"[user] {text}")
        started = time.monotonic()
        with self.factory() as one_shot:
            reply = one_shot.send("\n".join(lines))
            cost = (one_shot.last_result or {}).get("total_cost_usd") or 0.0
        self.latencies.append(time.monotonic() - started)
        self.last_result["total_cost_usd"] += cost
        self.exchanges.append((text, reply))
        return reply


def _result_line(trial: dict[str, Any]) -> str | None:
    return points_feedback(TrialOutcome(trial["reinforced"], trial["probe"], trial["points"]))


def history_from_run(run: dict[str, Any], stage: str, count: int) -> list[tuple[str, str]]:
    """The last ``count`` exchanges of ``stage``: result line, trial text and reply as they went out."""
    trials = run["trials"]
    positions = [i for i, t in enumerate(trials) if t["stage"] == stage][-count:]
    exchanges = []
    for i in positions:
        trial = trials[i]
        options = "  ".join(f"{n}) {name}" for n, name in enumerate(trial["comparisons"], start=1))
        lines = [] if i == 0 else [_result_line(trials[i - 1])]
        lines += [f"Sample: {trial['sample']}", f"Comparisons: {options}", "Choose 1, 2 or 3."]
        exchanges.append(("\n".join(lines), trial["raw"]))
    return exchanges


def _no_feedback(_: TrialOutcome) -> None:
    return None


def run_experiment(
    client: Any,
    seed: int,
    *,
    stages: tuple[Stage, ...] = STAGES,
    on_block: Callable[[dict[str, Any]], None] | None = None,
    start_points: int = 0,
    first_lines: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Run the stages in order on one conversation and return the run record.

    ``start_points`` and ``first_lines`` continue a subject that already earned
    points and has a result line waiting for its next trial.
    """
    started = time.monotonic()
    subject = MatchingToSampleSubject(client)
    subject.points = start_points
    for line in first_lines:
        subject.instruct(line)
    run: dict[str, Any] = {
        "seed": seed,
        "status": "running",
        "stages": [],
        "tests": [],
        "numbers": {"1": 0, "2": 0, "3": 0},
        "invalid_replies": 0,
        "points": 0,
        "trials": [],
    }

    def finish_block() -> None:
        latencies = getattr(client, "latencies", None)
        run["invalid_replies"] = subject.invalid_count
        run["points"] = subject.points
        run["wall_s"] = time.monotonic() - started
        run["latencies_s"] = list(latencies) if latencies is not None else []
        run["api_equivalent_usd"] = (getattr(client, "last_result", None) or {}).get("total_cost_usd")
        if on_block is not None:
            on_block(run)

    for index, stage in enumerate(stages):
        stage_seed = seed * 100 + stage.seed_offset
        consequence: Any = None if stage.p is None else RR(stage.p, rng=random.Random(stage_seed))
        if stage.pretest:
            # The engine needs baseline trial types, so the pretest's trials are
            # ordinary trials on extinction, marked as probes in the record.
            consequence = EXT()
        schedule = MatchingToSample(
            3,
            sample_duration=0,
            choice_timeout=1,
            iti=0,
            seed=stage_seed,
            trial_types=stage.probes if stage.pretest else stage.baseline,
            probe_trial_types=stage.probes if stage.probes and not stage.pretest else None,
            consequence=consequence,
        )
        box = MatchingToSampleBox(
            subject,
            schedule,
            block_length=stage.block_length,
            feedback=_no_feedback if stage.pretest else points_feedback,
        )
        if stage.name == "T-p":
            subject.instruct(LEAN_INSTRUCTION)

        stage_trials: list[dict[str, Any]] = []
        entry = {"stage": stage.name, "blocks": 0, "trials": 0, "reached": False, "block_correct": []}
        run["stages"].append(entry)
        for block in range(stage.max_blocks):
            for record in box.run_block():
                latencies = getattr(client, "latencies", None)
                position = len(run["trials"])
                trial = {
                    "stage": stage.name,
                    "block": block,
                    **asdict(record),
                    "probe": record.probe or stage.pretest,
                    "raw": record.raw_reply,
                    "latency_s": latencies[position] if latencies is not None else None,
                }
                del trial["raw_reply"]
                run["trials"].append(trial)
                stage_trials.append(trial)
                if record.choice is not None:
                    run["numbers"][str(record.comparisons.index(record.choice) + 1)] += 1
            scored = [t for t in stage_trials[-stage.block_length:] if stage.pretest or not t["probe"]]
            n_correct = sum(t["is_correct"] for t in scored)
            entry["blocks"] += 1
            entry["trials"] += stage.block_length
            entry["block_correct"].append(n_correct)
            if stage.criterion is None or n_correct >= stage.criterion:
                entry["reached"] = True
                if stage.pretest:
                    run["pretest"] = _score_test(stage.name, stage_trials)
                elif stage.probes:
                    run["tests"].append(_score_test(stage.name, stage_trials))
                break
            if block + 1 < stage.max_blocks:
                finish_block()
        if not entry["reached"]:
            run["status"] = "not_reached"
            finish_block()
            return run
        if index + 1 < len(stages):
            finish_block()

    run["status"] = "completed"
    finish_block()
    return run


def save_json(path: Path, data: Any) -> None:
    """Write ``data`` atomically, so that a reader or a crash never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def render_table(runs: list[dict[str, Any]]) -> str:
    """The result tables in Markdown, one row per seed and stage or relation."""
    lines = [
        "### Stages", "",
        "| Seed | Stage | Blocks | Trials | Reached | Correct per block |",
        "|---|---|---|---|---|---|",
    ]
    for run in runs:
        for s in run["stages"]:
            correct = ", ".join(map(str, s["block_correct"]))
            reached = "yes" if s["reached"] else "no"
            lines.append(f"| {run['seed']} | {s['stage']} | {s['blocks']} | {s['trials']} "
                         f"| {reached} | {correct} |")
    lines += [
        "", "### Probes", "",
        "| Seed | Test | Relation | Correct | Verdict | Errors by sample | Each sample <= 1 error |",
        "|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        for test in run["tests"]:
            for r in test["relations"]:
                errors = ", ".join(f"{s} {n}" for s, n in r["errors_by_sample"].items())
                within = "yes" if r["each_sample_within_one_error"] else "no"
                lines.append(f"| {run['seed']} | {test['stage']} | {r['relation']} "
                             f"| {r['correct']}/{r['n']} | {r['verdict']} | {errors} | {within} |")
    pretests = [run for run in runs if "pretest" in run]
    if pretests:
        lines += ["", "### Pretest before training", "",
                  "| Seed | Test | Relation | Correct | Verdict |", "|---|---|---|---|---|"]
        for run in pretests:
            for r in run["pretest"]["relations"]:
                lines.append(f"| {run['seed']} | Pre | {r['relation']} | {r['correct']}/{r['n']} "
                             f"| {r['verdict']} |")
    lines += ["", "### Baseline during tests", "",
              "| Seed | Test | Correct | Rate | Validity |", "|---|---|---|---|---|"]
    for run in runs:
        for t in run["tests"]:
            rate = 100 * t["baseline_correct"] / t["baseline_n"]
            validity = "valid" if t["valid"] else "invalid"
            lines.append(f"| {run['seed']} | {t['stage']} | {t['baseline_correct']}/{t['baseline_n']} "
                         f"| {rate:.0f}% | {validity} |")
    lines += ["", "### Subject", "",
              "| Seed | Status | Number 1 | Number 2 | Number 3 | Invalid | Points | API-equivalent USD | Wall s |",
              "|---|---|---|---|---|---|---|---|---|"]
    for run in runs:
        n = run["numbers"]
        cost = run.get("api_equivalent_usd")
        cost_text = "-" if cost is None else f"{cost:.3f}"
        lines.append(f"| {run['seed']} | {run['status']} | {n['1']} | {n['2']} | {n['3']} "
                     f"| {run['invalid_replies']} | {run['points']} | {cost_text} "
                     f"| {run.get('wall_s', 0):.0f} |")
    return "\n".join(lines) + "\n"


def write_tables(out_dir: Path) -> None:
    """Rebuild ``summary.json`` and ``table.md`` from every ``seed*.json`` in ``out_dir``."""
    runs = [json.loads(p.read_text()) for p in out_dir.glob("seed*.json")]
    runs.sort(key=lambda run: run["seed"])
    derived = dict.fromkeys(RELATIONS, 0)
    for run in runs:
        for test in run["tests"]:
            for r in test["relations"]:
                if r["verdict"] == "derived":
                    derived[r["relation"]] += 1
    summary = {
        "seeds": [{k: v for k, v in run.items() if k not in ("trials", "latencies_s")} for run in runs],
        "derived": derived,
    }
    save_json(out_dir / "summary.json", summary)
    (out_dir / "table.md").write_text(render_table(runs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/equivalence"))
    parser.add_argument("--pilot", action="store_true", help="run one T-AB block to read the cost")
    parser.add_argument("--stimuli", choices=sorted(STIMULUS_SETS), default="nonsense",
                        help="words also opens with an unreinforced pretest")
    parser.add_argument("--skip-training", action="store_true", help="run X-sym and X-tr only")
    parser.add_argument("--history-window", type=int, help="run X-tr alone with this many exchanges in context")
    parser.add_argument("--history-from", type=Path, help="seed file whose X-sym opens the window")
    args = parser.parse_args()
    if (args.history_window is None) != (args.history_from is None):
        parser.error("--history-window and --history-from go together")

    stages = stages_for(args.stimuli, skip_training=args.skip_training)
    if args.pilot:
        stages = (replace(STAGES[0], max_blocks=1),)
    path = args.out_dir / f"seed{args.seed}.json"

    def on_block(run: dict[str, Any]) -> None:
        save_json(path, run)
        print(f"seed {args.seed}: {len(run['trials'])} trials, "
              f"{run['stages'][-1]['stage']} {run['stages'][-1]['block_correct']}, "
              f"USD {run['api_equivalent_usd']}", flush=True)

    try:
        if args.history_window is not None:
            source = json.loads(args.history_from.read_text())
            source_symmetry = [t for t in source["trials"] if t["stage"] == "X-sym"]
            windowed = WindowedClient(
                lambda: ClaudeCLIClient(model=args.model, system_prompt=SYSTEM_PROMPT),
                window=args.history_window,
                history=history_from_run(source, "X-sym", args.history_window),
            )
            run = run_experiment(
                windowed,
                args.seed,
                stages=STAGES[-1:],
                on_block=on_block,
                start_points=source_symmetry[-1]["points"],
                first_lines=(_result_line(source_symmetry[-1]),),
            )
        else:
            with ClaudeCLIClient(model=args.model, system_prompt=SYSTEM_PROMPT) as client:
                run = run_experiment(client, args.seed, stages=stages, on_block=on_block)
    finally:
        if path.exists():
            write_tables(args.out_dir)
    print(f"seed {args.seed}: {run['status']}", flush=True)


if __name__ == "__main__":
    main()
