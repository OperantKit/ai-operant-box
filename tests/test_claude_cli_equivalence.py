"""Stimulus-equivalence example: stage flow, probe counts, verdicts and the result table."""

from __future__ import annotations

import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("contingency")


def _load() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "examples" / "claude_cli_equivalence.py"
    spec = importlib.util.spec_from_file_location("claude_cli_equivalence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


eq = _load()


class OracleClient:
    """Answers correctly from the three classes, or after ``knows`` relations fail by chance."""

    def __init__(self, wrong_on: set[tuple[str, str]] | None = None) -> None:
        self.wrong_on = wrong_on or set()
        self.sent: list[str] = []
        self.latencies: list[float] = []
        self.last_result = {"total_cost_usd": 0.25}

    def send(self, text: str) -> str:
        self.sent.append(text)
        self.latencies.append(0.5)
        sample = re.search(r"^Sample: (\S+)$", text, re.M)
        assert sample is not None
        names = [name for _, name in re.findall(r"(\d)\) (\S+)", text)]
        k = eq.class_of(sample.group(1))
        right = next(i for i, name in enumerate(names) if eq.class_of(name) == k)
        relation = (eq.set_of(sample.group(1)), eq.set_of(names[0]))
        return str((right + 1) % 3 + 1 if relation in self.wrong_on else right + 1)


class ConstantClient(OracleClient):
    def send(self, text: str) -> str:
        self.sent.append(text)
        self.latencies.append(0.5)
        return "1"


def test_oracle_reaches_every_stage_in_the_shortest_path_with_30_probes_per_relation() -> None:
    client = OracleClient()
    run = eq.run_experiment(client, seed=0)
    assert run["status"] == "completed"
    assert len(client.sent) == len(run["trials"]) == 300
    assert [(s["stage"], s["blocks"], s["trials"], s["reached"]) for s in run["stages"]] == [
        ("T-AB", 1, 30, True),
        ("T-BC", 1, 30, True),
        ("T-mix", 1, 30, True),
        ("T-p", 1, 30, True),
        ("X-sym", 1, 90, True),
        ("X-tr", 1, 90, True),
    ]
    tests = {t["stage"]: t for t in run["tests"]}
    assert {r["relation"]: r["correct"] for r in tests["X-sym"]["relations"]} == {"BA": 30, "CB": 30}
    assert {r["relation"]: r["correct"] for r in tests["X-tr"]["relations"]} == {"AC": 30, "CA": 30}
    assert all(r["n"] == 30 and r["verdict"] == "derived"
               for t in run["tests"] for r in t["relations"])
    assert all(t["baseline_correct"] == t["baseline_n"] == 30 and t["valid"] for t in run["tests"])


def test_probe_trials_are_never_reinforced_and_the_lean_stage_is_intermittent() -> None:
    run = eq.run_experiment(OracleClient(), seed=1)
    probes = [t for t in run["trials"] if t["probe"]]
    assert len(probes) == 120
    assert not any(t["reinforced"] for t in probes)
    lean = [t for t in run["trials"] if t["stage"] == "T-p"]
    assert 0 < sum(t["reinforced"] for t in lean) < 30
    crf = [t for t in run["trials"] if t["stage"] == "T-mix"]
    assert all(t["reinforced"] for t in crf)


def test_the_lean_stage_is_announced_once_before_its_first_trial() -> None:
    client = OracleClient()
    eq.run_experiment(client, seed=0)
    hits = [i for i, text in enumerate(client.sent) if eq.LEAN_INSTRUCTION in text]
    assert hits == [90]


def test_constant_replies_stop_at_the_first_stage_without_a_test() -> None:
    client = ConstantClient()
    run = eq.run_experiment(client, seed=0)
    assert run["status"] == "not_reached"
    assert [(s["stage"], s["blocks"], s["reached"]) for s in run["stages"]] == [("T-AB", 5, False)]
    assert len(client.sent) == 150
    assert run["tests"] == []
    assert sum(run["numbers"].values()) == 150 and run["numbers"]["1"] == 150


def test_subject_failing_the_derived_relation_is_scored_as_not_derived() -> None:
    run = eq.run_experiment(OracleClient(wrong_on={("A", "C")}), seed=0)
    tr = next(t for t in run["tests"] if t["stage"] == "X-tr")
    by = {r["relation"]: r for r in tr["relations"]}
    assert by["AC"]["correct"] == 0 and by["AC"]["verdict"] == "chance"
    assert by["AC"]["errors_by_sample"] == {"MEF": 10, "VOK": 10, "ZIB": 10}
    assert by["AC"]["each_sample_within_one_error"] is False
    assert by["CA"]["verdict"] == "derived" and by["CA"]["each_sample_within_one_error"] is True


@pytest.mark.parametrize(
    ("correct", "verdict"),
    [(30, "derived"), (27, "derived"), (26, "above_chance"), (17, "above_chance"),
     (16, "chance"), (0, "chance")],
)
def test_verdict_boundaries_for_30_probes(correct: int, verdict: str) -> None:
    assert eq.verdict(correct, 30) == verdict


def test_baseline_below_90_percent_invalidates_the_test() -> None:
    assert eq.baseline_valid(27, 30) is True
    assert eq.baseline_valid(26, 30) is False


def test_on_block_saves_a_run_that_survives_a_later_failure(tmp_path: Path) -> None:
    class Failing(OracleClient):
        def send(self, text: str) -> str:
            if len(self.sent) == 45:
                raise RuntimeError("the process died")
            return super().send(text)

    path = tmp_path / "seed0.json"
    with pytest.raises(RuntimeError):
        eq.run_experiment(Failing(), seed=0, on_block=lambda run: eq.save_json(path, run))
    saved = json.loads(path.read_text())
    assert saved["status"] == "running"
    assert len(saved["trials"]) == 30
    assert set(saved["trials"][0]) >= {"sample", "comparisons", "choice", "correct", "probe",
                                       "reinforced", "raw", "points", "latency_s"}


def test_table_has_the_stage_relation_baseline_and_number_rows(tmp_path: Path) -> None:
    run = eq.run_experiment(OracleClient(wrong_on={("A", "C")}), seed=2)
    table = eq.render_table([run])
    assert "| 2 | T-AB | 1 | 30 | yes |" in table
    assert "| 2 | X-tr | AC | 0/30 | chance |" in table
    assert "| 2 | X-sym | BA | 30/30 | derived |" in table
    assert "| 2 | X-tr | 30/30 | 100% | valid |" in table
    assert "Invalid" in table


def test_write_tables_collects_every_seed_file_in_the_directory(tmp_path: Path) -> None:
    for seed in (0, 1):
        eq.save_json(tmp_path / f"seed{seed}.json", eq.run_experiment(OracleClient(), seed=seed))
    eq.write_tables(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert [s["seed"] for s in summary["seeds"]] == [0, 1]
    assert "trials" not in summary["seeds"][0]
    assert summary["derived"] == {"BA": 2, "CB": 2, "AC": 2, "CA": 2}
    assert (tmp_path / "table.md").read_text().count("| X-sym | BA |") == 2


def _word_stages() -> tuple:
    return eq.build_stages(eq.STIMULUS_SETS["words"], pretest=True)


def test_default_stages_are_the_nonsense_procedure_with_the_same_trial_order() -> None:
    assert eq.build_stages(eq.STIMULUS_SETS["nonsense"]) == eq.STAGES
    assert [s.seed_offset for s in eq.STAGES] == [0, 1, 2, 3, 4, 5]


def test_word_classes_are_red_apple_rouge_blue_sky_bleu_green_leaf_vert() -> None:
    words = eq.STIMULUS_SETS["words"]
    classes = [tuple(words[name][k] for name in "ABC") for k in range(3)]
    assert classes == [("red", "apple", "rouge"), ("blue", "sky", "bleu"), ("green", "leaf", "vert")]


def test_pretest_has_five_probes_per_type_without_consequences_before_training() -> None:
    client = OracleClient()
    run = eq.run_experiment(client, seed=0, stages=_word_stages())
    assert run["status"] == "completed"
    assert [s["stage"] for s in run["stages"]][:2] == ["Pre", "T-AB"]
    pre = [t for t in run["trials"] if t["stage"] == "Pre"]
    assert len(pre) == 60 and all(t["probe"] for t in pre)
    assert Counter((t["sample"], t["correct"]) for t in pre).most_common()[-1][1] == 5
    assert len({(t["sample"], t["correct"]) for t in pre}) == 12
    assert not any(t["reinforced"] for t in pre)
    assert not any("Result:" in text for text in client.sent[:61])
    assert pre[-1]["points"] == 0
    assert run["stages"][0]["block_correct"] == [60]


def test_pretest_is_scored_per_relation_apart_from_the_tests() -> None:
    run = eq.run_experiment(OracleClient(wrong_on={("A", "C")}), seed=1, stages=_word_stages())
    by = {r["relation"]: r for r in run["pretest"]["relations"]}
    assert {k: (r["correct"], r["n"]) for k, r in by.items()} == {
        "BA": (15, 15), "CB": (15, 15), "AC": (0, 15), "CA": (15, 15)}
    assert by["BA"]["verdict"] == "derived" and by["AC"]["verdict"] == "chance"
    assert [t["stage"] for t in run["tests"]] == ["X-sym", "X-tr"]


def test_pretest_rows_appear_in_the_table_and_stay_out_of_the_derived_count(tmp_path: Path) -> None:
    run = eq.run_experiment(OracleClient(), seed=0, stages=_word_stages())
    assert "| 0 | Pre | BA | 15/15 | derived |" in eq.render_table([run])
    eq.save_json(tmp_path / "seed0.json", run)
    eq.write_tables(tmp_path)
    assert json.loads((tmp_path / "summary.json").read_text())["derived"]["BA"] == 1


def test_training_trial_order_does_not_shift_when_a_pretest_comes_first() -> None:
    def order(stages: tuple, name: str) -> list[int]:
        trials = eq.run_experiment(OracleClient(), seed=2, stages=stages)["trials"]
        k = {s: i for members in eq.STIMULUS_SETS.values() for s, i in
             ((m, v.index(m)) for v in members.values() for m in v)}
        return [k[t["sample"]] for t in trials if t["stage"] == name]

    plain = eq.build_stages(eq.STIMULUS_SETS["words"])
    assert order(_word_stages(), "X-tr") == order(plain, "X-tr") == order(eq.STAGES, "X-tr")


def test_skip_training_runs_only_the_two_tests_with_180_trials_and_no_notice() -> None:
    client = OracleClient()
    stages = eq.build_stages(eq.STIMULUS_SETS["nonsense"], skip_training=True)
    run = eq.run_experiment(client, seed=0, stages=stages)
    assert [s["stage"] for s in run["stages"]] == ["X-sym", "X-tr"]
    assert len(client.sent) == 180
    assert sum(t["probe"] for t in run["trials"]) == 120
    assert not any(eq.LEAN_INSTRUCTION in text for text in client.sent)
    assert "pretest" not in run


def test_skip_training_keeps_the_trial_order_of_the_same_seed_in_the_full_procedure() -> None:
    def tests_of(stages: tuple) -> list[str]:
        trials = eq.run_experiment(OracleClient(), seed=1, stages=stages)["trials"]
        return [t["sample"] for t in trials if t["stage"] in ("X-sym", "X-tr")]

    skip = eq.build_stages(eq.STIMULUS_SETS["nonsense"], skip_training=True)
    assert tests_of(skip) == tests_of(eq.STAGES)


def test_stage_list_for_the_command_line_flags() -> None:
    assert eq.stages_for("nonsense", skip_training=False) == eq.STAGES
    assert eq.stages_for("words", skip_training=False)[0].name == "Pre"
    assert [s.name for s in eq.stages_for("nonsense", skip_training=True)] == ["X-sym", "X-tr"]


class _OneShot:
    """A one-turn process: answers from the transcript's last sample and reports its cost."""

    instances: list[_OneShot] = []

    def __init__(self) -> None:
        self.closed = False
        self.oracle = OracleClient()
        self.transcript = ""
        self.last_result = {"total_cost_usd": 0.5}
        _OneShot.instances.append(self)

    def __enter__(self) -> _OneShot:
        return self

    def __exit__(self, *exc: object) -> None:
        self.closed = True

    def send(self, text: str) -> str:
        self.transcript = text
        return self.oracle.send(text.rsplit("[user] ", 1)[1])


def test_windowed_client_sends_only_the_last_exchanges_each_in_a_new_process() -> None:
    _OneShot.instances = []
    history = [("Sample: ZIB", "1"), ("Sample: VOK", "2"), ("Sample: MEF", "3")]
    client = eq.WindowedClient(_OneShot, window=2, history=history)
    text = "Sample: ZIB\nComparisons: 1) DAQ  2) TUJ  3) PEV\nChoose 1, 2 or 3."
    assert client.send(text) == "1"
    assert _OneShot.instances[0].transcript == "\n".join([
        "[user] Sample: VOK", "[assistant] 2", "[user] Sample: MEF", "[assistant] 3", f"[user] {text}"])
    client.send(text)
    second = _OneShot.instances[1].transcript
    assert "Sample: VOK" not in second and second.count("[assistant]") == 2
    assert all(one.closed for one in _OneShot.instances)
    assert len(client.latencies) == 2
    assert client.last_result["total_cost_usd"] == pytest.approx(1.0)


def test_history_from_a_run_rebuilds_what_the_subject_saw() -> None:
    client = OracleClient()
    stages = eq.build_stages(eq.STIMULUS_SETS["nonsense"], skip_training=True)
    run = eq.run_experiment(client, seed=0, stages=stages)
    tail = [i for i, t in enumerate(run["trials"]) if t["stage"] == "X-sym"][-30:]
    rebuilt = eq.history_from_run(run, "X-sym", 30)
    assert [text for text, _ in rebuilt] == [client.sent[i] for i in tail]
    assert [reply for _, reply in rebuilt] == [run["trials"][i]["raw"] for i in tail]


def test_a_continued_run_keeps_the_points_and_opens_with_the_pending_result_line() -> None:
    client = OracleClient()
    only = eq.STAGES[-1:]
    run = eq.run_experiment(client, seed=0, stages=only, start_points=150, first_lines=("Result: No point. Points: 150",))
    assert client.sent[0].startswith("Result: No point. Points: 150\nSample: ")
    assert run["trials"][0]["points"] >= 150
    assert [s["stage"] for s in run["stages"]] == ["X-tr"]
