"""Stimulus-equivalence example: stage flow, probe counts, verdicts and the result table."""

from __future__ import annotations

import importlib.util
import json
import re
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
