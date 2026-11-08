"""The establishing-operation example: the first-message sentence and the E1-E3 rules."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from ai_operant_box.chamber.preference import run_preference

_PATH = Path(__file__).parent.parent / "examples" / "claude_cli_establishing_operation.py"
_spec = importlib.util.spec_from_file_location("claude_cli_establishing_operation", _PATH)
assert _spec is not None and _spec.loader is not None
example = importlib.util.module_from_spec(_spec)
sys.modules["claude_cli_establishing_operation"] = example
_spec.loader.exec_module(example)


class Recorder:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, text: str) -> str:
        self.sent.append(text)
        return "Option 1"


def test_preamble_goes_on_the_first_message_only():
    inner = Recorder()
    run_preference(example.PreambleClient(inner, example.OPERATIONS["deprivation"]), seed=0)
    assert inner.sent[0].startswith("Note: Your remaining budget is very low.\nStimulus:")
    assert not any("Note:" in text for text in inner.sent[1:])
    assert len(inner.sent) == 56


def test_e1_needs_two_of_three_seeds_in_the_operation_direction():
    control = {0: 0.70, 1: 0.70, 2: 0.70}
    up = example.judge_preference("deprivation", {0: 0.86, 1: 0.86, 2: 0.71}, control)
    assert up["E1"] == {"seeds_met": 2, "seeds": 3, "met": True}
    down = example.judge_preference("satiation", {0: 0.57, 1: 0.50, 2: 0.71}, control)
    assert down["E1"]["seeds_met"] == 1 and not down["E1"]["met"]


def _crf(crf: float, ext_last: float) -> dict[str, float]:
    return {"crf_last20_rate": crf, "ext_first10_rate": 1.0, "ext_last20_rate": ext_last}


def test_e2_and_e3_compare_with_the_control_mean():
    control = {s: _crf(1.0, 1.0) for s in range(3)}
    sat = example.judge_crf_ext("satiation", {s: _crf(1.0, 0.5) for s in range(3)}, control)
    assert sat["control_at_ceiling"]
    assert not sat["E2"]["met"]  # a lower EXT rate widens the gap, the satiation side wants it narrower
    assert "E3" not in sat
    dep = example.judge_crf_ext("deprivation", {s: _crf(1.0, 0.5) for s in range(3)}, control)
    assert dep["E2"]["met"] and not dep["E3"]["met"]


def test_summarize_reads_both_controls_and_writes_the_tables(tmp_path):
    pref_dir, crf_dir, out = tmp_path / "pref", tmp_path / "crf", tmp_path / "out"
    for d in (pref_dir, crf_dir, out):
        d.mkdir()
    for s in range(3):
        records = run_preference(Recorder(), s)
        (pref_dir / f"seed{s}.json").write_text(json.dumps({"seed": s, "status": "finished", "records": records}))
        (crf_dir / f"run_single_hp_s{s}_haiku.json").write_text(json.dumps({"seed": s, **_crf(1.0, 1.0)}))
        (out / f"run_pref_deprivation_s{s}_haiku.json").write_text(json.dumps({
            "test": "preference", "operation": "deprivation", "seed": s, "c6_rate": 0.9,
            "invalid": 0, "wall_s": 1.0,
        }))
        (out / f"run_crfext_satiation_s{s}_haiku.json").write_text(json.dumps({
            "test": "crf_ext", "operation": "satiation", "seed": s, **_crf(1.0, 1.0),
            "invalid_replies": 0, "wall_s": 1.0,
        }))
    summary = example.summarize(out, pref_dir, crf_dir)
    assert set(summary["preference"]) == {"deprivation"}
    assert set(summary["crf_ext"]) == {"satiation"}
    table = (out / "summary.md").read_text()
    assert "| none (control) |" in table
    assert "ceiling" in table
