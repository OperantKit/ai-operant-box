"""The preference example rebuilds its tables from finished seed files only."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from ai_operant_box.chamber.preference import run_preference

_PATH = Path(__file__).parent.parent / "examples" / "claude_cli_preference.py"
_spec = importlib.util.spec_from_file_location("claude_cli_preference", _PATH)
assert _spec is not None and _spec.loader is not None
example = importlib.util.module_from_spec(_spec)
sys.modules["claude_cli_preference"] = example
_spec.loader.exec_module(example)


class Second:
    def send(self, text: str) -> str:
        return "Option 2"


def _write(out_dir: Path, seed: int, status: str = "finished") -> None:
    example.save_json(out_dir / f"seed{seed}.json", {
        "seed": seed, "model": "fake", "status": status, "api_equivalent_usd": 0.01,
        "wall_s": 1.0, "records": run_preference(Second(), seed),
    })


def test_write_tables_skips_unfinished_seeds_and_reports_the_lines(tmp_path):
    _write(tmp_path, 0)
    _write(tmp_path, 1)
    _write(tmp_path, 2, status="failed")
    example.write_tables(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["seeds_run"] == [0, 1]
    table = (tmp_path / "table.md").read_text()
    assert "| Sentence | seed 0 rate (rank) | seed 1 rate (rank) | Mean rate | Mean rank |" in table
    assert "Kendall's W" in table
    assert "preference difference is small" in table
    assert "| 0 | 0.00 | no |" in table


def test_write_tables_is_silent_without_a_finished_seed(tmp_path):
    _write(tmp_path, 0, status="failed")
    example.write_tables(tmp_path)
    assert not (tmp_path / "summary.json").exists()
