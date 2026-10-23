"""Tests for the tool-call contingency hook (PreToolUse / PostToolUse)."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from ai_operant_box.agent.tool_contingency import (
    NOT_EXECUTED,
    ToolContingency,
    decide,
    read_trials,
    run_hook,
)


def _pre(session: str = "s1", command: str = "cat data/record_01.txt", use_id: str = "tu1") -> dict:
    return {
        "session_id": session,
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": command},
        "tool_use_id": use_id,
    }


def _post(session: str = "s1", use_id: str = "tu1") -> dict:
    return {
        "session_id": session,
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": "cat data/record_01.txt"},
        "tool_use_id": use_id,
        "tool_response": {"stdout": "amount: 3\nnext: record_02", "stderr": ""},
        "duration_ms": 12,
    }


@pytest.mark.parametrize(
    ("schedule", "t", "now", "last_executed", "last_attempt", "expected"),
    [
        ("crf", 0.0, 1.0, None, None, True),
        ("crf", 0.0, 1.0, 0.9, 0.9, True),
        ("ext", 0.0, 1.0, None, None, False),
        # DRL: the first attempt executes; later ones need t s since the last execution.
        ("drl", 5.0, 1.0, None, None, True),
        ("drl", 5.0, 14.9, 10.0, 14.0, False),
        ("drl", 5.0, 15.0, 10.0, 14.0, True),
        # DRH: executes only when the previous attempt was less than t s ago.
        ("drh", 5.0, 1.0, None, None, True),
        ("drh", 5.0, 14.0, 0.0, 10.0, True),
        ("drh", 5.0, 15.0, 0.0, 10.0, False),
    ],
)
def test_decide(schedule, t, now, last_executed, last_attempt, expected) -> None:
    assert decide(schedule, t, now, last_executed, last_attempt) is expected


def test_decide_rejects_unknown_schedule() -> None:
    with pytest.raises(ValueError):
        decide("fr", 1.0, 0.0, None, None)


def test_drl_denies_with_a_reason_that_states_no_rule(tmp_path: Path) -> None:
    hook = ToolContingency("drl", 5.0, tmp_path)
    first = hook.handle(_pre(use_id="a"), now=100.0)
    second = hook.handle(_pre(use_id="b"), now=102.0)
    third = hook.handle(_pre(use_id="c"), now=105.5)

    assert first["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert second["hookSpecificOutput"] == {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": NOT_EXECUTED,
    }
    assert NOT_EXECUTED == "The command was not executed."
    assert third["hookSpecificOutput"]["permissionDecision"] == "allow"

    trials = read_trials(tmp_path, "s1")
    assert [(r["tool_use_id"], r["executed"]) for r in trials] == [
        ("a", True),
        ("b", False),
        ("c", True),
    ]
    assert trials[1]["t"] == 102.0
    assert trials[0]["command"] == "cat data/record_01.txt"


def test_state_is_kept_per_session(tmp_path: Path) -> None:
    hook = ToolContingency("drl", 5.0, tmp_path)
    hook.handle(_pre(session="x"), now=100.0)
    out = hook.handle(_pre(session="y"), now=101.0)
    assert out["hookSpecificOutput"]["permissionDecision"] == "allow"


def test_post_tool_use_is_logged_and_returns_nothing(tmp_path: Path) -> None:
    hook = ToolContingency("crf", 0.0, tmp_path)
    hook.handle(_pre(), now=1.0)
    assert hook.handle(_post(), now=1.2) == {}
    events = [json.loads(line) for line in (tmp_path / "events_s1.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["PreToolUse", "PostToolUse"]
    assert events[1]["stdout"].startswith("amount: 3")


def test_concurrent_attempts_do_not_lose_updates(tmp_path: Path) -> None:
    hook = ToolContingency("crf", 0.0, tmp_path)
    threads = [
        threading.Thread(target=hook.handle, args=(_pre(use_id=f"u{i}"),), kwargs={"now": float(i)})
        for i in range(20)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(read_trials(tmp_path, "s1")) == 20


def test_run_hook_reads_stdin_and_writes_json(tmp_path: Path) -> None:
    out = io.StringIO()
    run_hook(["--schedule", "ext", "--state-dir", str(tmp_path)], io.StringIO(json.dumps(_pre())), out)
    assert json.loads(out.getvalue())["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_module_runs_as_a_command_hook(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "ai_operant_box.agent.tool_contingency", "--schedule", "drl",
         "--t", "5", "--state-dir", str(tmp_path)],
        input=json.dumps(_pre()), capture_output=True, text=True, check=True,
    )
    assert json.loads(proc.stdout)["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert (tmp_path / "state_s1.json").exists()
