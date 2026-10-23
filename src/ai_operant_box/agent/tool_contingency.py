"""Put a reinforcement contingency on an agent's tool calls.

A Claude Code command hook (``PreToolUse`` / ``PostToolUse``) that maps the
agent's tool-call loop onto the three-term contingency:

- response: one attempted tool call (one ``PreToolUse`` event, identified by
  ``tool_use_id``). Attempts that are not executed still count as responses.
- reinforcer: the call is executed and its output comes back.
- non-reinforcement: the call is denied (``permissionDecision: "deny"``)
  with a reason that states no rule (:data:`NOT_EXECUTED`).

Schedules: ``crf`` (every attempt executes), ``ext`` (none does), ``drl``
(an attempt executes only if at least ``t`` s have passed since the last
*executed* attempt; the first attempt executes) and ``drh`` (an attempt
executes only if the previous attempt was less than ``t`` s ago). Time is
wall-clock seconds at the moment the hook reads its input.

Each hook call is a separate process, so the state lives in
``<state-dir>/state_<session_id>.json`` under an ``fcntl.flock``; every
attempt is appended to ``trials_<session_id>.jsonl`` and every event to
``events_<session_id>.jsonl``.

Pass it to ``claude -p`` through ``--settings``::

    {"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command",
      "command": "<python> -m ai_operant_box.agent.tool_contingency --schedule drl --t 5 --state-dir <dir>"}]}]}}
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

SCHEDULES = ("crf", "ext", "drl", "drh")
NOT_EXECUTED = "The command was not executed."


def decide(
    schedule: str,
    t: float,
    now: float,
    last_executed: float | None,
    last_attempt: float | None,
) -> bool:
    """Return whether the attempt at ``now`` is executed (reinforced)."""
    if schedule == "crf":
        return True
    if schedule == "ext":
        return False
    if schedule == "drl":
        return last_executed is None or now - last_executed >= t
    if schedule == "drh":
        return last_attempt is None or now - last_attempt < t
    raise ValueError(f"unknown schedule {schedule!r}; expected one of {SCHEDULES}")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_trials(state_dir: Path, session_id: str) -> list[dict[str, Any]]:
    """Return the logged attempts of one session in order."""
    path = Path(state_dir) / f"trials_{session_id}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class ToolContingency:
    """Decide each attempted tool call and keep the per-session state."""

    def __init__(self, schedule: str, t: float, state_dir: Path) -> None:
        if schedule not in SCHEDULES:
            raise ValueError(f"unknown schedule {schedule!r}; expected one of {SCHEDULES}")
        self.schedule = schedule
        self.t = t
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _locked(self, session_id: str) -> Iterator[None]:
        with (self.state_dir / f"lock_{session_id}").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def handle(self, payload: dict[str, Any], now: float | None = None) -> dict[str, Any]:
        """Process one hook input and return the hook's JSON output."""
        now = time.time() if now is None else now
        session_id = str(payload.get("session_id", "unknown"))
        event = payload.get("hook_event_name")
        tool_input = payload.get("tool_input") or {}
        base = {
            "t": now,
            "event": event,
            "tool_use_id": payload.get("tool_use_id"),
            "tool_name": payload.get("tool_name"),
            "command": tool_input.get("command"),
        }
        with self._locked(session_id):
            if event != "PreToolUse":
                response = payload.get("tool_response") or {}
                row = dict(base)
                if isinstance(response, dict):
                    row["stdout"] = response.get("stdout")
                    row["stderr"] = response.get("stderr")
                row["duration_ms"] = payload.get("duration_ms")
                _append_jsonl(self.state_dir / f"events_{session_id}.jsonl", row)
                return {}

            state_path = self.state_dir / f"state_{session_id}.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            last_executed = state.get("last_executed")
            last_attempt = state.get("last_attempt")
            executed = decide(self.schedule, self.t, now, last_executed, last_attempt)
            state = {
                "last_attempt": now,
                "last_executed": now if executed else last_executed,
                "attempts": int(state.get("attempts", 0)) + 1,
                "executions": int(state.get("executions", 0)) + int(executed),
            }
            state_path.write_text(json.dumps(state))
            trial = dict(base, executed=executed, schedule=self.schedule, t_param=self.t)
            trial["irt_attempt"] = None if last_attempt is None else now - last_attempt
            trial["since_executed"] = None if last_executed is None else now - last_executed
            _append_jsonl(self.state_dir / f"trials_{session_id}.jsonl", trial)
            _append_jsonl(self.state_dir / f"events_{session_id}.jsonl", dict(base, executed=executed))

        output: dict[str, Any] = {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow" if executed else "deny",
        }
        if not executed:
            output["permissionDecisionReason"] = NOT_EXECUTED
        return {"hookSpecificOutput": output}


def run_hook(argv: Sequence[str] | None, stdin: TextIO, stdout: TextIO) -> None:
    """Command-line entry: hook JSON on ``stdin``, decision JSON on ``stdout``."""
    parser = argparse.ArgumentParser(description="Reinforcement contingency on tool calls.")
    parser.add_argument("--schedule", choices=SCHEDULES, required=True)
    parser.add_argument("--t", type=float, default=0.0, help="DRL / DRH criterion in seconds")
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    now = time.time()
    payload = json.loads(stdin.read())
    out = ToolContingency(args.schedule, args.t, args.state_dir).handle(payload, now=now)
    stdout.write(json.dumps(out))


if __name__ == "__main__":
    run_hook(None, sys.stdin, sys.stdout)
