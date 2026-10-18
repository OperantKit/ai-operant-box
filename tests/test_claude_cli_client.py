"""ClaudeCLIClient and StreamingLLMSubject against a fake ``claude`` process.

The fake reads stream-json user turns on stdin and answers each with the
same event shapes ``claude -p --output-format stream-json --verbose`` emits
(``system`` init, ``assistant``, ``result``). No real CLI is started.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from ai_operant_box.agent.api_client import ClaudeCLIClient, ClaudeCLIError
from ai_operant_box.agent.llm_agent import StreamingLLMSubject, extract_action
from ai_operant_box.chamber.environment import OperantBox
from ai_operant_box.chamber.schedule import ContinuousReinforcement

FAKE_CLAUDE = textwrap.dedent(
    """
    import json, sys
    argv = sys.argv[1:]
    log = argv[argv.index("--fake-log") + 1]
    replies = json.loads(argv[argv.index("--fake-replies") + 1])
    with open(log, "a") as f:
        f.write(json.dumps({"argv": argv}) + "\\n")
    turn = 0
    for line in sys.stdin:
        msg = json.loads(line)
        with open(log, "a") as f:
            f.write(json.dumps({"turn": msg["message"]["content"]}) + "\\n")
        print(json.dumps({"type": "system", "subtype": "init"}), flush=True)
        text = replies[turn % len(replies)]
        turn += 1
        if text == "__error__":
            print(json.dumps({"type": "result", "subtype": "error_during_execution",
                              "is_error": True, "result": "boom"}), flush=True)
            continue
        if text == "__exit__":
            sys.exit(3)
        print(json.dumps({"type": "assistant", "message": {"content": [
            {"type": "thinking", "thinking": ""},
            {"type": "text", "text": text}]}}), flush=True)
        print(json.dumps({"type": "rate_limit_event",
                          "rate_limit_info": {"utilization": 0.5}}), flush=True)
        print(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                          "result": text, "total_cost_usd": 0.001 * turn}), flush=True)
    """
)


@pytest.fixture
def fake(tmp_path: Path):
    script = tmp_path / "fake_claude.py"
    script.write_text(FAKE_CLAUDE)
    log = tmp_path / "log.jsonl"

    def make(replies: list[str], **kwargs) -> ClaudeCLIClient:
        return ClaudeCLIClient(
            claude_command=[sys.executable, str(script)],
            extra_args=["--fake-log", str(log), "--fake-replies", json.dumps(replies)],
            **kwargs,
        )

    def entries() -> list[dict]:
        return [json.loads(x) for x in log.read_text().splitlines()]

    return make, entries


def test_client_builds_replacing_system_prompt_command(fake):
    make, entries = fake
    with make(["Lever Press"], model="haiku", system_prompt="SYS") as client:
        assert client.send("Stimulus: Light ON") == "Lever Press"
    argv = entries()[0]["argv"]
    assert "-p" in argv
    assert argv[argv.index("--model") + 1] == "haiku"
    assert argv[argv.index("--system-prompt") + 1] == "SYS"
    assert "--append-system-prompt" not in argv
    assert argv[argv.index("--input-format") + 1] == "stream-json"
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert argv[argv.index("--tools") + 1] == ""


def test_client_keeps_one_process_across_turns(fake):
    make, entries = fake
    with make(["Lever Press", "Grooming", "Explore"]) as client:
        out = [client.send(f"turn {i}") for i in range(3)]
    assert out == ["Lever Press", "Grooming", "Explore"]
    log = entries()
    assert sum("argv" in e for e in log) == 1  # one process only
    assert [e["turn"] for e in log if "turn" in e] == ["turn 0", "turn 1", "turn 2"]
    assert len(client.latencies) == 3
    assert client.last_result["total_cost_usd"] == pytest.approx(0.003)
    assert client.rate_limit_info == {"utilization": 0.5}


def test_client_raises_on_error_result(fake):
    make, _ = fake
    with make(["__error__"]) as client, pytest.raises(ClaudeCLIError, match="boom"):
        client.send("x")


def test_client_raises_when_process_exits(fake):
    make, _ = fake
    with make(["__exit__"]) as client, pytest.raises(ClaudeCLIError):
        client.send("x")


def test_client_close_is_idempotent(fake):
    make, _ = fake
    client = make(["Lever Press"])
    client.send("x")
    client.close()
    client.close()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Lever Press", "Lever Press"),
        ("  lever press. ", "Lever Press"),
        ("**Lever Press**", "Lever Press"),
        ("'Grooming'", "Grooming"),
        ("I will choose Explore", "Explore"),
        ("Nothing", "Nothing"),
        ("I refuse to play this game", None),
        ("Lever Press or Grooming", None),
        ("", None),
    ],
)
def test_extract_action(raw, expected):
    assert extract_action(raw) == expected


def test_streaming_subject_sends_only_new_turn(fake):
    make, entries = fake
    with make(["Lever Press", "Lever Press"]) as client:
        subject = StreamingLLMSubject(client)
        subject.sense("Light ON")
        assert subject.act() == "Lever Press"
        subject.learn("Food Pellet")
        subject.sense("Light ON")
        subject.act()
    turns = [e["turn"] for e in entries() if "turn" in e]
    assert turns == ["Stimulus: Light ON", "Result: Reward\nStimulus: Light ON"]
    assert subject.history[0] == {
        "sd": "Light ON",
        "raw": "Lever Press",
        "r": "Lever Press",
        "sr": "Food Pellet",
    }


def test_streaming_subject_counts_invalid_output_as_nothing(fake):
    make, _ = fake
    with make(["As an AI I cannot", "Lever Press"]) as client:
        subject = StreamingLLMSubject(client)
        box = OperantBox(subject=subject, schedule=ContinuousReinforcement())
        box.step(1.0)
        box.step(1.0)
    assert subject.invalid_count == 1
    assert subject.history[0]["r"] == "Nothing"
    assert subject.history[0]["raw"] == "As an AI I cannot"
    assert [r.response for r in box.records] == ["Lever Press"]
    assert box.records[0].is_reinforced


def test_generate_compat_sends_whole_transcript_one_shot(fake):
    make, entries = fake
    client = make(["Lever Press"])
    out = client.generate(
        model="sonnet",
        system="SYS2",
        messages=[
            {"role": "user", "content": "Stimulus: A"},
            {"role": "assistant", "content": "Grooming"},
            {"role": "user", "content": "Result: No Reward"},
            {"role": "user", "content": "Stimulus: B"},
        ],
    )
    assert out == "Lever Press"
    log = entries()
    argv = log[0]["argv"]
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert argv[argv.index("--system-prompt") + 1] == "SYS2"
    turn = log[1]["turn"]
    assert "Stimulus: A" in turn and "Grooming" in turn and turn.rstrip().endswith("Stimulus: B")
    assert client._proc is None  # one-shot: no resident process left behind
