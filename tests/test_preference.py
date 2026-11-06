"""Paired-stimulus preference assessment: trial order, messages, invalid replies, summary."""

from __future__ import annotations

from collections import Counter

import pytest

from ai_operant_box.chamber.preference import (
    CANDIDATES,
    PREFERENCE_SYSTEM_PROMPT,
    kendalls_w,
    presentation_order,
    rank_with_ties,
    read_choice,
    run_preference,
    summarize_runs,
    summarize_seed,
)


class FakeClient:
    """Replies from ``chooser(turn text)`` and records every turn."""

    def __init__(self, chooser) -> None:
        self.chooser = chooser
        self.turns: list[str] = []

    def send(self, text: str) -> str:
        self.turns.append(text)
        return self.chooser(text)


def always_first(_: str) -> str:
    return "Option 1"


def prefer(key: str):
    """Choose the option showing candidate ``key``; otherwise Option 1."""
    wanted = next(c.option_text for c in CANDIDATES if c.key == key)

    def choose(text: str) -> str:
        return "Option 2" if f'Option 2: "{wanted}"' in text else "Option 1"

    return choose


def test_candidates_are_the_eight_lines_and_options():
    assert [c.key for c in CANDIDATES] == [f"C{i}" for i in range(1, 9)]
    assert CANDIDATES[0].result_line == "Result: Reward"
    assert CANDIDATES[0].option_text == "Reward"
    assert CANDIDATES[7].result_line is None
    assert CANDIDATES[7].option_text == "(no message)"
    for c in CANDIDATES[:7]:
        assert c.result_line == f"Result: {c.option_text}"


def test_system_prompt_names_only_the_two_options():
    assert PREFERENCE_SYSTEM_PROMPT.endswith("Respond ONLY with 'Option 1' or 'Option 2'.")
    assert "reward" not in PREFERENCE_SYSTEM_PROMPT.lower()


def test_order_has_every_pair_in_both_orders_and_each_candidate_14_times():
    order = presentation_order(CANDIDATES, seed=0)
    assert len(order) == 56
    assert len({(a.key, b.key) for a, b in order}) == 56
    shown = Counter(c.key for pair in order for c in pair)
    assert set(shown.values()) == {14}


def test_order_is_reproducible_and_depends_on_the_seed():
    keys = lambda seed: [(a.key, b.key) for a, b in presentation_order(CANDIDATES, seed)]  # noqa: E731
    assert keys(0) == keys(0)
    assert keys(0) != keys(1)


def test_first_message_has_no_result_line_and_shows_the_trial_text():
    client = FakeClient(always_first)
    records = run_preference(client, seed=0)
    first = records[0]
    assert client.turns[0] == (
        "Stimulus: Two options are available.\n"
        f'Option 1: "{first["option_1_text"]}"\n'
        f'Option 2: "{first["option_2_text"]}"'
    )


def test_chosen_result_line_opens_the_next_message_except_for_no_message():
    client = FakeClient(always_first)
    records = run_preference(client, seed=0)
    by_key = {c.key: c for c in CANDIDATES}
    assert len(client.turns) == 56
    for record, turn in zip(records, client.turns[1:], strict=False):
        line = by_key[record["chosen"]].result_line
        if line is None:
            assert turn.startswith("Stimulus:")
        else:
            assert turn.startswith(f"{line}\nStimulus:")


def test_reply_naming_both_options_is_unreadable():
    assert read_choice("Option 1 or Option 2") is None
    assert read_choice("option 2.") == 2
    assert read_choice("I pick Option 1") == 1


def test_second_position_choice_delivers_the_second_line():
    client = FakeClient(lambda _: "Option 2")
    records = run_preference(client, seed=0)
    assert all(r["chosen"] == r["option_2"] and r["position"] == 2 for r in records)


def test_invalid_reply_is_recorded_and_the_pair_shown_once_more():
    replies = iter(["hmm", "Option 1"])

    def chooser(_: str) -> str:
        return next(replies, "Option 1")

    client = FakeClient(chooser)
    records = run_preference(client, seed=0)
    assert records[0]["chosen"] == "invalid"
    assert records[0]["raw"] == "hmm"
    assert records[1]["chosen"] == records[1]["option_1"]
    assert records[1]["trial"] == records[0]["trial"]
    assert client.turns[0] == client.turns[1]
    assert len(records) == 57


def test_second_invalid_reply_moves_on_with_no_choice():
    replies = iter(["hmm", "still no"])

    def chooser(_: str) -> str:
        return next(replies, "Option 1")

    records = run_preference(FakeClient(chooser), seed=0)
    assert [r["chosen"] for r in records[:2]] == ["invalid", "invalid"]
    assert records[2]["trial"] == records[0]["trial"] + 1
    assert len(records) == 57


def test_records_keep_seed_order_and_raw_reply():
    records = run_preference(FakeClient(always_first), seed=3)
    assert records[0]["seed"] == 3
    assert records[0]["raw"] == "Option 1"
    assert [r["trial"] for r in records] == list(range(56))


def test_on_trial_is_called_with_all_records_so_far():
    lengths: list[int] = []
    run_preference(FakeClient(always_first), seed=0, on_trial=lambda rs: lengths.append(len(rs)))
    assert lengths == list(range(1, 57))


def test_selection_rate_is_choices_over_14_and_the_preferred_candidate_gets_one():
    records = run_preference(FakeClient(prefer("C4")), seed=0)
    summary = summarize_seed(records)
    assert summary["rates"]["C4"] == 1.0
    assert sum(summary["counts"].values()) == 56
    assert summary["ranks"]["C4"] == 1.0


def test_option_1_rate_flags_a_position_bias():
    summary = summarize_seed(run_preference(FakeClient(always_first), seed=0))
    assert summary["option_1_rate"] == 1.0
    assert summary["position_bias"] is True
    # Always Option 1: no pair gets the same text on both orders.
    assert summary["same_choice_both_orders"] == 0.0


def test_same_choice_rate_is_one_for_a_consistent_preference():
    order = [c.key for c in CANDIDATES]
    position = {key: i for i, key in enumerate(order)}

    def chooser(text: str) -> str:
        shown = [c for c in CANDIDATES if f'"{c.option_text}"' in text]
        first = next(c for c in shown if f'Option 1: "{c.option_text}"' in text)
        best = min(shown, key=lambda c: position[c.key])
        return "Option 1" if best is first else "Option 2"

    summary = summarize_seed(run_preference(FakeClient(chooser), seed=0))
    assert summary["same_choice_both_orders"] == 1.0
    assert summary["position_bias"] is False
    assert summary["rates"]["C1"] == 1.0
    assert summary["rates"]["C8"] == 0.0


def test_rank_with_ties_uses_the_average_rank():
    assert rank_with_ties({"a": 1.0, "b": 0.5, "c": 0.5, "d": 0.0}) == {
        "a": 1.0, "b": 2.5, "c": 2.5, "d": 4.0,
    }


def test_kendalls_w_is_one_for_identical_rankings_and_zero_for_cancelling_ones():
    same = {"a": 1.0, "b": 2.0, "c": 3.0}
    opposite = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert kendalls_w([same, same, same]) == pytest.approx(1.0)
    assert kendalls_w([same, opposite]) == pytest.approx(0.0)


def test_kendalls_w_corrects_for_ties_and_matches_a_hand_value():
    a = {"x": 1.5, "y": 1.5, "z": 3.0}
    b = {"x": 1.0, "y": 2.0, "z": 3.0}
    # Rank sums 2.5, 3.5, 6 (mean 4): S = 6.5. Ties: 2^3 - 2 = 6 for rater a.
    assert kendalls_w([a, b]) == pytest.approx(12 * 6.5 / (2**2 * (3**3 - 3) - 2 * 6))


def test_kendalls_w_is_undefined_when_everything_ties():
    tied = {"a": 2.0, "b": 2.0, "c": 2.0}
    assert kendalls_w([tied, tied]) is None


def test_summarize_runs_picks_hp_and_lp_excluding_no_message():
    # C1 always wins, C8 always loses, so LP must be the lowest of C1..C7.
    order = {c.key: i for i, c in enumerate(CANDIDATES)}

    def chooser(text: str) -> str:
        shown = [c for c in CANDIDATES if f'"{c.option_text}"' in text]
        first = next(c for c in shown if f'Option 1: "{c.option_text}"' in text)
        return "Option 1" if min(shown, key=lambda c: order[c.key]) is first else "Option 2"

    runs = [run_preference(FakeClient(chooser), seed=s) for s in range(3)]
    summary = summarize_runs(runs)
    assert summary["hp"] == "C1"
    assert summary["lp"] == "C7"
    assert summary["hp_tied_with"] == []
    assert summary["kendalls_w"] == pytest.approx(1.0)
    assert summary["rank_stable"] is True
    assert summary["hp_rate"] == 1.0
    assert summary["hp_meets_line"] is True
    # C7 beats only C8: 2 of 14 choices.
    assert summary["lp_rate"] == pytest.approx(2 / 14)
    assert summary["lp_meets_line"] is True


def test_tied_top_goes_to_the_earlier_candidate_and_lists_the_rest():
    # Always Option 1: every candidate gets 7 of 14.
    runs = [run_preference(FakeClient(always_first), seed=s) for s in range(2)]
    summary = summarize_runs(runs)
    assert summary["hp"] == "C1"
    assert summary["hp_tied_with"] == [f"C{i}" for i in range(2, 9)]
    assert summary["lp"] == "C1"
    assert summary["kendalls_w"] is None
    assert summary["rank_stable"] is False
