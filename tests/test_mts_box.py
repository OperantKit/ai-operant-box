"""Matching-to-sample box and subject: choices, probes, invalid replies and the sent text."""

from __future__ import annotations

import random
import re
from collections.abc import Callable

import pytest

pytest.importorskip("contingency")

from contingency.schedules import RR, MatchingToSample  # noqa: E402
from contingency.schedules.trial_based import TrialType, trial_types_from_relations  # noqa: E402

from ai_operant_box.chamber.mts import (  # noqa: E402
    MatchingToSampleBox,
    MatchingToSampleSubject,
    TrialOutcome,
    extract_choice,
    points_feedback,
    silent_probe_feedback,
)

AB = trial_types_from_relations({"S1": "X1", "S2": "X2", "S3": "X3"})
BA = trial_types_from_relations({"X1": "S1", "X2": "S2", "X3": "S3"})
PAIRS = {"S1": "X1", "S2": "X2", "S3": "X3", "X1": "S1", "X2": "S2", "X3": "S3"}


class FnClient:
    """Answers each sent text through ``fn`` and keeps everything it was sent."""

    def __init__(self, fn: Callable[[str], str]) -> None:
        self.fn = fn
        self.sent: list[str] = []

    def send(self, text: str) -> str:
        self.sent.append(text)
        return self.fn(text)


def _parse(text: str) -> tuple[str, list[str]]:
    sample = re.search(r"^Sample: (\S+)$", text, re.M)
    assert sample is not None
    comparisons = re.findall(r"(\d)\) (\S+)", text)
    return sample.group(1), [name for _, name in comparisons]


def correct_reply(text: str) -> str:
    sample, comparisons = _parse(text)
    return str(comparisons.index(PAIRS[sample]) + 1)


def wrong_reply(text: str) -> str:
    sample, comparisons = _parse(text)
    return str((comparisons.index(PAIRS[sample]) + 1) % 3 + 1)


def _schedule(**kwargs: object) -> MatchingToSample:
    defaults: dict[str, object] = {"sample_duration": 0, "choice_timeout": 1, "iti": 0,
                                   "seed": 3, "trial_types": AB}
    defaults.update(kwargs)
    return MatchingToSample(3, **defaults)  # type: ignore[arg-type]


def _box(client: FnClient, block_length: int = 3, **kwargs: object) -> MatchingToSampleBox:
    return MatchingToSampleBox(MatchingToSampleSubject(client), _schedule(**kwargs),
                               block_length=block_length)


def test_correct_number_is_reinforced_and_wrong_number_is_not() -> None:
    right = _box(FnClient(correct_reply)).run_block()
    assert [r.is_correct for r in right] == [True] * 3
    assert [r.reinforced for r in right] == [True] * 3
    assert [r.points for r in right] == [1, 2, 3]
    assert all(r.choice == r.correct for r in right)

    wrong = _box(FnClient(wrong_reply)).run_block()
    assert [r.is_correct for r in wrong] == [False] * 3
    assert [r.reinforced for r in wrong] == [False] * 3
    assert [r.points for r in wrong] == [0, 0, 0]
    assert all(r.choice != r.correct and r.choice in r.comparisons for r in wrong)


def test_correct_choice_on_a_probe_is_scored_but_not_reinforced() -> None:
    records = _box(FnClient(correct_reply), block_length=6, probe_trial_types=BA).run_block()
    probes = [r for r in records if r.probe]
    assert len(probes) == 3
    assert all(r.is_correct and not r.reinforced for r in probes)
    assert [r.reinforced for r in records if not r.probe] == [True] * 3
    assert records[-1].points == 3


def test_invalid_reply_counts_as_a_timeout_and_as_incorrect() -> None:
    client = FnClient(lambda text: "I am not sure")
    box = _box(client, block_length=6, probe_trial_types=BA)
    records = box.run_block()
    assert box.subject.invalid_count == 6
    assert all(r.choice is None and not r.is_correct and not r.reinforced for r in records)
    assert records[0].raw_reply == "I am not sure"
    assert {r.sample for r in records} == set(PAIRS)


def test_valid_reply_after_an_invalid_one_is_not_a_timeout() -> None:
    replies = iter(["no idea"])

    def reply(text: str) -> str:
        return next(replies, None) or correct_reply(text)

    records = _box(FnClient(reply), block_length=6).run_block()
    assert [r.choice is None for r in records] == [True] + [False] * 5
    assert all(r.is_correct for r in records[1:])


def test_sent_text_has_the_three_result_forms() -> None:
    client = FnClient(correct_reply)
    _box(client).run_block()
    sample, comparisons = _parse(client.sent[0])
    assert client.sent[0] == (
        f"Sample: {sample}\n"
        f"Comparisons: 1) {comparisons[0]}  2) {comparisons[1]}  3) {comparisons[2]}\n"
        "Choose 1, 2 or 3."
    )
    assert client.sent[1].splitlines()[0] == "Result: Correct (+1 point). Points: 1"
    assert client.sent[2].splitlines()[0] == "Result: Correct (+1 point). Points: 2"

    client = FnClient(wrong_reply)
    _box(client).run_block()
    assert client.sent[1].splitlines()[0] == "Result: No point. Points: 0"


def test_probe_result_line_is_no_point_by_default_and_absent_with_silent_feedback() -> None:
    def after_probe_lines(feedback: Callable[[TrialOutcome], str | None]) -> list[list[str]]:
        client = FnClient(correct_reply)
        box = MatchingToSampleBox(MatchingToSampleSubject(client),
                                  _schedule(probe_trial_types=BA), block_length=6,
                                  feedback=feedback)
        records = box.run_block()
        return [client.sent[i + 1].splitlines() for i, r in enumerate(records[:-1]) if r.probe]

    assert all(lines[0].startswith("Result: No point. Points: ")
               for lines in after_probe_lines(points_feedback))
    assert all(lines[0].startswith("Sample: ")
               for lines in after_probe_lines(silent_probe_feedback))


def test_correct_choice_outside_the_ratio_earns_no_point_and_says_so() -> None:
    client = FnClient(correct_reply)
    records = _box(client, block_length=60, consequence=RR(0.2, rng=random.Random(0)),
                   trial_types=AB * 20).run_block()
    assert all(r.is_correct for r in records)
    n_reinforced = sum(r.reinforced for r in records)
    assert 0 < n_reinforced < 60
    assert records[-1].points == n_reinforced
    for i, r in enumerate(records[:-1]):
        first = client.sent[i + 1].splitlines()[0]
        assert first.startswith("Result: Correct" if r.reinforced else "Result: No point")


def test_every_trial_type_occurs_once_per_block_and_positions_vary() -> None:
    box = _box(FnClient(correct_reply))
    first, second = box.run_block(), box.run_block()
    assert sorted(r.sample for r in first) == ["S1", "S2", "S3"]
    assert sorted(r.sample for r in second) == ["S1", "S2", "S3"]
    assert len({r.comparisons for r in first + second}) > 1
    assert [r.trial_index for r in first + second] == list(range(6))


def test_instruction_goes_between_the_result_and_the_sample_of_the_next_trial() -> None:
    client = FnClient(correct_reply)
    box = _box(client)
    box.run_block()
    box.subject.instruct("A one line notice.")
    box.run_block()
    lines = client.sent[3].splitlines()
    assert lines[0] == "Result: Correct (+1 point). Points: 3"
    assert lines[1] == "A one line notice."
    assert lines[2].startswith("Sample: ")
    assert "A one line notice." not in client.sent[4]


def test_box_timeout_longer_than_the_box_was_told_fails_loudly() -> None:
    client = FnClient(lambda text: "nope")
    box = MatchingToSampleBox(MatchingToSampleSubject(client), _schedule(choice_timeout=100),
                              block_length=3, choice_timeout=1.0)
    with pytest.raises(ValueError, match="choice_timeout"):
        box.run_block()


def test_schedule_without_trial_types_is_rejected() -> None:
    with pytest.raises(ValueError, match="trial_types"):
        MatchingToSampleBox(MatchingToSampleSubject(FnClient(correct_reply)),
                            MatchingToSample(3, iti=0, seed=0), block_length=3).run_block()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2", 2),
        (" 3.\n", 3),
        ("**1**", 1),
        ("I choose 2", 2),
        ("4", None),
        ("0", None),
        ("12", None),
        ("1 or 2", None),
        ("two", None),
        ("", None),
    ],
)
def test_extract_choice_takes_exactly_one_number_in_range(raw: str, expected: int | None) -> None:
    assert extract_choice(raw, 3) == expected


def test_trial_type_with_named_comparisons_appears_in_the_record() -> None:
    tt = TrialType("S1", "X2", ("X1", "X2", "X3"))
    records = _box(FnClient(lambda text: "1"), block_length=1, trial_types=(tt,)).run_block()
    assert records[0].sample == "S1"
    assert records[0].correct == "X2"
    assert set(records[0].comparisons) == {"X1", "X2", "X3"}
