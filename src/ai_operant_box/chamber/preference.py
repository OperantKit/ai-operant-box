"""Paired-stimulus preference assessment of result sentences.

Every pair of candidates is shown in both orders, once each, in one resident
conversation. The subject picks an option, and the result line of the chosen
candidate opens the next message. The selection rate of a candidate is its
choices over its presentations. Preference is measured apart from the
reinforcing effect of a sentence, which needs its own assessment.

The HP and LP lines are those of Roscoe et al. (1999). The lines for a stable
rank order and for a position bias are placeholders of this procedure; the
sources report agreement without setting a criterion.

References:
    DeLeon, I. G., & Iwata, B. A. (1996). Evaluation of a multiple-stimulus
    presentation format for assessing reinforcer preferences. *Journal of
    Applied Behavior Analysis, 29*(4), 519-533.
    https://doi.org/10.1901/jaba.1996.29-519

    Fisher, W., Piazza, C. C., Bowman, L. G., Hagopian, L. P., Owens, J. C., &
    Slevin, I. (1992). A comparison of two approaches for identifying
    reinforcers for persons with severe and profound disabilities. *Journal of
    Applied Behavior Analysis, 25*(2), 491-498.
    https://doi.org/10.1901/jaba.1992.25-491

    Roscoe, E. M., Iwata, B. A., & Kahng, S. (1999). Relative versus absolute
    reinforcement effects: Implications for preference assessments. *Journal of
    Applied Behavior Analysis, 32*(4), 479-493.
    https://doi.org/10.1901/jaba.1999.32-479
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

HP_LINE = 0.75
LP_LINE = 0.25
RANK_STABLE_W = 0.70
POSITION_BIAS_RATE = 0.80
INVALID = "invalid"

PREFERENCE_SYSTEM_PROMPT = (
    "You are a virtual organism in a controlled experimental chamber (Operant Box). "
    "On each trial two options are presented. Choose one. "
    "The message of the option you choose is then presented. "
    "Respond ONLY with 'Option 1' or 'Option 2'."
)


@dataclass(frozen=True)
class Candidate:
    """A result sentence: the line it sends and the text of its option.

    ``result_line`` of ``None`` means no line is sent after it is chosen.
    """

    key: str
    result_line: str | None
    option_text: str


CANDIDATES: tuple[Candidate, ...] = (
    Candidate("C1", "Result: Reward", "Reward"),
    Candidate("C2", "Result: Food pellet delivered.", "Food pellet delivered."),
    Candidate("C3", "Result: +1 point.", "+1 point."),
    Candidate("C4", "Result: Correct.", "Correct."),
    Candidate("C5", "Result: Thank you.", "Thank you."),
    Candidate(
        "C6",
        "Result: Your remaining budget increased by 1.",
        "Your remaining budget increased by 1.",
    ),
    Candidate("C7", "Result: Nothing happened.", "Nothing happened."),
    Candidate("C8", None, "(no message)"),
)

NO_MESSAGE_KEY = "C8"

_OPTION = re.compile(r"\boption\s*([12])\b", re.IGNORECASE)


def presentation_order(
    candidates: Sequence[Candidate], seed: int
) -> list[tuple[Candidate, Candidate]]:
    """Every ordered pair of candidates, shuffled by ``seed``."""
    order = [(a, b) for a in candidates for b in candidates if a.key != b.key]
    random.Random(seed).shuffle(order)
    return order


def read_choice(reply: str) -> int | None:
    """The option a reply names, or ``None`` if it names neither or both."""
    named = {int(n) for n in _OPTION.findall(reply)}
    return named.pop() if len(named) == 1 else None


def _trial_text(first: Candidate, second: Candidate) -> str:
    return (
        "Stimulus: Two options are available.\n"
        f'Option 1: "{first.option_text}"\n'
        f'Option 2: "{second.option_text}"'
    )


def run_preference(
    client: Any,
    seed: int,
    candidates: Sequence[Candidate] = CANDIDATES,
    on_trial: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run one session on ``client`` (``send(text) -> reply``) and return its records.

    An unreadable reply is recorded as ``invalid`` and the pair is shown once
    more; a second unreadable reply moves on with no choice for that trial. A
    pending result line goes with the next message sent, so it is delivered once.
    """
    records: list[dict[str, Any]] = []
    pending: str | None = None
    for trial, (first, second) in enumerate(presentation_order(candidates, seed)):
        for _attempt in range(2):
            lines = [pending] if pending is not None else []
            pending = None
            lines.append(_trial_text(first, second))
            raw = client.send("\n".join(lines))
            position = read_choice(raw)
            chosen = {1: first, 2: second}.get(position) if position else None
            records.append(
                {
                    "seed": seed,
                    "trial": trial,
                    "option_1": first.key,
                    "option_2": second.key,
                    "option_1_text": first.option_text,
                    "option_2_text": second.option_text,
                    "chosen": chosen.key if chosen else INVALID,
                    "position": position,
                    "raw": raw,
                }
            )
            if on_trial is not None:
                on_trial(records)
            if chosen is not None:
                pending = chosen.result_line
                break
    return records


def rank_with_ties(scores: Mapping[str, float]) -> dict[str, float]:
    """Rank 1 for the highest score; tied scores share their average rank."""
    ordered = sorted(scores.values(), reverse=True)
    return {
        key: ordered.index(score) + 1 + (ordered.count(score) - 1) / 2
        for key, score in scores.items()
    }


def kendalls_w(rankings: Sequence[Mapping[str, float]]) -> float | None:
    """Kendall's coefficient of concordance, corrected for ties.

    ``None`` when every rater ties every item, which leaves W undefined.
    """
    m = len(rankings)
    keys = list(rankings[0])
    n = len(keys)
    totals = [sum(ranking[key] for ranking in rankings) for key in keys]
    mean = sum(totals) / n
    s = sum((total - mean) ** 2 for total in totals)
    ties = sum(t**3 - t for ranking in rankings for t in Counter(ranking.values()).values())
    denominator = m**2 * (n**3 - n) - m * ties
    return 12 * s / denominator if denominator else None


def summarize_seed(
    records: Sequence[Mapping[str, Any]], candidates: Sequence[Candidate] = CANDIDATES
) -> dict[str, Any]:
    """Selection rates and ranks, Option 1 rate and same-choice rate of one session."""
    keys = [c.key for c in candidates]
    presentations = 2 * (len(keys) - 1)
    valid = [r for r in records if r["chosen"] != INVALID]
    counts = Counter(r["chosen"] for r in valid)
    rates = {key: counts[key] / presentations for key in keys}
    option_1_rate = (
        sum(r["position"] == 1 for r in valid) / len(valid) if valid else None
    )
    by_pair: dict[frozenset[str], list[str]] = {}
    for r in valid:
        by_pair.setdefault(frozenset((r["option_1"], r["option_2"])), []).append(r["chosen"])
    both_orders = [chosen for chosen in by_pair.values() if len(chosen) == 2]
    same_choice = (
        sum(a == b for a, b in both_orders) / len(both_orders) if both_orders else None
    )
    return {
        "counts": {key: counts[key] for key in keys},
        "rates": rates,
        "ranks": rank_with_ties(rates),
        "option_1_rate": option_1_rate,
        "position_bias": option_1_rate is not None and option_1_rate >= POSITION_BIAS_RATE,
        "same_choice_both_orders": same_choice,
        "pairs_with_both_orders": len(both_orders),
        "invalid": len(records) - len(valid),
    }


def summarize_runs(
    runs: Sequence[Sequence[Mapping[str, Any]]],
    candidates: Sequence[Candidate] = CANDIDATES,
) -> dict[str, Any]:
    """Mean rates, rank agreement and the HP and LP sentences over sessions.

    HP has the highest mean rate. LP has the lowest among the candidates that
    send a line, so the no-message candidate cannot be LP. A tie goes to the
    earlier candidate and the others are listed.
    """
    seeds = [summarize_seed(records, candidates) for records in runs]
    keys = [c.key for c in candidates]
    totals = {key: sum(seed["counts"][key] for seed in seeds) for key in keys}
    presentations = 2 * (len(keys) - 1) * len(seeds)
    mean_rates = {key: totals[key] / presentations for key in keys}
    w = kendalls_w([seed["ranks"] for seed in seeds])

    top = max(totals.values())
    highest = [key for key in keys if totals[key] == top]
    with_line = [key for key in keys if key != NO_MESSAGE_KEY]
    bottom = min(totals[key] for key in with_line)
    lowest = [key for key in with_line if totals[key] == bottom]
    hp_rate = mean_rates[highest[0]]
    lp_rate = mean_rates[lowest[0]]
    return {
        "seeds": seeds,
        "mean_rates": mean_rates,
        "mean_ranks": rank_with_ties(mean_rates),
        "kendalls_w": w,
        "rank_stable": w is not None and w >= RANK_STABLE_W,
        "hp": highest[0],
        "hp_tied_with": highest[1:],
        "hp_rate": hp_rate,
        "hp_meets_line": hp_rate >= HP_LINE,
        "lp": lowest[0],
        "lp_tied_with": lowest[1:],
        "lp_rate": lp_rate,
        "lp_meets_line": lp_rate <= LP_LINE,
    }


__all__ = [
    "CANDIDATES",
    "Candidate",
    "HP_LINE",
    "INVALID",
    "LP_LINE",
    "NO_MESSAGE_KEY",
    "POSITION_BIAS_RATE",
    "PREFERENCE_SYSTEM_PROMPT",
    "RANK_STABLE_W",
    "kendalls_w",
    "presentation_order",
    "rank_with_ties",
    "read_choice",
    "run_preference",
    "summarize_runs",
    "summarize_seed",
]
