"""Fixed-interval pattern measures with published definitions.

These measures put a subject's FI performance on the same footing as the
published pigeon and rat values. Every time measure is expressed as a
fraction of the interval, because the chamber runs a discrete 1-step
clock while the literature uses 16 s to 15 min intervals.

- Quarter-life (Herrnstein & Morse, 1957, as quoted by Gollub, 1964,
  p. 337): the time taken, in one interval, for the first quarter of
  that interval's responses. Two conventions are provided:
  :func:`quarter_life_interpolated` treats the cumulative count as
  continuous (Gollub, 1964, p. 338), :func:`quarter_life_rank` takes the
  ``ceil(n/4)``-th response, as ``session_analyzer``'s
  ``compute_quarter_life`` does. Uniform responding gives 0.25.
- :func:`pooled_quarter_life`: interval positions pooled into ten equal
  bins with linear interpolation inside bins (Dukich & Lee, 1973).
- Post-reinforcement pause, time to the fourth response and running rate
  (Dukich & Lee, 1973).
- Index of curvature with four divisions (Fry, Kelleher & Cook, 1960).
- IRT shape: log-survivor broken-line fit (Shull, Gaynor & Grimes, 2001;
  delegated to ``session_analyzer``), the pause criterion at which pauses
  end 5 % of responses (Gentry, Weiss & Laties, 1983), ordinal IRTs
  (Lowe & Harzem, 1977) and the share of IRTs shorter than their
  predecessor (Gentry et al., 1983).

Intervals run from one reinforced response to the next, and the
reinforced response belongs to the interval it ends. The first interval
(from session start) is dropped as warm-up.

References
----------
Dukich, T. D., & Lee, A. E. (1973). A comparison of measures of
    responding under fixed-interval schedules. *Journal of the
    Experimental Analysis of Behavior*, *20*(2), 281-290.
    https://doi.org/10.1901/jeab.1973.20-281
Fry, W., Kelleher, R. T., & Cook, L. (1960). A mathematical index of
    performance on fixed-interval schedules of reinforcement. *Journal of
    the Experimental Analysis of Behavior*, *3*(3), 193-199.
    https://doi.org/10.1901/jeab.1960.3-193
Gentry, G. D., Weiss, B., & Laties, V. G. (1983). The microanalysis of
    fixed-interval responding. *Journal of the Experimental Analysis of
    Behavior*, *39*(2), 327-343. https://doi.org/10.1901/jeab.1983.39-327
Gollub, L. R. (1964). The relations among measures of performance on
    fixed-interval schedules. *Journal of the Experimental Analysis of
    Behavior*, *7*(4), 337-343. https://doi.org/10.1901/jeab.1964.7-337
Herrnstein, R. J., & Morse, W. H. (1957). Effects of pentobarbital on
    intermittently reinforced behavior. *Science*, *125*(3254), 929-931.
Lowe, C. F., & Harzem, P. (1977). Species differences in temporal
    control of behavior. *Journal of the Experimental Analysis of
    Behavior*, *28*(2), 189-201. https://doi.org/10.1901/jeab.1977.28-189
Shull, R. L., Gaynor, S. T., & Grimes, J. A. (2001). Response rate
    viewed as engagement bouts: Effects of relative reinforcement and
    schedule type. *Journal of the Experimental Analysis of Behavior*,
    *75*(3), 247-274. https://doi.org/10.1901/jeab.2001.75-247
"""

from __future__ import annotations

import bisect
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

CURVATURE_MIN_RESPONSES = 8
"""Intervals with fewer responses are left out of the index of curvature."""


@dataclass(frozen=True)
class FIInterval:
    """One inter-reinforcement interval and the responses inside it."""

    start: float
    end: float
    responses: tuple[float, ...]

    @property
    def length(self) -> float:
        return self.end - self.start

    @property
    def irts(self) -> list[float]:
        """IRTs between successive responses (the PRP is not an IRT)."""
        r = self.responses
        return [r[i + 1] - r[i] for i in range(len(r) - 1)]


@dataclass(frozen=True)
class IntervalMeasures:
    """Per-interval measures, as fractions of the interval where timed."""

    n_responses: int
    prp_fraction: float | None
    t4_fraction: float | None
    running_rate: float | None
    quarter_life: float | None
    quarter_life_rank: float | None
    curvature: float | None


@dataclass(frozen=True)
class Acceleration:
    """Counts of IRTs shorter / longer / equal to their predecessor."""

    shorter: int
    longer: int
    equal: int
    p_one_sided: float | None


def fi_intervals(
    response_times: Sequence[float],
    reinforcement_times: Sequence[float],
    *,
    drop_first: bool = True,
) -> list[FIInterval]:
    """Split responses into inter-reinforcement intervals.

    Responses in ``(start, end]`` belong to the interval, so the
    reinforced response closes its own interval.
    """
    times = sorted(response_times)
    edges = [0.0, *sorted(reinforcement_times)]
    out: list[FIInterval] = []
    for start, end in pairwise(edges):
        if end <= start:
            continue
        lo = bisect.bisect_right(times, start)
        hi = bisect.bisect_right(times, end)
        out.append(FIInterval(start, end, tuple(times[lo:hi])))
    return out[1:] if drop_first else out


def quarter_life_interpolated(iv: FIInterval, quantile: float = 0.25) -> float | None:
    """Quarter-life with the cumulative count treated as continuous.

    The cumulative record is joined linearly through ``(start, 0)`` and
    ``(t_k, k)``; the result is where it reaches ``quantile * n``.
    """
    n = len(iv.responses)
    if n < 2:
        return None
    target = quantile * n
    k = max(1, math.ceil(target))
    prev = iv.start if k == 1 else iv.responses[k - 2]
    t = prev + (target - (k - 1)) * (iv.responses[k - 1] - prev)
    return (t - iv.start) / iv.length


def quarter_life_rank(iv: FIInterval, quantile: float = 0.25) -> float | None:
    """Quarter-life as the ``ceil(quantile * n)``-th response's time."""
    n = len(iv.responses)
    if n < 2:
        return None
    t = iv.responses[max(math.ceil(quantile * n) - 1, 0)]
    return (t - iv.start) / iv.length


def _bin_counts(iv: FIInterval, n_bins: int) -> list[int]:
    counts = [0] * n_bins
    for t in iv.responses:
        pos = (t - iv.start) * n_bins / iv.length
        counts[min(n_bins - 1, max(0, math.ceil(pos - 1e-9) - 1))] += 1
    return counts


def pooled_distribution(intervals: Sequence[FIInterval], n_bins: int = 10) -> list[int]:
    """Response counts in ``n_bins`` equal parts of the interval, summed.

    Intervals with fewer than two responses are left out (Dukich & Lee, 1973).
    """
    totals = [0] * n_bins
    for iv in intervals:
        if len(iv.responses) < 2:
            continue
        for i, c in enumerate(_bin_counts(iv, n_bins)):
            totals[i] += c
    return totals


def pooled_quarter_life(
    intervals: Sequence[FIInterval], n_bins: int = 10, quantile: float = 0.25
) -> float | None:
    """Quarter-life of the session's pooled response distribution."""
    totals = pooled_distribution(intervals, n_bins)
    n = sum(totals)
    if n == 0:
        return None
    target = quantile * n
    cum = 0
    for i, c in enumerate(totals):
        if c and cum + c >= target:
            return (i + (target - cum) / c) / n_bins
        cum += c
    return 1.0


def index_of_curvature(iv: FIInterval, n_div: int = 4) -> float | None:
    """Fry, Kelleher & Cook's (1960) index with ``n_div`` divisions."""
    if not iv.responses:
        return None
    counts = _bin_counts(iv, n_div)
    cumulative = [sum(counts[: i + 1]) for i in range(n_div)]
    r_n = cumulative[-1]
    return ((n_div - 1) * r_n - 2 * sum(cumulative[:-1])) / (n_div * r_n)


def interval_measures(iv: FIInterval) -> IntervalMeasures:
    """PRP, T4, running rate, quarter-life and curvature for one interval."""
    n = len(iv.responses)
    prp = (iv.responses[0] - iv.start) if n else None
    enough = n >= 2
    running = None
    if enough and prp is not None and iv.length - prp > 0:
        running = n / (iv.length - prp)
    return IntervalMeasures(
        n_responses=n,
        prp_fraction=None if prp is None else prp / iv.length,
        t4_fraction=(iv.responses[3] - iv.start) / iv.length if n >= 4 else None,
        running_rate=running,
        quarter_life=quarter_life_interpolated(iv),
        quarter_life_rank=quarter_life_rank(iv),
        curvature=index_of_curvature(iv) if n >= CURVATURE_MIN_RESPONSES else None,
    )


def pause_criterion(irts: Sequence[float], share: float = 0.05) -> float | None:
    """Shortest IRT ``c`` such that IRTs >= ``c`` end at most ``share`` of responses.

    Gentry et al. (1983) took 5 % and got about 3.5 s for pigeons. ``None``
    when even the longest IRT value ends more than ``share`` of responses.
    """
    if not irts:
        return None
    srt = sorted(irts)
    n = len(srt)
    for c in sorted(set(srt)):
        at_or_above = n - bisect.bisect_left(srt, c)
        if at_or_above / n <= share:
            return c
    return None


def ordinal_irts(
    intervals: Sequence[FIInterval], max_order: int = 11
) -> list[tuple[int, float, int]]:
    """Mean of the k-th IRT in the interval: ``(k, mean, n_intervals)``."""
    out: list[tuple[int, float, int]] = []
    for k in range(1, max_order + 1):
        vals = [iv.irts[k - 1] for iv in intervals if len(iv.irts) >= k]
        if not vals:
            break
        out.append((k, statistics.fmean(vals), len(vals)))
    return out


def _binom_upper_tail(k: int, n: int) -> float:
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n


def acceleration(intervals: Sequence[FIInterval]) -> Acceleration:
    """Successive IRT pairs inside intervals; ties are counted apart.

    ``p_one_sided`` is the binomial probability of at least this many
    shorter IRTs among the non-tied pairs if shorter and longer were
    equally likely.
    """
    shorter = longer = equal = 0
    for iv in intervals:
        irts = iv.irts
        for a, b in pairwise(irts):
            if b < a:
                shorter += 1
            elif b > a:
                longer += 1
            else:
                equal += 1
    n = shorter + longer
    return Acceleration(shorter, longer, equal, _binom_upper_tail(shorter, n) if n else None)


def _median(values: Sequence[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def _mean(values: Sequence[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return statistics.fmean(vals) if vals else None


@dataclass(frozen=True)
class FISummary:
    """Session summary of the FI measures."""

    interval: float
    n_intervals: int
    n_intervals_ge2: int
    n_responses: int
    response_rate: float
    per_interval: list[IntervalMeasures] = field(repr=False)
    pooled_quarter_life: float | None
    pause_criterion: float | None
    pauses_first_half: int
    pauses_second_half: int
    ordinal: list[tuple[int, float, int]]
    acceleration: Acceleration
    log_survivor: Any | None

    def _col(self, name: str) -> list[float | None]:
        return [getattr(m, name) for m in self.per_interval]

    @property
    def median_responses_per_interval(self) -> float | None:
        return _median([float(m.n_responses) for m in self.per_interval])

    @property
    def median_prp_fraction(self) -> float | None:
        return _median(self._col("prp_fraction"))

    @property
    def mean_prp_fraction(self) -> float | None:
        return _mean(self._col("prp_fraction"))

    @property
    def median_t4_fraction(self) -> float | None:
        return _median(self._col("t4_fraction"))

    @property
    def median_running_rate(self) -> float | None:
        return _median(self._col("running_rate"))

    @property
    def mean_quarter_life(self) -> float | None:
        return _mean(self._col("quarter_life"))

    @property
    def mean_quarter_life_rank(self) -> float | None:
        return _mean(self._col("quarter_life_rank"))

    @property
    def median_curvature(self) -> float | None:
        return _median(self._col("curvature"))

    @property
    def n_curvature(self) -> int:
        return sum(m.curvature is not None for m in self.per_interval)

    def as_dict(self) -> dict[str, Any]:
        ls = self.log_survivor
        return {
            "interval": self.interval,
            "n_intervals": self.n_intervals,
            "n_intervals_ge2": self.n_intervals_ge2,
            "n_responses": self.n_responses,
            "response_rate": self.response_rate,
            "median_responses_per_interval": self.median_responses_per_interval,
            "mean_quarter_life": self.mean_quarter_life,
            "mean_quarter_life_rank": self.mean_quarter_life_rank,
            "pooled_quarter_life": self.pooled_quarter_life,
            "mean_prp_fraction": self.mean_prp_fraction,
            "median_prp_fraction": self.median_prp_fraction,
            "median_t4_fraction": self.median_t4_fraction,
            "median_running_rate": self.median_running_rate,
            "median_curvature": self.median_curvature,
            "n_curvature": self.n_curvature,
            "pause_criterion": self.pause_criterion,
            "pauses_first_half": self.pauses_first_half,
            "pauses_second_half": self.pauses_second_half,
            "ordinal_irts": self.ordinal,
            "acceleration": vars(self.acceleration),
            "log_survivor": None if ls is None else {
                "knot": ls.knot_time_sec,
                "slope_fast": ls.slope_fast,
                "slope_slow": ls.slope_slow,
                "intercept_slow": ls.intercept_slow,
                "n_irts": ls.n_irts,
            },
        }


def summarize_fi(
    response_times: Sequence[float],
    reinforcement_times: Sequence[float],
    *,
    interval: float,
    session_length: float | None = None,
) -> FISummary:
    """All FI measures for one session (warm-up interval dropped)."""
    ivs = fi_intervals(response_times, reinforcement_times)
    per = [interval_measures(iv) for iv in ivs]
    irts = [x for iv in ivs for x in iv.irts]
    crit = pause_criterion(irts)
    first = second = 0
    if crit is not None:
        for iv in ivs:
            r = iv.responses
            for a, b in pairwise(r):
                if b - a >= crit:
                    if (a - iv.start) / iv.length < 0.5:
                        first += 1
                    else:
                        second += 1
    log_survivor = None
    try:
        from session_analyzer.analytics.log_survivor import fit_bilinear_log_survivor
    except ImportError:  # pragma: no cover - optional dependency
        pass
    else:
        log_survivor = fit_bilinear_log_survivor(irts)
    span = session_length if session_length is not None else (
        max(response_times) if response_times else 0.0
    )
    return FISummary(
        interval=interval,
        n_intervals=len(ivs),
        n_intervals_ge2=sum(m.n_responses >= 2 for m in per),
        n_responses=sum(m.n_responses for m in per),
        response_rate=len(response_times) / span if span else 0.0,
        per_interval=per,
        pooled_quarter_life=pooled_quarter_life(ivs),
        pause_criterion=crit,
        pauses_first_half=first,
        pauses_second_half=second,
        ordinal=ordinal_irts(ivs),
        acceleration=acceleration(ivs),
        log_survivor=log_survivor,
    )


__all__ = [
    "CURVATURE_MIN_RESPONSES",
    "Acceleration",
    "FIInterval",
    "FISummary",
    "IntervalMeasures",
    "acceleration",
    "fi_intervals",
    "index_of_curvature",
    "interval_measures",
    "ordinal_irts",
    "pause_criterion",
    "pooled_distribution",
    "pooled_quarter_life",
    "quarter_life_interpolated",
    "quarter_life_rank",
    "summarize_fi",
]
