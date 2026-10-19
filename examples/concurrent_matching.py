"""Generalized matching with a numerical subject on Conc VI VI.

Runs ``MatchingAgent`` on four reinforcer ratios (1:1, 1:3, 3:1, 1:9;
overall 1 reinforcer per 10 s, 1 step = 1 s) and fits
``log(B1/B2) = a * log(R1/R2) + log b`` over the conditions, skipping the
first ``--warmup`` steps of each session.

    python examples/concurrent_matching.py --cod 2 --sensitivity 1
"""

from __future__ import annotations

import argparse
import random

from ai_operant_box.agent.matching_agent import MatchingAgent
from ai_operant_box.analytics.matching import choice_totals, fit_generalized_matching
from ai_operant_box.chamber.concurrent import ConcurrentBox, concurrent_vi_vi

RATIOS = [(1, 1), (1, 3), (3, 1), (1, 9)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--warmup", type=float, default=1000.0)
    parser.add_argument("--cod", type=float, default=2.0)
    parser.add_argument("--sensitivity", type=float, default=1.0)
    parser.add_argument("--mean-visit", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()

    totals = []
    print("ratio  VI_left  VI_right  B_left  B_right  R_left  R_right")
    for i, (r_left, r_right) in enumerate(RATIOS):
        mean_left = 10.0 * (r_left + r_right) / r_left
        mean_right = 10.0 * (r_left + r_right) / r_right
        seed = args.seed + i
        box = ConcurrentBox(
            MatchingAgent(args.sensitivity, rng=random.Random(seed), mean_visit=args.mean_visit),
            concurrent_vi_vi(mean_left, mean_right, cod=args.cod, seed=seed),
        )
        box.run(args.steps)
        t = choice_totals(box.records, start=args.warmup)
        totals.append(t)
        print(
            f"{r_left}:{r_right}  {mean_left:7.1f}  {mean_right:8.1f}  "
            f"{t.responses['left']:6d}  {t.responses['right']:7d}  "
            f"{t.reinforcers['left']:6d}  {t.reinforcers['right']:7d}"
        )

    fit = fit_generalized_matching(
        [t.responses["left"] for t in totals],
        [t.responses["right"] for t in totals],
        [t.reinforcers["left"] for t in totals],
        [t.reinforcers["right"] for t in totals],
    )
    print(f"a={fit.a:.3f} log_b={fit.log_b:.3f} r2={fit.r_squared:.3f} n={fit.n}")


if __name__ == "__main__":
    main()
