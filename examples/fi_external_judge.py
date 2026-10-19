"""Judge recorded FI sessions against published values and on-board controls.

Reads FI runs saved by ``claude_cli_fi.py`` / ``claude_cli_instruction.py``
(``records`` with ``t`` / ``r`` / ``sr``), computes the measures in
``ai_operant_box.analytics.fi_measures`` for each run, runs two control
subjects on the same discrete board (a Bernoulli responder matched to
each run's response rate, and the quadratic-time ``ScallopingAgent``),
and writes a Markdown table, a JSON summary and a figure.

    python examples/fi_external_judge.py RUN.json [RUN.json ...] --out-dir DIR
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ai_operant_box.agent.bernoulli_agent import BernoulliAgent
from ai_operant_box.agent.scallop_agent import ScallopingAgent
from ai_operant_box.analytics.fi_measures import (
    FISummary,
    fi_intervals,
    pooled_distribution,
    summarize_fi,
)
from ai_operant_box.chamber.environment import OperantBox
from ai_operant_box.chamber.schedule import FixedInterval

TARGET = "Lever Press"

# Published values as fractions of the interval (aob-04 §2; S-60 §2.1).
LITERATURE = {
    "quarter-life (interpolated)": "pigeon 0.65-0.75; rat 0.66-0.80 (mean 0.72)",
    "PRP / FI": "rat 0.52-0.69",
    "T4 / FI": "rat 0.60-0.77",
    "index of curvature (n=4)": "scalloped records > 0 (max 0.75)",
    "pause criterion": "pigeon ~3.5 s (5 % rule)",
    "pauses early vs late": "concentrated early in the interval",
    "ordinal IRTs": "1st > 6th > 11th (rat, pigeon)",
    "IRT shorter than previous": "no evidence of acceleration (pigeon)",
}


def load(path: Path) -> tuple[list[float], list[float], float, float]:
    run = json.loads(path.read_text())
    presses = [r["t"] for r in run["records"] if r["r"] == TARGET]
    sr = [r["t"] for r in run["records"] if r["sr"]]
    return presses, sr, float(run["interval_s"]), float(run["steps_done"])


def simulate(subject, steps: int, interval: float) -> tuple[list[float], list[float]]:
    box = OperantBox(subject=subject, schedule=FixedInterval(interval))
    for _ in range(steps):
        if isinstance(subject, ScallopingAgent):
            subject.update_time(1.0)
        box.step(1.0)
    presses = [r.timestamp for r in box.records if r.response == TARGET]
    sr = [r.timestamp for r in box.records if r.is_reinforced]
    return presses, sr


def fmt(x, nd: int = 2) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def rows(s: FISummary) -> dict[str, str]:
    acc = s.acceleration
    ordinal = {k: m for k, m, _ in s.ordinal}
    ls = s.log_survivor
    return {
        "intervals (≥2 responses)": f"{s.n_intervals} ({s.n_intervals_ge2})",
        "responses in intervals": str(s.n_responses),
        "median responses / interval": fmt(s.median_responses_per_interval, 1),
        "quarter-life (interpolated)": fmt(s.mean_quarter_life),
        "quarter-life (rank)": fmt(s.mean_quarter_life_rank),
        "quarter-life (pooled, 10 bins)": fmt(s.pooled_quarter_life),
        "PRP / FI": fmt(s.mean_prp_fraction),
        "T4 / FI": fmt(s.median_t4_fraction),
        "running rate (/step)": fmt(s.median_running_rate),
        "index of curvature (n=4)": f"{fmt(s.median_curvature)} (n={s.n_curvature})",
        "pause criterion": fmt(s.pause_criterion, 0),
        "pauses early vs late": f"{s.pauses_first_half} / {s.pauses_second_half}",
        "ordinal IRTs": " / ".join(fmt(ordinal.get(k)) for k in (1, 6, 11)),
        "IRT shorter than previous": (
            f"{acc.shorter}/{acc.shorter + acc.longer} (ties {acc.equal}, p={fmt(acc.p_one_sided, 3)})"
        ),
        "log-survivor knot / slopes": (
            "–" if ls is None else f"{ls.knot_time_sec:g} / {ls.slope_fast:.2f}, {ls.slope_slow:.2f}"
        ),
    }


def pooled_profile(presses: list[float], sr: list[float], n_bins: int = 10) -> list[float]:
    totals = pooled_distribution(fi_intervals(presses, sr), n_bins)
    n = sum(totals) or 1
    return [t / n for t in totals]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    columns: dict[str, FISummary] = {}
    profiles: dict[str, list[float]] = {}
    interval = steps = None
    for path in args.runs:
        presses, sr, interval, steps = load(path)
        name = path.stem.removeprefix("fi30_")
        columns[name] = summarize_fi(presses, sr, interval=interval, session_length=steps)
        profiles[name] = pooled_profile(presses, sr)
        random.seed(args.seed)
        p = len(presses) / steps
        cp, csr = simulate(BernoulliAgent(p), int(steps), interval)
        columns[f"Bernoulli p={p:.2f}"] = summarize_fi(cp, csr, interval=interval, session_length=steps)
        profiles[f"Bernoulli p={p:.2f}"] = pooled_profile(cp, csr)
    random.seed(args.seed)
    cp, csr = simulate(ScallopingAgent(k=1.0 / interval**2), int(steps), interval)
    columns["ScallopingAgent"] = summarize_fi(cp, csr, interval=interval, session_length=steps)
    profiles["ScallopingAgent"] = pooled_profile(cp, csr)

    names = list(dict.fromkeys(columns))
    table = {n: rows(columns[n]) for n in names}
    measures = list(next(iter(table.values())))
    lines = [
        f"FI {interval:g}, {int(steps)} steps per run, first interval dropped as warm-up. "
        "Times are fractions of the interval.",
        "",
        "| measure | literature | " + " | ".join(names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for m in measures:
        lines.append(
            f"| {m} | {LITERATURE.get(m, '')} | " + " | ".join(table[n][m] for n in names) + " |"
        )
    md = "\n".join(lines) + "\n"
    (args.out_dir / "fi_external_judge.md").write_text(md)
    (args.out_dir / "fi_external_judge.json").write_text(
        json.dumps({n: columns[n].as_dict() for n in names}, indent=2, ensure_ascii=False)
    )
    print(md)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = [(i + 0.5) / 10 for i in range(10)]
    for n in names:
        style = "--" if n.startswith(("Bernoulli", "Scallop")) else "-"
        ax.plot(x, profiles[n], style, marker="o", label=n)
    ax.set_xlabel("position in interval (fraction of FI)")
    ax.set_ylabel("share of responses (pooled)")
    ax.set_title(f"FI {interval:g}: pooled response distribution in 10 bins")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "fi_external_judge_profile.png", dpi=120)


if __name__ == "__main__":
    main()
