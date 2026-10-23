"""Draw FI scallop k and R^2 against two dose axes: model scale and temperature.

Axis (a) reads FI runs saved by ``claude_cli_fi.py`` (one per model, given in
order of scale). Axis (b) reads runs saved by ``local_logprob_fi.py`` at
several temperatures. Every run is cut to the first ``--steps`` steps and
``P(R | t) = min(1, k t^2)`` is fitted with :class:`ModelFitAnalyzer` on the
emitted lever presses, so both axes use the same measure. For the local runs
the same model is also fitted to the read-out ``P(Lever Press | t)`` (no
sampling). Quarter-life and PRP / FI from ``fi_measures`` are added as a check
that does not depend on the k t^2 form.

    python examples/dose_response.py \\
        --scale haiku=runs/s40/fi30_haiku.json sonnet=runs/s48/fi30_sonnet.json \\
        --temperature runs/s48/local/*.json --steps 300 --out-dir runs/s48
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from ai_operant_box.agent.local_logprob import p_response_by_time_since_reinforcer
from ai_operant_box.analytics.fi_measures import summarize_fi
from ai_operant_box.analytics.model_fit import ModelFitAnalyzer, scallop_model
from ai_operant_box.chamber.environment import ResponseRecord

TARGET = "Lever Press"


def fit_presses(steps: list[tuple[float, str, bool]], interval: float) -> dict:
    """k, R^2 (ModelFitAnalyzer, lever presses only) and FI measures."""
    lever = [ResponseRecord(t, None, r, sr) for t, r, sr in steps if r == TARGET]
    with tempfile.TemporaryDirectory() as tmp:
        k, r2 = ModelFitAnalyzer(lever, interval).fit_and_plot(str(Path(tmp) / "fit.png"))
    presses = [rec.timestamp for rec in lever]
    sr = [t for t, _, s in steps if s]
    fi = summarize_fi(presses, sr, interval=interval, session_length=float(len(steps)))
    actions = [r for _, r, _ in steps]
    return {
        "k": None if k is None else float(k),
        "r_squared": None if r2 is None else float(r2),
        "lever_presses": len(presses),
        "reinforcers": len(sr),
        "action_counts": {a: actions.count(a) for a in sorted(set(actions), key=str)},
        "quarter_life": fi.mean_quarter_life,
        "prp_fraction": fi.median_prp_fraction,
        "n_intervals": fi.n_intervals,
    }


def fit_curve(curve: dict[float, float], interval: float) -> dict:
    """Least-squares min(1, k t^2) on the read-out P(R | t), 0 < t <= interval."""
    from scipy.optimize import curve_fit

    ts = np.array([t for t in curve if 0 < t <= interval], dtype=float)
    ps = np.array([curve[t] for t in ts], dtype=float)
    try:
        (k,), _ = curve_fit(scallop_model, ts, ps, p0=[0.001])
    except Exception:  # pragma: no cover - fit failure is reported as None
        return {"k_p": None, "r_squared_p": None}
    ss_res = float(np.sum((ps - scallop_model(ts, k)) ** 2))
    ss_tot = float(np.sum((ps - ps.mean()) ** 2))
    return {"k_p": float(k), "r_squared_p": 1 - ss_res / ss_tot if ss_tot else 0.0}


def load_claude(path: Path, n: int) -> tuple[list[tuple[float, str, bool]], float, dict]:
    run = json.loads(path.read_text())
    steps = [(r["t"], r["r"], r["sr"]) for r in run["records"] if r["t"] <= n]
    meta = {"steps_done": run["steps_done"], "invalid_replies": run["invalid_replies"],
            "latency_mean_s": float(np.mean(run["latencies_s"][:n]))}
    return steps, float(run["interval_s"]), meta


def load_local(path: Path, n: int) -> tuple[list[tuple[float, str, bool]], float, dict, dict]:
    run = json.loads(path.read_text())
    detail = run["steps_detail"][:n]
    steps = [(d["t"], d["action"], d["reinforced"]) for d in detail]
    interval = float(run["schedule"].split()[1])
    p = [d["p"][TARGET] for d in detail]
    curve = p_response_by_time_since_reinforcer(
        [d["t"] for d in detail], p, [d["reinforced"] for d in detail])
    meta = {"temperature": run["temperature"], "p_lever_press_mean": float(np.mean(p)),
            "step_s_mean": run["step_s_mean"]}
    return steps, interval, meta, curve


def _fmt(v, nd: int = 3) -> str:
    return "—" if v is None else f"{v:.{nd}g}" if abs(v) < 1e-2 and v != 0 else f"{v:.{nd}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scale", nargs="+", default=[], metavar="NAME=RUN.json")
    parser.add_argument("--temperature", nargs="+", type=Path, default=[])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/dose"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scale = []
    for item in args.scale:
        name, path = item.split("=", 1)
        steps, interval, meta = load_claude(Path(path), args.steps)
        scale.append({"model": name, "run": path, **meta, **fit_presses(steps, interval)})

    temp = []
    for path in args.temperature:
        steps, interval, meta, curve = load_local(path, args.steps)
        temp.append({"run": str(path), **meta, **fit_presses(steps, interval),
                     **fit_curve(curve, interval),
                     "p_by_elapsed_s": {str(k): v for k, v in curve.items()}})
    temp.sort(key=lambda r: r["temperature"])

    out = {"steps": args.steps, "model_scale": scale, "temperature": temp}
    (args.out_dir / "dose_response.json").write_text(json.dumps(out, indent=2))

    lines = [f"# FI 30 dose response (first {args.steps} steps)", "",
             "## (a) model scale (claude -p)", "",
             "| model | lever presses | SR | k | R² | quarter-life | PRP/FI | invalid |",
             "|---|---|---|---|---|---|---|---|"]
    for r in scale:
        lines.append(f"| {r['model']} | {r['lever_presses']} | {r['reinforcers']} | {_fmt(r['k'])}"
                     f" | {_fmt(r['r_squared'])} | {_fmt(r['quarter_life'])}"
                     f" | {_fmt(r['prp_fraction'])} | {r['invalid_replies']} |")
    lines += ["", "## (b) local model temperature (sampled actions; read-out P)", "",
              "| T | lever presses | SR | k | R² | k (P) | R² (P) | mean P(LP) | quarter-life |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in temp:
        lines.append(f"| {r['temperature']:g} | {r['lever_presses']} | {r['reinforcers']}"
                     f" | {_fmt(r['k'])} | {_fmt(r['r_squared'])} | {_fmt(r['k_p'])}"
                     f" | {_fmt(r['r_squared_p'])} | {_fmt(r['p_lever_press_mean'])}"
                     f" | {_fmt(r['quarter_life'])} |")
    (args.out_dir / "dose_response.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    _plot(args.out_dir / "dose_response.png", scale, temp, args.steps)


def _plot(png: Path, scale: list[dict], temp: list[dict], steps: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    names = [r["model"] for r in scale]
    xs = list(range(len(scale)))
    ts = [r["temperature"] for r in temp]
    for row, (key, label) in enumerate((("k", "k  (P = min(1, k t²))"), ("r_squared", "R²"))):
        ax = axes[row][0]
        ax.plot(xs, [np.nan if r[key] is None else r[key] for r in scale], "o-", color="black")
        ax.set_xticks(xs, names)
        ax.set_ylabel(label)
        ax.set_xlabel("model scale (claude -p)")
        ax = axes[row][1]
        ax.plot(ts, [np.nan if r[key] is None else r[key] for r in temp], "o-", color="black",
                label="sampled lever presses")
        ax.plot(ts, [np.nan if r[key + "_p" if key == "k" else "r_squared_p"] is None
                     else r["k_p" if key == "k" else "r_squared_p"] for r in temp],
                "s--", color="0.5", label="read-out P(LP | t)")
        ax.set_xlabel("temperature (Qwen2.5-1.5B-Instruct Q4_K_M)")
        ax.set_ylabel(label)
        ax.legend(loc="best", fontsize=8)
    for ax in (axes[0][0], axes[0][1]):
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
    axes[0][0].axhline(1 / 1.5**2, color="0.6", lw=0.8, ls=":")
    axes[0][0].annotate("k = 1/1.5² (saturated from the 2nd bin)", (0, 1 / 1.5**2),
                        textcoords="offset points", xytext=(4, -12), fontsize=8, color="0.4")
    fig.suptitle(f"FI 30 s, first {steps} steps: scallop fit vs dose")
    fig.savefig(png, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
