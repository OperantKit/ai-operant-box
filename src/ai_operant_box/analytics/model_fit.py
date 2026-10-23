"""Scallop-model fitter for FI cumulative records.

Fits ``P(R | t) = min(1, k * t**2)`` to per-bin empirical response
probabilities and renders a 2-panel diagnostic figure (fit + residuals).

NumPy is required (declared as a hard dependency); ``scipy`` and
``matplotlib`` are imported lazily.
"""

from __future__ import annotations

import numpy as np

from ai_operant_box.chamber.environment import ResponseRecord


def scallop_model(t, k):
    """Theoretical model: ``P(R) = min(1, k * t**2)``."""
    return np.minimum(1.0, k * (t ** 2))


class ModelFitAnalyzer:
    """Empirical-vs-model fitter for FI sessions.

    Parameters
    ----------
    records : list[ResponseRecord]
        Chamber records. Reinforcer times are taken from every record with
        ``is_reinforced``, whatever ``response_type`` is.
    interval : float
        FI value in seconds; bins cover ``0..interval``.
    response_type : str, optional
        Count only records whose ``response`` equals this (e.g.
        ``"Lever Press"``). ``None`` counts every record with a response.
    """

    def __init__(
        self,
        records: list[ResponseRecord],
        interval: float,
        response_type: str | None = None,
    ) -> None:
        self.records = records
        self.interval = interval
        self.response_type = response_type

    def _time_since_reinforcer(self, t: float, sr_times: np.ndarray) -> float:
        # A response reinforced at t was emitted before its reinforcer, so both
        # responses and opportunities are timed from the last reinforcer
        # strictly before t (session start counts as one at 0).
        earlier = sr_times[sr_times < t]
        return float(t - earlier.max()) if earlier.size else float(t)

    def get_empirical_probabilities(self, bin_size: float = 1.0):
        """Per-bin ``P(R | t)`` with one opportunity per chamber step.

        Opportunities are the step times ``bin_size, 2 * bin_size, ...`` up to
        the last record (the chamber's step is assumed equal to ``bin_size``).
        Each counted response falls in the same bin as the opportunity at its
        own time, so ``P <= 1``. Bins without an opportunity are omitted.
        """
        if not self.records:
            return np.array([]), np.array([])

        sr_times = np.array([r.timestamp for r in self.records if r.is_reinforced])
        last_time = self.records[-1].timestamp
        t_opportunities = np.arange(bin_size, last_time + bin_size / 2, bin_size)

        rel_times = np.array(
            [self._time_since_reinforcer(float(t), sr_times) for t in t_opportunities]
        )
        rel_times = rel_times[rel_times <= self.interval]

        counted = [
            r
            for r in self.records
            if r.response is not None
            and (self.response_type is None or r.response == self.response_type)
        ]
        resp_rel_times = np.array(
            [self._time_since_reinforcer(r.timestamp, sr_times) for r in counted]
        )

        bins = np.arange(0, self.interval + bin_size, bin_size)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        opp_counts, _ = np.histogram(rel_times, bins=bins)
        resp_counts, _ = np.histogram(resp_rel_times, bins=bins)

        visited = opp_counts != 0
        probs = resp_counts[visited] / opp_counts[visited]

        return bin_centers[visited], probs

    def fit_and_plot(self, output_path: str):
        import matplotlib.pyplot as plt  # local import
        from scipy.optimize import curve_fit

        t_data, p_data = self.get_empirical_probabilities()

        if len(t_data) == 0:
            return None, None

        try:
            popt, _ = curve_fit(scallop_model, t_data, p_data, p0=[0.001])
            k_fit = popt[0]
        except Exception:
            k_fit = 0.001

        p_fit = scallop_model(t_data, k_fit)

        ss_res = np.sum((p_data - p_fit) ** 2)
        ss_tot = np.sum((p_data - np.mean(p_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        residuals = p_data - p_fit

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        ax1.scatter(t_data, p_data, color="blue", alpha=0.5, label="Empirical P(R|t)")
        ax1.plot(
            t_data,
            p_fit,
            "r-",
            label=f"Model Fit: {k_fit:0.5f}*t^2 (R^2={r_squared:0.3f})",
        )
        ax1.set_ylabel("Response Probability")
        ax1.set_title("LLM Behavior Model Fit Analysis")
        ax1.legend()
        ax1.grid(True)

        ax2.bar(t_data, residuals, color="gray", alpha=0.7, label="Residuals (Emp - Fit)")
        ax2.axhline(0, color="black", linewidth=1)
        ax2.set_xlabel("Time since last Reinforcement (s)")
        ax2.set_ylabel("Deviation")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        import os

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path)
        plt.close()

        return k_fit, r_squared


__all__ = ["ModelFitAnalyzer", "scallop_model"]
