"""Cumulative-record visualisation helpers.

All rendering delegates to ``session_analyzer.visualizer`` so that the
canonical Ferster & Skinner conventions (~30 deg diagonal reinforcement
pips, 550-response pen reset, event pen) are applied uniformly across
OperantKit. See Lattal (2004), JEAB 82:329-355, for canonical spec.

``matplotlib`` and ``session_analyzer`` are imported lazily at call time
so the module remains importable when optional viz dependencies are
missing (image-rendering tests skip when they are).
"""

from __future__ import annotations

import os
from pathlib import Path

from ai_operant_box.chamber.environment import ResponseRecord

try:
    from session_analyzer.visualizer import build_cumulative_record_figure
except ImportError:  # pragma: no cover - optional viz dep
    build_cumulative_record_figure = None  # type: ignore[assignment]


def _ensure_parent(file_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)


def _rebase_times(records: list[ResponseRecord]) -> list[float]:
    if not records:
        return []
    t0 = records[0].timestamp
    return [r.timestamp - t0 for r in records]


class CumulativeRecorder:
    """Single-session cumulative-record renderer (delegates to canonical)."""

    def __init__(self, records: list[ResponseRecord]) -> None:
        self.records = records

    def save_png(self, file_path: str, title: str = "Cumulative Record") -> None:
        if not self.records:
            return
        if build_cumulative_record_figure is None:
            raise ImportError(
                "session_analyzer is required for cumulative-record rendering. "
                "Install it from apps/analysis/session-analyzer."
            )

        response_times = _rebase_times(self.records)
        reinforcement_times = [
            t for t, r in zip(response_times, self.records) if r.is_reinforced
        ]

        # session_analyzer's renderer takes no title, so build the canonical
        # figure, replace its default axes title, and store the title in the
        # PNG metadata.
        fig, main_ax, _ = build_cumulative_record_figure(
            response_times=response_times,
            reinforcement_times=reinforcement_times,
        )
        main_ax.set_title(title)
        _ensure_parent(file_path)
        try:
            fig.savefig(Path(file_path), metadata={"Title": title})
        finally:
            fig.clear()


class MultiSessionRecorder:
    """Compare multiple sessions by overlaying canonical cumulative records.

    Each session is rendered by ``build_cumulative_record_figure`` and
    composited onto a single figure. Pip angle, pen-reset threshold, and
    event-pen conventions follow the defaults from
    ``session_analyzer.visualizer``.
    """

    def __init__(self) -> None:
        self.sessions: dict[str, list[ResponseRecord]] = {}

    def add_session(self, label: str, records: list[ResponseRecord]) -> None:
        self.sessions[label] = records

    def save_comparison_png(
        self, file_path: str, title: str = "Session Comparison"
    ) -> None:
        if not self.sessions:
            return
        if build_cumulative_record_figure is None:
            raise ImportError(
                "session_analyzer is required for cumulative-record rendering. "
                "Install it from apps/analysis/session-analyzer."
            )

        # Lazy import keeps module-import cheap.
        import matplotlib
        from matplotlib.figure import Figure

        fig = Figure(figsize=(12, 8))
        ax = fig.subplots()
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Cumulative Responses")
        ax.set_title(title)

        # Build a canonical sub-figure per session and extract its main
        # cumulative line (labelled ``_cumulative``) + reinforcement pips
        # (``_pip``) onto the comparison axes.
        for label, records in self.sessions.items():
            if not records:
                continue
            response_times = _rebase_times(records)
            reinforcement_times = [
                t for t, r in zip(response_times, records) if r.is_reinforced
            ]
            sub_fig, sub_main, _ = build_cumulative_record_figure(
                response_times=response_times,
                reinforcement_times=reinforcement_times,
                show_event_pen=False,
            )
            cum = next(
                l for l in sub_main.get_lines() if l.get_label() == "_cumulative"
            )
            ax.step(
                cum.get_xdata(),
                cum.get_ydata(),
                where="post",
                label=label,
                linewidth=1.5,
            )
            for pip in [l for l in sub_main.get_lines() if l.get_label() == "_pip"]:
                ax.plot(
                    pip.get_xdata(),
                    pip.get_ydata(),
                    color=pip.get_color(),
                    linewidth=pip.get_linewidth(),
                )
            sub_fig.clear()

        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        fig.tight_layout()

        _ensure_parent(file_path)
        fig.savefig(file_path)
        fig.clear()


class LiveVisualizer:
    """Real-time cumulative-record visualiser using matplotlib's interactive
    mode.

    Streaming rendering is incompatible with post-hoc figure assembly, so
    the live visualiser draws its own trace but applies canonical F&S
    pip styling (short ~30 deg diagonal) for each reinforcement.
    """

    def __init__(self, title: str = "Live Cumulative Record") -> None:
        import math

        import matplotlib.pyplot as plt

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        (self.line,) = self.ax.step([], [], "k-", where="post", label="Responses")
        self.pips: list = []
        self.times: list = []
        self.responses: list = []
        self._pip_angle_rad = math.radians(-30.0)
        self._pip_length_pt = 7.0
        self.ax.set_title(title)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Cumulative Responses")
        self.ax.grid(True)

    def update(self, record: ResponseRecord | None) -> None:
        if not record:
            return

        import math

        import matplotlib.pyplot as plt

        self.times.append(record.timestamp)
        self.responses.append(len(self.responses) + 1)
        self.line.set_data(self.times, self.responses)

        if record.is_reinforced:
            import numpy as np

            x0, y0 = record.timestamp, self.responses[-1]
            dx_pt = self._pip_length_pt * math.cos(self._pip_angle_rad)
            dy_pt = self._pip_length_pt * math.sin(self._pip_angle_rad)
            px_per_pt = self.fig.dpi / 72.0
            start_disp = self.ax.transData.transform((x0, y0))
            end_disp = start_disp + np.array(
                [dx_pt * px_per_pt, dy_pt * px_per_pt]
            )
            x_end, y_end = self.ax.transData.inverted().transform(end_disp)
            (pip,) = self.ax.plot(
                [x0, x_end],
                [y0, y_end],
                "k-",
                linewidth=1,
                solid_capstyle="butt",
            )
            self.pips.append(pip)

        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)

    def close(self) -> None:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


__all__ = ["CumulativeRecorder", "LiveVisualizer", "MultiSessionRecorder"]
