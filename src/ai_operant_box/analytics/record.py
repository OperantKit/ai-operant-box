"""Cumulative-record visualisation helpers.

``matplotlib`` is imported lazily inside each function so the module can be
imported without the dependency installed (image-rendering tests skip when
matplotlib is missing).
"""

from __future__ import annotations

import os

from ai_operant_box.chamber.environment import ResponseRecord


class CumulativeRecorder:
    """Single-session cumulative-record renderer."""

    def __init__(self, records: list[ResponseRecord]) -> None:
        self.records = records

    def save_png(self, file_path: str, title: str = "Cumulative Record") -> None:
        if not self.records:
            return

        import matplotlib.pyplot as plt  # local import keeps module import cheap

        times = [r.timestamp for r in self.records]
        y = list(range(1, len(self.records) + 1))

        plt.figure(figsize=(12, 6))
        plt.step(times, y, where="post", label="Cumulative Responses", color="black", linewidth=1)

        reinforcement_times = [r.timestamp for r in self.records if r.is_reinforced]
        for rt in reinforcement_times:
            try:
                idx = next(i for i, r in enumerate(self.records) if r.timestamp == rt)
                plt.plot(
                    [rt, rt],
                    [y[idx], y[idx] - (max(y) * 0.05)],
                    color="red",
                    linewidth=1.5,
                )
            except StopIteration:
                continue

        plt.xlabel("Time (seconds)")
        plt.ylabel("Cumulative Responses")
        plt.title(title)
        plt.grid(True, which="both", linestyle="--", alpha=0.5)

        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        plt.savefig(file_path)
        plt.close()


class MultiSessionRecorder:
    """Compare multiple sessions in a single cumulative-record plot."""

    def __init__(self) -> None:
        self.sessions: dict[str, list[ResponseRecord]] = {}

    def add_session(self, label: str, records: list[ResponseRecord]) -> None:
        self.sessions[label] = records

    def save_comparison_png(
        self, file_path: str, title: str = "Session Comparison"
    ) -> None:
        if not self.sessions:
            return

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))

        for label, records in self.sessions.items():
            if not records:
                continue
            start_t = records[0].timestamp
            times = [r.timestamp - start_t for r in records]
            y = list(range(1, len(records) + 1))

            plt.step(times, y, where="post", label=label, linewidth=1.5)

            reinforcement_times = [
                r.timestamp - start_t for r in records if r.is_reinforced
            ]
            for rt in reinforcement_times:
                try:
                    idx = next(
                        i
                        for i, r in enumerate(records)
                        if (r.timestamp - start_t) == rt
                    )
                    plt.plot(
                        [rt, rt], [y[idx], y[idx] - (max(y) * 0.03)], linewidth=1
                    )
                except StopIteration:
                    continue

        plt.xlabel("Normalized Time (seconds)")
        plt.ylabel("Cumulative Responses")
        plt.title(title)
        plt.legend()
        plt.grid(True, which="both", linestyle="--", alpha=0.5)

        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        plt.savefig(file_path)
        plt.close()


class LiveVisualizer:
    """Real-time cumulative-record visualiser using matplotlib's interactive mode."""

    def __init__(self, title: str = "Live Cumulative Record") -> None:
        import matplotlib.pyplot as plt

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        (self.line,) = self.ax.step([], [], "k-", where="post", label="Responses")
        self.pips: list = []
        self.times: list = []
        self.responses: list = []
        self.ax.set_title(title)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Cumulative Responses")
        self.ax.grid(True)

    def update(self, record: ResponseRecord | None) -> None:
        if not record:
            return

        import matplotlib.pyplot as plt

        self.times.append(record.timestamp)
        self.responses.append(len(self.responses) + 1)

        self.line.set_data(self.times, self.responses)

        if record.is_reinforced:
            y_val = self.responses[-1]
            (pip,) = self.ax.plot(
                [record.timestamp, record.timestamp],
                [y_val, y_val - 5 if y_val > 5 else 0],
                "r-",
                linewidth=1,
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
