"""Canonical Ferster & Skinner cumulative-record tests for ai-operant-box.

Reference: Lattal (2004), JEAB 82:329-355.
"""

from __future__ import annotations

from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import pytest  # noqa: E402

pytest.importorskip("session_analyzer")

from ai_operant_box.analytics.record import (  # noqa: E402
    CumulativeRecorder,
    MultiSessionRecorder,
)
from ai_operant_box.chamber.environment import ResponseRecord  # noqa: E402


def _record(t: float, reinforced: bool = False) -> ResponseRecord:
    return ResponseRecord(
        timestamp=t, sd=None, response="lever", is_reinforced=reinforced
    )


@pytest.mark.unit
class TestCumulativeRecorderDelegation:
    """CumulativeRecorder must delegate rendering to session_analyzer so that
    the canonical pip/reset/event-pen conventions are applied uniformly."""

    def test_save_png_calls_canonical_renderer(self, tmp_path) -> None:
        records = [_record(0.1), _record(0.2, reinforced=True), _record(0.3)]
        rec = CumulativeRecorder(records)
        out = tmp_path / "single.png"
        import ai_operant_box.analytics.record as record_mod

        real = record_mod.build_cumulative_record_figure
        with patch(
            "ai_operant_box.analytics.record.build_cumulative_record_figure",
            side_effect=real,
        ) as mock_build:
            rec.save_png(str(out))
        assert mock_build.called
        call = mock_build.call_args
        # Positional or kw: response_times and reinforcement_times must be
        # passed with session-relative seconds.
        kwargs = call.kwargs
        args = call.args
        resp = kwargs.get("response_times", args[0] if args else None)
        reinf = kwargs.get("reinforcement_times", args[1] if len(args) > 1 else None)
        # Times are rebased to session start (first response = t=0).
        assert list(resp) == pytest.approx([0.0, 0.1, 0.2])
        assert list(reinf) == pytest.approx([0.1])

    def test_save_png_produces_valid_file(self, tmp_path) -> None:
        records = [_record(0.1), _record(0.2, reinforced=True), _record(0.3)]
        rec = CumulativeRecorder(records)
        out = tmp_path / "out.png"
        rec.save_png(str(out))
        assert out.exists()
        assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_save_png_writes_title(self, tmp_path) -> None:
        records = [_record(0.1), _record(0.2, reinforced=True), _record(0.3)]
        rec = CumulativeRecorder(records)
        import ai_operant_box.analytics.record as record_mod

        real = record_mod.build_cumulative_record_figure
        titles: list = []

        def spy(*args, **kwargs):
            built = real(*args, **kwargs)
            fig, main_ax = built[0], built[1]
            original_clear = fig.clear

            def recording_clear(*a, **kw):
                texts = [main_ax.get_title()]
                if fig._suptitle is not None:
                    texts.append(fig._suptitle.get_text())
                titles.append(texts)
                return original_clear(*a, **kw)

            fig.clear = recording_clear
            return built

        out = tmp_path / "titled.png"
        with patch(
            "ai_operant_box.analytics.record.build_cumulative_record_figure",
            side_effect=spy,
        ):
            rec.save_png(str(out), title="CRF, haiku")
        # One title on the record, replacing the default "Cumulative Record".
        assert titles == [["CRF, haiku"]]

        from PIL import Image

        with Image.open(out) as img:
            assert img.info.get("Title") == "CRF, haiku"

    def test_empty_records_skip_render(self, tmp_path) -> None:
        rec = CumulativeRecorder([])
        with patch(
            "ai_operant_box.analytics.record.build_cumulative_record_figure"
        ) as mock_build:
            rec.save_png(str(tmp_path / "empty.png"))
        assert not mock_build.called


@pytest.mark.unit
class TestMultiSessionRecorderDelegation:
    """MultiSessionRecorder must also delegate to session_analyzer."""

    def test_save_comparison_png_uses_canonical_renderer(self, tmp_path) -> None:
        rec = MultiSessionRecorder()
        rec.add_session("A", [_record(0.1), _record(0.2, reinforced=True)])
        rec.add_session("B", [_record(0.1), _record(0.3, reinforced=True)])
        out = tmp_path / "multi.png"
        # Wrap the real renderer so we can assert it was called per-session
        # while still getting back valid Figure/Axes objects to consume.
        import ai_operant_box.analytics.record as record_mod

        real = record_mod.build_cumulative_record_figure
        calls: list[tuple] = []

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return real(*args, **kwargs)

        with patch(
            "ai_operant_box.analytics.record.build_cumulative_record_figure",
            side_effect=spy,
        ):
            rec.save_comparison_png(str(out))
        assert len(calls) == 2  # one canonical build per session

    def test_save_comparison_png_produces_valid_file(self, tmp_path) -> None:
        rec = MultiSessionRecorder()
        rec.add_session("A", [_record(0.1), _record(0.2, reinforced=True)])
        rec.add_session("B", [_record(0.1), _record(0.3, reinforced=True)])
        out = tmp_path / "out.png"
        rec.save_comparison_png(str(out))
        assert out.exists()
        assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
