"""Smoke tests covering the package import surface and the CLI entry point."""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_package_imports():
    import ai_operant_box

    assert hasattr(ai_operant_box, "OperantBox")
    assert hasattr(ai_operant_box, "Subject")
    assert hasattr(ai_operant_box, "InductionAgent")


def test_chamber_imports():
    from ai_operant_box.chamber import OperantBox, VirtualClock
    from ai_operant_box.chamber.schedule import ContinuousReinforcement

    assert OperantBox is not None
    assert VirtualClock is not None
    assert ContinuousReinforcement is not None


def test_schedules_imports():
    from ai_operant_box.schedules import (
        InterdependentSchedule,
        RandomIntervalSchedule,
        RandomRatioSchedule,
        VariableIntervalSchedule,
        VariableRatioSchedule,
    )

    assert all(
        x is not None
        for x in (
            InterdependentSchedule,
            RandomIntervalSchedule,
            RandomRatioSchedule,
            VariableIntervalSchedule,
            VariableRatioSchedule,
        )
    )


def test_anthropic_not_required_at_import():
    """Importing the package or the api_client module must not require anthropic."""
    import ai_operant_box.agent.api_client as api_mod
    import ai_operant_box.agent.llm_agent as llm_mod

    assert hasattr(api_mod, "AnthropicClient")
    assert hasattr(llm_mod, "LLMSubject")


def test_cli_version():
    result = subprocess.run(
        [sys.executable, "-m", "ai_operant_box.cli", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "ai-operant-box" in result.stdout


def test_cli_smoke_subcommand():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai_operant_box.cli",
            "smoke",
            "--interval",
            "10",
            "--duration",
            "30",
            "--delta",
            "1",
            "--k",
            "0.05",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "responses=" in result.stdout
    assert "reinforced=" in result.stdout


def test_anthropic_client_lazy_import_skipped_when_missing():
    """If anthropic is not installed, instantiating AnthropicClient must raise RuntimeError."""
    pytest.importorskip
    from ai_operant_box.agent.api_client import AnthropicClient

    try:
        import anthropic  # noqa: F401  # type: ignore[import-not-found]

        anthropic_present = True
    except ImportError:
        anthropic_present = False

    if not anthropic_present:
        with pytest.raises(RuntimeError, match="anthropic"):
            AnthropicClient()
    else:  # pragma: no cover - depends on optional dep
        # If installed but no env var set, RuntimeError should still be raised.
        import os

        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with pytest.raises(RuntimeError):
                AnthropicClient()
        finally:
            if old is not None:
                os.environ["ANTHROPIC_API_KEY"] = old
