"""Minimal CLI entry point for ``ai-operant-box``.

Currently exposes only ``--version`` and a ``smoke`` subcommand that runs a
short deterministic scallop-agent session and prints the response count.
Real LLM-driven runs will be added once the schedule integration with
``contingency-py`` lands.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from ai_operant_box.agent.scallop_agent import ScallopingAgent
from ai_operant_box.chamber.environment import OperantBox, VirtualClock
from ai_operant_box.chamber.schedule import FixedInterval


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-operant-box",
        description="LLM-as-subject operant chamber simulator (scaffold CLI).",
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit.")

    sub = parser.add_subparsers(dest="command")

    smoke = sub.add_parser(
        "smoke",
        help="Run a short deterministic scallop-agent session (no LLM).",
    )
    smoke.add_argument("--interval", type=float, default=30.0, help="FI interval (s).")
    smoke.add_argument("--duration", type=float, default=60.0, help="Session duration (s).")
    smoke.add_argument("--delta", type=float, default=1.0, help="Step size (s).")
    smoke.add_argument("--k", type=float, default=0.005, help="Scallop coefficient.")

    return parser


def _read_version() -> str:
    try:
        from importlib.metadata import version  # py>=3.8

        return version("ai-operant-box")
    except Exception:
        return "0.0.0"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"ai-operant-box {_read_version()}")
        return 0

    if args.command == "smoke":
        clock = VirtualClock()
        schedule = FixedInterval(interval=args.interval)
        agent = ScallopingAgent(k=args.k)
        box = OperantBox(subject=agent, schedule=schedule, clock=clock)

        steps = int(args.duration / args.delta)
        for _ in range(steps):
            agent.update_time(args.delta)
            box.step(args.delta)

        responses = len(box.records)
        reinforced = sum(1 for r in box.records if r.is_reinforced)
        print(f"responses={responses} reinforced={reinforced}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
