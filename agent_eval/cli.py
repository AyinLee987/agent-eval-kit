"""Minimal CLI: run a task set against an adapter and print a scorecard.

    python -m agent_eval run --tasks benchmarks/tasks.json \\
        --agent adapters.bare_baseline:build_agent \\
        --outcome-adapter adapters.bare_baseline:adapt \\
        [--dump results.json]

``--agent`` and ``--outcome-adapter`` are ``module:attribute`` references,
resolved the same way console-script entry points are — this keeps the CLI
usable with any adapter without the toolkit knowing about it upfront.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any, Callable

from .harness import EvalHarness
from .scoring import RuleScorer, ToolUsageScorer, TrajectoryScorer


def _resolve(reference: str) -> Any:
    module_name, _, attr = reference.partition(":")
    if not attr:
        raise ValueError(f"Expected 'module:attribute', got {reference!r}.")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent_eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a task set and print a scorecard.")
    run_parser.add_argument("--tasks", required=True, help="Path to a task set JSON file.")
    run_parser.add_argument("--agent", required=True, help="module:factory returning a fresh agent.")
    run_parser.add_argument(
        "--outcome-adapter",
        required=True,
        help="module:function turning the agent's run result into an AgentOutcome.",
    )
    run_parser.add_argument("--dump", help="Optional path to write the full scorecard as JSON.")

    args = parser.parse_args(argv)

    if args.command == "run":
        build_agent: Callable[[], Any] = _resolve(args.agent)
        outcome_adapter = _resolve(args.outcome_adapter)
        harness = EvalHarness(
            build_agent=build_agent,
            outcome_adapter=outcome_adapter,
            tasks=args.tasks,
            scorers=[RuleScorer(), ToolUsageScorer(), TrajectoryScorer()],
        )
        scorecard = harness.run_all()
        print(scorecard.render())
        if args.dump:
            scorecard.dump(args.dump)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
