"""Pluggable scorers over :class:`agent_eval.types.AgentOutcome`.

Each scorer is independent and returns a small dict of named metrics for one
task. :class:`agent_eval.harness.EvalHarness` runs whichever scorers it is
given and merges their outputs — add a new scorer without touching the
others.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Protocol

from .types import AgentOutcome

# A task is a plain dict loaded from the task set (see harness.py). Scorers
# only read the keys they care about, so task sets stay framework-agnostic
# too — e.g. {"id": ..., "prompt": ..., "expect_substrings": [...], "expect_tool": ...}
Task = Dict[str, Any]


class Scorer(Protocol):
    """A named scorer: task + outcome -> {metric_name: value}."""

    name: str

    def score(self, task: Task, outcome: AgentOutcome) -> Dict[str, Any]: ...


class RuleScorer:
    """Checks expected substrings appear in the answer and the run finished

    cleanly rather than being force-stopped (hitting the step/token budget,
    an unhandled error, ...).
    """

    name = "rule"

    def __init__(self, finished_value: str = "finished") -> None:
        self.finished_value = finished_value

    def score(self, task: Task, outcome: AgentOutcome) -> Dict[str, Any]:
        answer = outcome.answer.lower()
        substrings = task.get("expect_substrings", [])
        substrings_ok = all(s.lower() in answer for s in substrings)
        return {"rule_pass": substrings_ok and outcome.stop_reason == self.finished_value}


class ToolUsageScorer:
    """Checks whether the task's expected tool was actually invoked.

    Returns ``used_expected_tool=None`` for tasks that don't declare an
    ``expect_tool`` — those are excluded from aggregation rather than
    counted as failures.
    """

    name = "tool_usage"

    def score(self, task: Task, outcome: AgentOutcome) -> Dict[str, Any]:
        expected = task.get("expect_tool")
        if not expected:
            return {"used_expected_tool": None}
        return {"used_expected_tool": outcome.used_tool(expected)}


class TrajectoryScorer:
    """Scores *how* the answer was reached, not just the final output.

    Averages three signals: the run finished cleanly, it used the expected
    tool (when the task declares one), and no step produced a tool error.
    This catches "right answer via the wrong path" that output-only scoring
    misses — e.g. an agent that stumbles onto the right substring after a
    tool call failed.
    """

    name = "trajectory"

    def __init__(self, finished_value: str = "finished", error_marker: str = "ERROR") -> None:
        self.finished_value = finished_value
        self.error_marker = error_marker

    def score(self, task: Task, outcome: AgentOutcome) -> Dict[str, Any]:
        components = [1.0 if outcome.stop_reason == self.finished_value else 0.0]

        expected = task.get("expect_tool")
        if expected:
            components.append(1.0 if outcome.used_tool(expected) else 0.0)

        components.append(0.0 if outcome.had_error(self.error_marker) else 1.0)

        return {"trajectory_score": sum(components) / len(components)}


class LLMJudgeScorer:
    """Optional LLM-as-judge pass, decoupled from any specific LLM client.

    Callers pass a ``judge_fn(task, outcome) -> bool`` closure — this keeps
    the toolkit free of a dependency on any particular model SDK. Report a
    judge/rule agreement rate alongside this (see
    :meth:`agent_eval.harness.Scorecard.judge_rule_agreement`) as a cheap
    calibration proxy: a judge that rarely agrees with a deterministic rule
    check is adding noise, not signal.
    """

    name = "judge"

    def __init__(self, judge_fn: Callable[[Task, AgentOutcome], bool]) -> None:
        self.judge_fn = judge_fn

    def score(self, task: Task, outcome: AgentOutcome) -> Dict[str, Any]:
        return {"judge_pass": self.judge_fn(task, outcome)}
