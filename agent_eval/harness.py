"""Runs a task set through an agent factory and a set of scorers.

Framework-agnostic by construction: the harness never touches a specific
agent class. It takes a ``build_agent`` factory and an ``outcome_adapter``
that turns whatever that factory's ``run`` method returns into an
:class:`agent_eval.types.AgentOutcome`. Swap either one to evaluate a
different agent with the same task set and scorers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from os import PathLike
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from .scoring import Scorer, Task
from .types import AgentOutcome
from .score_reporting import (
    average, coverage_line, error_record, metric_summary, numeric,
    score_outcome, validate_tasks, write_json_report,
)

TaskSet = Union[str, Sequence[Task]]


@dataclass
class TaskResult:
    """Per-task record: the task id, the normalized outcome, and every

    scorer's output merged into one dict.
    """

    task_id: str
    outcome: AgentOutcome
    scores: Dict[str, Any] = field(default_factory=dict)
    metric_statuses: Dict[str, str] = field(default_factory=dict)
    scorer_errors: List[Dict[str, str]] = field(default_factory=list)
    execution_error: Optional[Dict[str, str]] = None


@dataclass
class Scorecard:
    """Aggregate metrics across a task run.

    Aggregation is metric-name-driven rather than a fixed schema: whatever
    keys the configured scorers produce get averaged (booleans as 0/1,
    ``None`` values excluded from that metric's average rather than counted
    against it). This is what lets a new :class:`~agent_eval.scoring.Scorer`
    plug in without editing this class.
    """

    results: List[TaskResult]

    @property
    def total(self) -> int:
        return len(self.results)

    def aggregate(self) -> Dict[str, float]:
        """Return {metric_name: mean value} over every scored task."""

        return average((r.scores, r.metric_statuses) for r in self.results)

    def metric_summary(self) -> Dict[str, dict]:
        return metric_summary((r.scores, r.metric_statuses) for r in self.results)

    def execution_summary(self) -> Dict[str, Any]:
        errors = sum(r.execution_error is not None for r in self.results)
        success = sum(r.execution_error is None and r.outcome.success for r in self.results)
        return {"planned": self.total, "execution_errors": errors,
                "agent_failed": self.total - errors - success, "successful": success,
                "success_rate": success / self.total if self.total else None,
                "scorer_errors": sum(e["stage"] == "score" for r in self.results for e in r.scorer_errors),
                "unavailable_scorers": sum(e["stage"] != "score" for r in self.results for e in r.scorer_errors)}

    def judge_rule_agreement(self) -> Optional[float]:
        """Fraction of tasks where ``judge_pass`` and ``rule_pass`` agree.

        ``None`` when either scorer wasn't run. See
        :class:`agent_eval.scoring.LLMJudgeScorer` for why this matters.
        """

        pairs = [
            (r.scores["rule_pass"], r.scores["judge_pass"])
            for r in self.results
            if isinstance(r.scores.get("rule_pass"), bool)
            and isinstance(r.scores.get("judge_pass"), bool)
            and r.metric_statuses.get("rule_pass", "valid") == "valid"
            and r.metric_statuses.get("judge_pass", "valid") == "valid"
        ]
        if not pairs:
            return None
        return sum(1 for rule, judge in pairs if bool(rule) == bool(judge)) / len(pairs)

    def render(self) -> str:
        """Return a printable scorecard string."""

        agg = self.aggregate()
        lines = ["=" * 58, "EVAL SCORECARD", "=" * 58]
        header = f"{'Task':<22}" + "".join(f"{key[:9]:<10}" for key in agg)
        lines.append(header)
        lines.append("-" * 58)
        for result in self.results:
            row = f"{result.task_id:<22}"
            for key in agg:
                value = result.scores.get(key)
                cell = f"{'-':<10}" if numeric(value) is None else f"{float(value):<10.2f}"
                row += cell
            lines.append(row)
        lines.append("-" * 58)
        lines.append(f"Execution: {self.execution_summary()}")
        for key, detail in self.metric_summary().items():
            lines.append(coverage_line(key, detail))
        agreement = self.judge_rule_agreement()
        if agreement is not None:
            lines.append(f"judge/rule agreement: {agreement:.0%}")
        lines.append("=" * 58)
        return "\n".join(lines)

    def dump(self, path: str) -> None:
        """Write versioned scores, normalized evidence and failure accounting."""

        payload = {
            "schema_version": 2,
            "scoring_version": 2,
            "aggregate": self.aggregate(),
            "metric_summary": self.metric_summary(),
            "execution_summary": self.execution_summary(),
            "judge_rule_agreement": self.judge_rule_agreement(),
            "results": [
                {
                    "task_id": r.task_id,
                    "answer": r.outcome.answer,
                    "success": r.outcome.success,
                    "stop_reason": r.outcome.stop_reason,
                    "steps": r.outcome.steps,
                    "tokens": r.outcome.tokens,
                    "scores": r.scores,
                    "metric_statuses": r.metric_statuses,
                    "scorer_errors": r.scorer_errors,
                    "execution_error": r.execution_error,
                    "trajectory": [asdict(step) for step in r.outcome.trajectory],
                }
                for r in self.results
            ],
        }
        write_json_report(path, payload)


class EvalHarness:
    """Runs ``build_agent()`` over a task set, scores each run, aggregates."""

    def __init__(
        self,
        build_agent: Callable[[], Any],
        outcome_adapter: Callable[[Any], AgentOutcome],
        tasks: TaskSet,
        scorers: Sequence[Scorer],
        run: Optional[Callable[[Any, str], Any]] = None,
    ) -> None:
        """
        build_agent: zero-arg factory returning a fresh agent per task, so
            state (memory, context) never leaks between tasks.
        outcome_adapter: turns whatever ``run`` returns into an AgentOutcome.
        tasks: a path to a JSON file of task dicts, or the list itself.
        scorers: scorers to run against every task.
        run: how to execute one task against an agent. Defaults to
            ``lambda agent, prompt: agent.run(prompt)`` — override for
            agents with a different entry point.
        """

        self.build_agent = build_agent
        self.outcome_adapter = outcome_adapter
        self.tasks = tasks
        self.scorers = list(scorers)
        self.run = run or (lambda agent, prompt: agent.run(prompt))

    def load_tasks(self) -> List[Task]:
        if isinstance(self.tasks, (str, PathLike)):
            with open(self.tasks, "r", encoding="utf-8") as fh:
                return validate_tasks(json.load(fh))
        return validate_tasks(list(self.tasks))

    def run_all(self) -> Scorecard:
        results: List[TaskResult] = []
        for task in self.load_tasks():
            execution_error = None
            stage = "build"
            try:
                agent = self.build_agent()
                stage = "run"
                raw_result = self.run(agent, task["prompt"])
                stage = "adapt"
                outcome = self.outcome_adapter(raw_result)
                if not isinstance(outcome, AgentOutcome):
                    raise TypeError("Adapter must return AgentOutcome.")
            except Exception as exc:
                execution_error = error_record(stage, exc)
                outcome = AgentOutcome("", False, "execution_error", 0, 0)
            scores, statuses, errors = score_outcome(
                self.scorers, task, outcome,
                unavailable="execution_error" if execution_error else None,
            )
            results.append(TaskResult(
                task_id=task["id"], outcome=outcome, scores=scores,
                metric_statuses=statuses, scorer_errors=errors,
                execution_error=execution_error,
            ))

        return Scorecard(results=results)
