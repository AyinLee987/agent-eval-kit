"""Runs a task set through an agent factory and a set of scorers.

Framework-agnostic by construction: the harness never touches a specific
agent class. It takes a ``build_agent`` factory and an ``outcome_adapter``
that turns whatever that factory's ``run`` method returns into an
:class:`agent_eval.types.AgentOutcome`. Swap either one to evaluate a
different agent with the same task set and scorers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from .scoring import Scorer, Task
from .types import AgentOutcome

TaskSet = Union[str, Sequence[Task]]


@dataclass
class TaskResult:
    """Per-task record: the task id, the normalized outcome, and every

    scorer's output merged into one dict.
    """

    task_id: str
    outcome: AgentOutcome
    scores: Dict[str, Any] = field(default_factory=dict)


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

        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for result in self.results:
            for key, value in result.scores.items():
                if value is None:
                    continue
                numeric = float(value) if isinstance(value, (bool, Number)) else None
                if numeric is None:
                    continue
                sums[key] = sums.get(key, 0.0) + numeric
                counts[key] = counts.get(key, 0) + 1
        return {key: sums[key] / counts[key] for key in sums if counts[key]}

    def judge_rule_agreement(self) -> Optional[float]:
        """Fraction of tasks where ``judge_pass`` and ``rule_pass`` agree.

        ``None`` when either scorer wasn't run. See
        :class:`agent_eval.scoring.LLMJudgeScorer` for why this matters.
        """

        pairs = [
            (r.scores["rule_pass"], r.scores["judge_pass"])
            for r in self.results
            if "rule_pass" in r.scores and "judge_pass" in r.scores
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
                cell = f"{'-':<10}" if value is None else f"{float(value):<10.2f}"
                row += cell
            lines.append(row)
        lines.append("-" * 58)
        for key, value in agg.items():
            lines.append(f"avg {key}: {value:.2f}")
        agreement = self.judge_rule_agreement()
        if agreement is not None:
            lines.append(f"judge/rule agreement: {agreement:.0%}")
        lines.append("=" * 58)
        return "\n".join(lines)

    def dump(self, path: str) -> None:
        """Write the full scorecard (per-task scores, no raw trajectories) to a JSON file."""

        payload = {
            "aggregate": self.aggregate(),
            "judge_rule_agreement": self.judge_rule_agreement(),
            "results": [
                {
                    "task_id": r.task_id,
                    "answer": r.outcome.answer,
                    "stop_reason": r.outcome.stop_reason,
                    "steps": r.outcome.steps,
                    "tokens": r.outcome.tokens,
                    "scores": r.scores,
                }
                for r in self.results
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)


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
        if isinstance(self.tasks, str):
            with open(self.tasks, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return list(self.tasks)

    def run_all(self) -> Scorecard:
        results: List[TaskResult] = []
        for task in self.load_tasks():
            agent = self.build_agent()
            raw_result = self.run(agent, task["prompt"])
            outcome = self.outcome_adapter(raw_result)

            scores: Dict[str, Any] = {}
            for scorer in self.scorers:
                scores.update(scorer.score(task, outcome))

            results.append(TaskResult(task_id=task["id"], outcome=outcome, scores=scores))

        return Scorecard(results=results)
