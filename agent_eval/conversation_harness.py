"""Runs multi-turn conversations and scores them at both the turn level
(reusing the existing single-turn :class:`~agent_eval.scoring.Scorer`
instances) and the conversation level
(:class:`~agent_eval.conversation_scoring.ConversationScorer` instances).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from os import PathLike
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple, Union

from .conversation_scoring import ConversationScorer, ConversationTask
from .scoring import Scorer
from .types import AgentOutcome, ConversationOutcome, Turn
from .score_reporting import (
    average, coverage_line, error_record, metric_summary, numeric,
    score_outcome, validate_tasks, write_json_report,
)

ConversationTaskSet = Union[str, Sequence[ConversationTask]]


@dataclass
class TurnRecord:
    """One turn's normalized outcome plus every turn-scorer's output."""

    turn_index: int
    outcome: AgentOutcome
    scores: Dict[str, Any] = field(default_factory=dict)
    metric_statuses: Dict[str, str] = field(default_factory=dict)
    scorer_errors: List[Dict[str, str]] = field(default_factory=list)
    execution_error: Dict[str, str] | None = None


@dataclass
class ConversationResult:
    """One conversation's full outcome, per-turn records, and

    conversation-level scores.
    """

    conversation_id: str
    outcome: ConversationOutcome
    turn_records: List[TurnRecord] = field(default_factory=list)
    conversation_scores: Dict[str, Any] = field(default_factory=dict)
    metric_statuses: Dict[str, str] = field(default_factory=dict)
    scorer_errors: List[Dict[str, str]] = field(default_factory=list)
    execution_error: Dict[str, str] | None = None

    def __post_init__(self) -> None:
        # Directly constructed scorecards still account for every real turn.
        if not self.turn_records:
            self.turn_records = [TurnRecord(index, turn.outcome)
                                 for index, turn in enumerate(self.outcome.turns, 1)]
        if (len(self.turn_records) != len(self.outcome.turns)
                or [record.turn_index for record in self.turn_records]
                != list(range(1, len(self.outcome.turns) + 1))):
            raise ValueError("Turn records must cover every conversation turn in order.")


def _average_numeric(score_dicts: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    return average((scores, {}) for scores in score_dicts)


@dataclass
class ConversationScorecard:
    """Aggregate metrics across a set of conversations, at both levels."""

    results: List[ConversationResult]

    @property
    def total(self) -> int:
        return len(self.results)

    def aggregate(self) -> Dict[str, float]:
        """Conversation-level metric averages (one value per conversation)."""

        return average((r.conversation_scores, r.metric_statuses) for r in self.results)

    def aggregate_turns(self) -> Dict[str, float]:
        """Turn-level metric averages, pooled across every turn of every
        conversation (not averaged per-conversation first — a conversation
        with more turns contributes proportionally more turn samples).
        """

        return average((t.scores, t.metric_statuses) for r in self.results for t in r.turn_records)

    def metric_summary(self) -> Dict[str, dict]:
        return metric_summary((r.conversation_scores, r.metric_statuses) for r in self.results)

    def turn_metric_summary(self) -> Dict[str, dict]:
        return metric_summary((t.scores, t.metric_statuses) for r in self.results for t in r.turn_records)

    def execution_summary(self) -> Dict[str, Any]:
        records = [t for r in self.results for t in r.turn_records]
        errors = [e for r in self.results for e in r.scorer_errors]
        errors.extend(e for t in records for e in t.scorer_errors)
        success = sum(r.execution_error is None and bool(r.outcome.turns)
                      and all(t.outcome.success for t in r.outcome.turns)
                      for r in self.results)
        return {"planned_conversations": self.total, "successful_conversations": success,
                "success_rate": success / self.total if self.total else None,
                "planned_turns": len(records),
                "blocked_turns": sum(t.outcome.stop_reason == "blocked" for t in records),
                "execution_errors": sum(r.execution_error is not None for r in self.results),
                "scorer_errors": sum(e["stage"] == "score" for e in errors),
                "unavailable_scorers": sum(e["stage"] != "score" for e in errors)}

    def render(self) -> str:
        """Return a printable scorecard: aggregates, then a per-conversation breakdown down to each turn's scores."""

        lines = ["=" * 58, "CONVERSATION SCORECARD", "=" * 58]
        total_turns = sum(len(r.turn_records) for r in self.results)

        lines.append(f"Per-conversation (n={self.total}):")
        for key, detail in self.metric_summary().items():
            lines.append("  " + coverage_line(key, detail))

        lines.append(f"Per-turn, pooled across all turns (n={total_turns}):")
        for key, detail in self.turn_metric_summary().items():
            lines.append("  " + coverage_line(key, detail))
        lines.append(f"Execution: {self.execution_summary()}")

        lines.append("-" * 58)
        for result in self.results:
            lines.append(f"{result.conversation_id} ({len(result.turn_records)} turns):")
            for key, value in result.conversation_scores.items():
                if numeric(value) is not None:
                    lines.append(f"    {key}: {float(value):.2f}")
            for turn in result.turn_records:
                summary = ", ".join(
                    f"{k}={float(v):.2f}" if numeric(v) is not None else f"{k}={v}"
                    for k, v in turn.scores.items()
                )
                lines.append(f"    turn {turn.turn_index}: {summary}")
        lines.append("=" * 58)
        return "\n".join(lines)

    def dump(self, path: str) -> None:
        """Write normalized evidence, scores and explicit failure accounting."""

        payload = {
            "schema_version": 2,
            "scoring_version": 2,
            "aggregate": self.aggregate(),
            "aggregate_turns": self.aggregate_turns(),
            "metric_summary": self.metric_summary(),
            "turn_metric_summary": self.turn_metric_summary(),
            "execution_summary": self.execution_summary(),
            "conversations": [
                {
                    "conversation_id": r.conversation_id,
                    "conversation_scores": r.conversation_scores,
                    "metric_statuses": r.metric_statuses,
                    "scorer_errors": r.scorer_errors,
                    "execution_error": r.execution_error,
                    "turns": [
                        {
                            "turn_index": t.turn_index,
                            "user_message": r.outcome.turns[t.turn_index - 1].user_message,
                            "answer": t.outcome.answer,
                            "success": t.outcome.success,
                            "stop_reason": t.outcome.stop_reason,
                            "trajectory": [asdict(step) for step in t.outcome.trajectory],
                            "scores": t.scores,
                            "metric_statuses": t.metric_statuses,
                            "scorer_errors": t.scorer_errors,
                            "execution_error": t.execution_error,
                        }
                        for t in r.turn_records
                    ],
                }
                for r in self.results
            ],
        }
        write_json_report(path, payload)


class ConversationHarness:
    """Runs a set of multi-turn conversations, scoring each turn and each
    conversation as a whole.

    Unlike :class:`agent_eval.harness.EvalHarness` (a fresh agent per
    task), this builds exactly one agent *per conversation* — turns within
    a conversation deliberately share state, which is the entire point of
    evaluating multi-turn behavior. See
    ``adapters/conversation_history.py`` for how state is threaded through
    an agent whose own ``run()`` method is otherwise stateless per call.
    """

    def __init__(
        self,
        build_agent_and_history: Callable[[], Tuple[Any, Any]],
        outcome_adapter: Callable[[Any], AgentOutcome],
        conversations: ConversationTaskSet,
        turn_scorers: Sequence[Scorer] = (),
        conversation_scorers: Sequence[ConversationScorer] = (),
        run: Any = None,
    ) -> None:
        """
        build_agent_and_history: zero-arg factory returning ``(agent,
            history)`` for one fresh conversation. ``history`` must expose
            ``append_turn(user_message, assistant_message)``.
        outcome_adapter: turns whatever ``run`` returns into an AgentOutcome.
        conversations: a path to a JSON file of conversation dicts
            (``{"id", "turns": [{"prompt", ...}, ...], ...}``), or the list
            itself.
        turn_scorers / conversation_scorers: scorers run per-turn and
            per-conversation respectively; either may be empty.
        run: how to execute one turn against the agent. Defaults to
            ``lambda agent, prompt: agent.run(prompt)``.
        """

        self.build_agent_and_history = build_agent_and_history
        self.outcome_adapter = outcome_adapter
        self.conversations = conversations
        self.turn_scorers = list(turn_scorers)
        self.conversation_scorers = list(conversation_scorers)
        self.run = run or (lambda agent, prompt: agent.run(prompt))

    def load_conversations(self) -> List[ConversationTask]:
        if isinstance(self.conversations, (str, PathLike)):
            with open(self.conversations, "r", encoding="utf-8") as fh:
                return validate_tasks(json.load(fh), conversations=True)
        return validate_tasks(list(self.conversations), conversations=True)

    def run_all(self) -> ConversationScorecard:
        results: List[ConversationResult] = []
        for conversation_task in self.load_conversations():
            execution_error = None
            try:
                agent, history = self.build_agent_and_history()
            except Exception as exc:
                execution_error = error_record("build", exc)
            turns: List[Turn] = []
            turn_records: List[TurnRecord] = []

            for turn_index, turn_spec in enumerate(conversation_task["turns"], start=1):
                prompt = turn_spec["prompt"]
                turn_error = None
                unavailable = "blocked" if execution_error else None
                outcome = AgentOutcome("", False, "blocked", 0, 0)
                if execution_error is None:
                    stage = "run"
                    try:
                        raw_result = self.run(agent, prompt)
                        stage = "adapt"
                        outcome = self.outcome_adapter(raw_result)
                        if not isinstance(outcome, AgentOutcome):
                            raise TypeError("Adapter must return AgentOutcome.")
                        stage = "history"
                        history.append_turn(prompt, outcome.answer)
                    except Exception as exc:
                        turn_error = execution_error = error_record(stage, exc)
                        unavailable = "execution_error"
                        if stage != "history":
                            outcome = AgentOutcome("", False, "execution_error", 0, 0)
                else:
                    turn_error = {"stage": "blocked", "type": "DependencyFailure",
                                  "message": "Conversation setup or an earlier turn failed."}
                turns.append(Turn(user_message=prompt, outcome=outcome))

                scores, statuses, errors = score_outcome(
                    self.turn_scorers, turn_spec, outcome, unavailable=unavailable,
                )
                turn_records.append(TurnRecord(
                    turn_index=turn_index, outcome=outcome, scores=scores,
                    metric_statuses=statuses, scorer_errors=errors, execution_error=turn_error,
                ))

            conversation_outcome = ConversationOutcome(turns=turns)
            conversation_scores, statuses, errors = score_outcome(
                self.conversation_scorers, conversation_task, conversation_outcome,
                unavailable="execution_error" if execution_error else None,
            )

            results.append(
                ConversationResult(
                    conversation_id=conversation_task["id"],
                    outcome=conversation_outcome,
                    turn_records=turn_records,
                    conversation_scores=conversation_scores,
                    metric_statuses=statuses, scorer_errors=errors,
                    execution_error=execution_error,
                )
            )

        return ConversationScorecard(results=results)
