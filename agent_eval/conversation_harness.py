"""Runs multi-turn conversations and scores them at both the turn level
(reusing the existing single-turn :class:`~agent_eval.scoring.Scorer`
instances) and the conversation level
(:class:`~agent_eval.conversation_scoring.ConversationScorer` instances).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from numbers import Number
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple, Union

from .conversation_scoring import ConversationScorer, ConversationTask
from .scoring import Scorer
from .types import AgentOutcome, ConversationOutcome, Turn

ConversationTaskSet = Union[str, Sequence[ConversationTask]]


@dataclass
class TurnRecord:
    """One turn's normalized outcome plus every turn-scorer's output."""

    turn_index: int
    outcome: AgentOutcome
    scores: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationResult:
    """One conversation's full outcome, per-turn records, and

    conversation-level scores.
    """

    conversation_id: str
    outcome: ConversationOutcome
    turn_records: List[TurnRecord] = field(default_factory=list)
    conversation_scores: Dict[str, Any] = field(default_factory=dict)


def _average_numeric(score_dicts: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for scores in score_dicts:
        for key, value in scores.items():
            if value is None or not isinstance(value, (bool, Number)):
                continue
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sums if counts[key]}


@dataclass
class ConversationScorecard:
    """Aggregate metrics across a set of conversations, at both levels."""

    results: List[ConversationResult]

    @property
    def total(self) -> int:
        return len(self.results)

    def aggregate(self) -> Dict[str, float]:
        """Conversation-level metric averages (one value per conversation)."""

        return _average_numeric(r.conversation_scores for r in self.results)

    def aggregate_turns(self) -> Dict[str, float]:
        """Turn-level metric averages, pooled across every turn of every
        conversation (not averaged per-conversation first — a conversation
        with more turns contributes proportionally more turn samples).
        """

        return _average_numeric(t.scores for r in self.results for t in r.turn_records)

    def render(self) -> str:
        """Return a printable scorecard: aggregates, then a per-conversation breakdown down to each turn's scores."""

        lines = ["=" * 58, "CONVERSATION SCORECARD", "=" * 58]
        total_turns = sum(len(r.turn_records) for r in self.results)

        lines.append(f"Per-conversation (n={self.total}):")
        for key, value in self.aggregate().items():
            lines.append(f"  avg {key}: {value:.2f}")

        lines.append(f"Per-turn, pooled across all turns (n={total_turns}):")
        for key, value in self.aggregate_turns().items():
            lines.append(f"  avg {key}: {value:.2f}")

        lines.append("-" * 58)
        for result in self.results:
            lines.append(f"{result.conversation_id} ({len(result.turn_records)} turns):")
            for key, value in result.conversation_scores.items():
                if isinstance(value, (bool, Number)):
                    lines.append(f"    {key}: {float(value):.2f}")
            for turn in result.turn_records:
                summary = ", ".join(
                    f"{k}={float(v):.2f}" if isinstance(v, (bool, Number)) else f"{k}={v}"
                    for k, v in turn.scores.items()
                )
                lines.append(f"    turn {turn.turn_index}: {summary}")
        lines.append("=" * 58)
        return "\n".join(lines)

    def dump(self, path: str) -> None:
        """Write the full scorecard (every turn's scores, no raw trajectories) to a JSON file."""

        payload = {
            "aggregate": self.aggregate(),
            "aggregate_turns": self.aggregate_turns(),
            "conversations": [
                {
                    "conversation_id": r.conversation_id,
                    "conversation_scores": r.conversation_scores,
                    "turns": [
                        {
                            "turn_index": t.turn_index,
                            "user_message": r.outcome.turns[t.turn_index - 1].user_message,
                            "answer": t.outcome.answer,
                            "scores": t.scores,
                        }
                        for t in r.turn_records
                    ],
                }
                for r in self.results
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)


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
        if isinstance(self.conversations, str):
            with open(self.conversations, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return list(self.conversations)

    def run_all(self) -> ConversationScorecard:
        results: List[ConversationResult] = []
        for conversation_task in self.load_conversations():
            agent, history = self.build_agent_and_history()
            turns: List[Turn] = []
            turn_records: List[TurnRecord] = []

            for turn_index, turn_spec in enumerate(conversation_task["turns"], start=1):
                prompt = turn_spec["prompt"]
                raw_result = self.run(agent, prompt)
                outcome = self.outcome_adapter(raw_result)
                turns.append(Turn(user_message=prompt, outcome=outcome))

                scores: Dict[str, Any] = {}
                for scorer in self.turn_scorers:
                    scores.update(scorer.score(turn_spec, outcome))
                turn_records.append(TurnRecord(turn_index=turn_index, outcome=outcome, scores=scores))

                history.append_turn(prompt, outcome.answer)

            conversation_outcome = ConversationOutcome(turns=turns)
            conversation_scores: Dict[str, Any] = {}
            for scorer in self.conversation_scorers:
                conversation_scores.update(scorer.score(conversation_task, conversation_outcome))

            results.append(
                ConversationResult(
                    conversation_id=conversation_task["id"],
                    outcome=conversation_outcome,
                    turn_records=turn_records,
                    conversation_scores=conversation_scores,
                )
            )

        return ConversationScorecard(results=results)
