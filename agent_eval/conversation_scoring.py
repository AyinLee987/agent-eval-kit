"""Scorers over a full :class:`agent_eval.types.ConversationOutcome`.

Mirrors :mod:`agent_eval.scoring`'s shape (a scorer is a name + a
``score(task, outcome) -> {metric: value}`` callable) but at the
conversation level — a later turn can only be judged correctly in light of
earlier ones, which is exactly what a per-turn scorer can't see.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Protocol

from .types import ConversationOutcome
from .judge import validate_judge_result

ConversationTask = Dict[str, Any]


class ConversationScorer(Protocol):
    """A named scorer: conversation task + outcome -> {metric_name: value}."""

    name: str

    def score(self, task: ConversationTask, outcome: ConversationOutcome) -> Dict[str, Any]: ...


class ConversationJudgeScorer:
    """LLM-as-judge over a full conversation transcript.

    Wraps a ``judge_fn`` built by
    :func:`agent_eval.judge.build_conversation_judge_fn` — see that
    function for the dimensions it grades by default (knowledge retention,
    conversation completeness) and why the judge model must differ from the
    model that produced the conversation being judged (self-preference
    bias).
    """

    name = "conversation_judge"

    def __init__(self, judge_fn: Callable[[ConversationTask, ConversationOutcome], Dict[str, Any]],
                 dimensions=None) -> None:
        self.judge_fn = judge_fn
        declared = dimensions if dimensions is not None else getattr(judge_fn, "metric_names", None)
        self.dimensions = tuple(declared) if declared is not None else None
        self.metric_names = tuple(f"judge_{key}" for key in self.dimensions or ())

    def score(self, task: ConversationTask, outcome: ConversationOutcome) -> Dict[str, Any]:
        result = validate_judge_result(self.judge_fn(task, outcome), self.dimensions)
        if self.dimensions is None:
            self.dimensions = tuple(key for key in result if key != "rationale")
            self.metric_names = tuple(f"judge_{key}" for key in self.dimensions)
        scores = {f"judge_{key}": value for key, value in result.items() if key != "rationale"}
        scores["judge_rationale"] = result.get("rationale")
        return scores
