"""agent-eval-kit: a framework-agnostic evaluation toolkit for LLM agents.

The core contract is :class:`agent_eval.types.AgentOutcome` — any agent
(ReAct loop, a bare tool-calling loop, LangChain, ...) can be evaluated by
this toolkit as long as its result can be adapted into that shape. Nothing
in ``agent_eval`` imports a specific agent framework.
"""

from .types import AgentOutcome, ToolCall, TrajectoryStep
from .scoring import (
    RuleScorer,
    Scorer,
    TrajectoryScorer,
    ToolUsageScorer,
    LLMJudgeScorer,
)
from .harness import EvalHarness, Scorecard, TaskResult
from .retrieval_metrics import (
    RetrievalCase,
    RetrievalReport,
    evaluate_retrieval,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from .concurrency_bench import ConcurrencyCase, ConcurrencyReport, benchmark

__all__ = [
    "AgentOutcome",
    "ToolCall",
    "TrajectoryStep",
    "RuleScorer",
    "Scorer",
    "TrajectoryScorer",
    "ToolUsageScorer",
    "LLMJudgeScorer",
    "EvalHarness",
    "Scorecard",
    "TaskResult",
    "RetrievalCase",
    "RetrievalReport",
    "evaluate_retrieval",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
    "ConcurrencyCase",
    "ConcurrencyReport",
    "benchmark",
]
