"""agent-eval-kit: a framework-agnostic evaluation toolkit for LLM agents.

The core contract is :class:`agent_eval.types.AgentOutcome` — any agent
(ReAct loop, a bare tool-calling loop, LangChain, ...) can be evaluated by
this toolkit as long as its result can be adapted into that shape. Nothing
in ``agent_eval`` imports a specific agent framework.
"""

from .types import AgentOutcome, ConversationOutcome, ToolCall, TrajectoryStep, Turn
from .scoring import (
    AnswerRelevancyScorer,
    RuleScorer,
    Scorer,
    TrajectoryScorer,
    ToolUsageScorer,
    LLMJudgeScorer,
    TrajectoryJudgeScorer,
)
from .conversation_scoring import ConversationJudgeScorer, ConversationScorer
from .judge import (
    build_answer_relevancy_judge_fn,
    build_answer_relevancy_prompt,
    build_conversation_judge_fn,
    build_conversation_judge_prompt,
    build_judge_prompt,
    build_llm_judge_fn,
    render_conversation,
    render_trajectory,
)
from .harness import EvalHarness, Scorecard, TaskResult
from .conversation_harness import (
    ConversationHarness,
    ConversationResult,
    ConversationScorecard,
    TurnRecord,
)
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
    "ConversationOutcome",
    "ToolCall",
    "TrajectoryStep",
    "Turn",
    "AnswerRelevancyScorer",
    "RuleScorer",
    "Scorer",
    "TrajectoryScorer",
    "ToolUsageScorer",
    "LLMJudgeScorer",
    "TrajectoryJudgeScorer",
    "ConversationJudgeScorer",
    "ConversationScorer",
    "build_answer_relevancy_judge_fn",
    "build_answer_relevancy_prompt",
    "build_conversation_judge_fn",
    "build_conversation_judge_prompt",
    "build_judge_prompt",
    "build_llm_judge_fn",
    "render_conversation",
    "render_trajectory",
    "EvalHarness",
    "Scorecard",
    "TaskResult",
    "ConversationHarness",
    "ConversationResult",
    "ConversationScorecard",
    "TurnRecord",
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
