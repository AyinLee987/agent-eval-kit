"""Tests for the multi-turn conversation harness, using a fake agent that

reads its own conversation history back from a ConversationHistoryProvider
— exercising the exact statefulness contract ConversationHarness relies on,
without any real LLM or network call.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from adapters.conversation_history import ConversationHistoryProvider
from agent_eval.conversation_harness import ConversationHarness
from agent_eval.conversation_scoring import ConversationJudgeScorer
from agent_eval.judge import build_conversation_judge_fn
from agent_eval.scoring import RuleScorer
from agent_eval.types import AgentOutcome


class _FakeResult:
    def __init__(self, answer: str) -> None:
        self.answer = answer


class _RecallingFakeAgent:
    """Answers "what did I tell you?" from its own injected history;

    otherwise just echoes the prompt back, tagging whether it saw any
    prior turns at all (so a test can assert history was actually threaded
    through, not just present in the harness's own bookkeeping).
    """

    def __init__(self, history: ConversationHistoryProvider) -> None:
        self.history = history

    def run(self, prompt: str) -> _FakeResult:
        prior = self.history.prepare(prompt)
        if "what did i tell you" in prompt.lower():
            first_user_message = next((m["content"] for m in prior if m["role"] == "user"), None)
            return _FakeResult(f"You told me: {first_user_message}" if first_user_message else "You told me nothing yet.")
        return _FakeResult(f"ack[{len(prior)} prior messages]: {prompt}")


def _adapt(result: _FakeResult) -> AgentOutcome:
    return AgentOutcome(answer=result.answer, success=True, stop_reason="finished", steps=1, tokens=1)


def _build_agent_and_history():
    history = ConversationHistoryProvider()
    return _RecallingFakeAgent(history), history


def test_history_is_threaded_across_turns():
    conversations = [
        {
            "id": "recall-check",
            "turns": [
                {"prompt": "My favorite color is teal."},
                {"prompt": "What did I tell you?", "expect_substrings": ["teal"]},
            ],
        }
    ]
    harness = ConversationHarness(
        build_agent_and_history=_build_agent_and_history,
        outcome_adapter=_adapt,
        conversations=conversations,
        turn_scorers=[RuleScorer()],
    )

    scorecard = harness.run_all()

    assert scorecard.total == 1
    turns = scorecard.results[0].turn_records
    assert len(turns) == 2
    assert "teal" in turns[1].outcome.answer
    assert turns[1].scores == {
        "run_completed": True, "answer_correct": True, "rule_pass": True,
    }


def test_each_conversation_gets_a_fresh_agent_and_history():
    conversations = [
        {"id": "conv-a", "turns": [{"prompt": "hello"}]},
        {"id": "conv-b", "turns": [{"prompt": "hello again"}]},
    ]
    harness = ConversationHarness(
        build_agent_and_history=_build_agent_and_history,
        outcome_adapter=_adapt,
        conversations=conversations,
    )

    scorecard = harness.run_all()

    # Both conversations' first turn sees zero prior messages — a leaked
    # history object between conversations would show up as a nonzero count
    # on conv-b's single turn.
    for result in scorecard.results:
        assert "ack[0 prior messages]" in result.turn_records[0].outcome.answer


def test_conversation_level_judge_scorer_is_namespaced_and_aggregated():
    conversations = [
        {"id": "c1", "turns": [{"prompt": "hi"}, {"prompt": "bye"}]},
        {"id": "c2", "turns": [{"prompt": "hi"}]},
    ]

    def fake_judge(task: Dict[str, Any], outcome) -> Dict[str, Any]:
        # Score improves with more turns, just to get two different values.
        return {"knowledge_retention": 0.5 + 0.1 * len(outcome.turns), "rationale": "synthetic"}

    judge_fn = fake_judge  # already matches the (task, outcome) -> dict shape
    harness = ConversationHarness(
        build_agent_and_history=_build_agent_and_history,
        outcome_adapter=_adapt,
        conversations=conversations,
        conversation_scorers=[ConversationJudgeScorer(judge_fn)],
    )

    scorecard = harness.run_all()

    assert scorecard.results[0].conversation_scores == {
        "judge_knowledge_retention": 0.7,
        "judge_rationale": "synthetic",
    }
    assert scorecard.results[1].conversation_scores == {
        "judge_knowledge_retention": 0.6,
        "judge_rationale": "synthetic",
    }
    assert scorecard.aggregate()["judge_knowledge_retention"] == pytest.approx(0.65)


def test_scorecard_renders_and_dumps(tmp_path):
    conversations = [{"id": "c1", "turns": [{"prompt": "hi", "expect_substrings": ["ack"]}]}]
    harness = ConversationHarness(
        build_agent_and_history=_build_agent_and_history,
        outcome_adapter=_adapt,
        conversations=conversations,
        turn_scorers=[RuleScorer()],
    )

    scorecard = harness.run_all()
    rendered = scorecard.render()
    assert "CONVERSATION SCORECARD" in rendered
    assert "c1" in rendered

    dump_path = tmp_path / "conversations.json"
    scorecard.dump(str(dump_path))
    assert dump_path.exists()


def test_build_conversation_judge_fn_uses_expected_intentions_in_prompt():
    captured: List[Dict[str, str]] = []

    def fake_chat(messages: List[Dict[str, str]]) -> str:
        captured.extend(messages)
        return '{"knowledge_retention": 1.0, "conversation_completeness": 1.0, "rationale": "ok"}'

    judge_fn = build_conversation_judge_fn(fake_chat)
    task = {"expected_intentions": ["remember the favorite color"]}

    result = judge_fn(task, _FakeConversationOutcome())

    assert result == {"knowledge_retention": 1.0, "conversation_completeness": 1.0, "rationale": "ok"}
    assert "remember the favorite color" in captured[-1]["content"]


class _FakeConversationOutcome:
    """Minimal stand-in with the one attribute render_conversation needs."""

    turns: list = []
