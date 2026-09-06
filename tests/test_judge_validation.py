import json

import pytest

from agent_eval.judge import (
    JudgeValidationError, build_answer_relevancy_judge_fn,
    build_conversation_judge_fn, build_llm_judge_fn, render_conversation,
    validate_judge_result,
)
from agent_eval.types import AgentOutcome, ConversationOutcome, TrajectoryStep, Turn


@pytest.mark.parametrize("value", [True, False, "0.5", None, -0.1, 1.1, 10 ** 400, float("nan"), float("inf")])
def test_judge_rejects_non_numeric_or_invalid_scores(value):
    with pytest.raises(JudgeValidationError):
        validate_judge_result({"quality": value}, ("quality",))


@pytest.mark.parametrize("result", [{}, {"rationale": "great"}, [],
                                    {"quality": 1, "extra": 1},
                                    {"quality": 1, "rationale": 42}])
def test_judge_rejects_missing_or_malformed_schema(result):
    with pytest.raises(JudgeValidationError):
        validate_judge_result(result, ("quality",))


def test_judge_allows_numeric_endpoints_and_optional_text_rationale():
    assert validate_judge_result({"a": 0, "b": 1}, ("a", "b")) == {
        "a": 0.0, "b": 1.0, "rationale": "",
    }


@pytest.mark.parametrize("reply", [
    '{"quality": 0, "quality": 1}',
    '{"quality": 0} {"quality": 1}',
    '{"quality": NaN}',
    '{"rationale": "no dimensions"}',
    '[{"quality": 1}]',
    'prefix {"quality": 1}',
    '```json\n[{"quality": 1}]\n```',
])
def test_trajectory_judge_never_selects_a_favourable_or_incomplete_json(reply):
    judge = build_llm_judge_fn(lambda _: reply, dimensions=("quality",))
    outcome = AgentOutcome("answer", True, "finished", 1, 1)
    with pytest.raises(JudgeValidationError):
        judge({"prompt": "q"}, outcome)


def test_all_builders_declare_and_enforce_complete_dimensions():
    outcome = AgentOutcome("answer", True, "finished", 1, 1)
    relevancy = build_answer_relevancy_judge_fn(lambda _: '{"answer_relevancy": true}')
    conversation = build_conversation_judge_fn(lambda _: '{"knowledge_retention": 1}')
    assert relevancy.metric_names == ("answer_relevancy",)
    with pytest.raises(JudgeValidationError):
        relevancy({"prompt": "q"}, outcome)
    with pytest.raises(JudgeValidationError):
        conversation({}, ConversationOutcome([Turn("q", outcome)]))


def test_conversation_grounding_has_call_results_and_failures_as_evidence():
    def conversation(calls):
        step = TrajectoryStep.from_dict({"tool_calls": calls})
        return ConversationOutcome([Turn("q", AgentOutcome(
            "The source says X", True, "finished", 1, 1, [step],
        ))])
    absent = render_conversation(conversation([]))
    supported = render_conversation(conversation([
        {"id": "c1", "name": "search", "arguments": {"query": "q"},
         "observation": "source_id=doc-7: X", "ok": True},
    ]))
    failed = render_conversation(conversation([
        {"id": "c1", "name": "search", "arguments": {"query": "q"},
         "observation": "ERROR: unavailable", "ok": False},
    ]))
    assert "doc-7" in supported and "doc-7" not in absent
    assert "ERROR: unavailable" in failed
    assert len({absent, supported, failed}) == 3
