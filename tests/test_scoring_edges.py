from agent_eval.scoring import RuleScorer
from agent_eval.types import AgentOutcome


def test_structured_answers_cannot_override_a_wrong_field_with_a_duplicate_key():
    result = AgentOutcome('{"sum": 0, "sum": 4}', True, "finished", 1, 1)
    assert RuleScorer().score({"expect_fields": {"/sum": 4}}, result)["answer_correct"] is False


def test_structured_answer_rejects_nonfinite_numbers_in_unchecked_fields():
    result = AgentOutcome('{"sum": 4, "extra": 1e999}', True, "finished", 1, 1)
    assert RuleScorer().score({"expect_fields": {"/sum": 4}}, result)["answer_correct"] is False
