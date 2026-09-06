"""Exercise observable grading contracts, including plausible false positives."""

from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from adapters.react_agent_adapter import adapt
from agent_eval.scoring import (
    AnswerRelevancyScorer,
    LLMJudgeScorer,
    RuleScorer,
    ToolUsageScorer,
    TrajectoryJudgeScorer,
    TrajectoryScorer,
)
from agent_eval.types import AgentOutcome, ToolCall, TrajectoryStep


def _outcome(answer="391", calls=(), **overrides):
    values = dict(answer=answer, success=True, stop_reason="finished", steps=len(calls), tokens=0,
                  trajectory=[TrajectoryStep(tool_calls=[call]) for call in calls])
    values.update(overrides)
    return AgentOutcome(**values)


def _call(name, arguments=None, observation="ok", **fields):
    return ToolCall(name=name, arguments={} if arguments is None else arguments,
                    observation=observation, **fields)


def _contract_pass(contract, calls, **overrides):
    return ToolUsageScorer().score({"tool_contract": contract}, _outcome(calls=calls, **overrides))["tool_contract_pass"]


def test_adapter_retains_every_call_and_its_own_result_and_failure():
    record = {
        "thought": "A two-tool batch",
        "action": {"name": "first", "arguments": {"q": "one"}},
        "observation": "first succeeded\nERROR second failed",
        "tool_calls": [
            {"id": "a", "name": "first", "arguments": {"q": "one"},
             "observation": "first succeeded", "ok": True, "status": "succeeded"},
            {"id": "b", "name": "second", "arguments": {"nested": {"value": 2}},
             "observation": "ERROR second failed", "ok": False, "status": "failed", "error": "network failure"},
        ],
    }
    result = SimpleNamespace(answer="done", success=True, stop_reason="finished", steps=1, tokens=0,
                             trajectory=[record])
    outcome = adapt(result)
    assert [call.id for call in outcome.tool_calls] == ["a", "b"]
    assert outcome.tool_calls[1].arguments == {"nested": {"value": 2}}
    assert outcome.tool_calls[1].observation == "ERROR second failed"
    assert outcome.tool_calls[1].status == "failed"
    assert outcome.tool_calls[1].error == "network failure"
    assert outcome.used_tool("second")
    assert outcome.had_error()
    assert outcome.raw is result
    assert not TrajectoryScorer().score({}, outcome)["tool_error_free"]


def test_legacy_action_is_supported_but_an_explicit_empty_batch_is_not_backfilled():
    legacy = {"action": {"name": "await_jobs", "args": {"job_id": "j"}}, "observation": "ready"}
    restored = TrajectoryStep.from_dict(legacy)
    assert restored.calls[0].name == "await_jobs"
    assert restored.calls[0].arguments == {"job_id": "j"}
    assert restored.calls[0].observation == "ready"
    empty = TrajectoryStep.from_dict({**legacy, "tool_calls": []})
    assert empty.calls == []
    assert not _outcome(trajectory=[empty]).used_tool("await_jobs")


def test_serializing_a_legacy_step_does_not_erase_its_call_on_restore():
    step = TrajectoryStep(action=ToolCall("calculator", {"expression": "6*7"}), observation="42")
    restored = TrajectoryStep.from_dict(json.loads(json.dumps(asdict(step))))
    assert restored.calls == step.calls


@pytest.mark.parametrize("arguments", [None, [], "{malformed"])
def test_malformed_call_arguments_are_preserved_for_grading(arguments):
    step = TrajectoryStep.from_dict({"tool_calls": [{"id": "bad", "name": "calculator", "arguments": arguments}]})
    assert step.calls[0].arguments == arguments


@pytest.mark.parametrize("step", [
    TrajectoryStep(action=ToolCall("first"), observation="okay\nERROR: later failure"),
    TrajectoryStep(tool_calls=[], error="fatal execution error"),
    TrajectoryStep(tool_calls=[_call("failed", observation=None, status="timed_out")]),
])
def test_error_detection_includes_legacy_later_lines_explicit_errors_and_status(step):
    assert _outcome(trajectory=[step]).had_error()


@pytest.mark.parametrize("answer", ["3910", "1391", "1,391", "3,910", "391.5", "-391", "−391", "391e1", "3.91e3", "item391"])
def test_numeric_substrings_cannot_match_a_different_numeric_token(answer):
    assert RuleScorer().score({"expect_substrings": ["391"]}, _outcome(answer))["answer_correct"] is False


@pytest.mark.parametrize("answer", ["391", "The answer is 391.", "391.0", "3.91e2", "+391", "391元"])
def test_equivalent_full_numeric_tokens_satisfy_legacy_assertions(answer):
    assert RuleScorer().score({"expect_substrings": ["391"]}, _outcome(answer))["answer_correct"] is True


@pytest.mark.parametrize("task", [{}, {"expect_substrings": []}, {"expect_fields": {}}])
def test_no_answer_assertions_means_unassessed_not_correct(task):
    assert RuleScorer().score(task, _outcome("anything")) == {
        "run_completed": True, "answer_correct": None, "rule_pass": None,
    }


def test_run_completion_requires_success_even_with_a_finished_stop_reason():
    assert RuleScorer().score({"expect_number": 391}, _outcome(success=False)) == {
        "run_completed": False, "answer_correct": True, "rule_pass": False,
    }


@pytest.mark.parametrize("answer,spec,expected", [
    ("1,391", 1391, True),
    ("3.149", {"value": 3.14, "abs_tol": 0.01}, True),
    ("3.151", {"value": 3.14, "abs_tol": 0.01}, False),
    ("100.5", {"value": 100, "rel_tol": 0.01}, True),
    ("-3.91e2", -391, True),
    ("23 * 17 = 391", 391, False),
    ("No numeric answer", 391, False),
    ("1e999999999999999999", 391, False),
])
def test_numeric_answer_contracts_enforce_tolerance_and_unambiguous_extraction(answer, spec, expected):
    assert RuleScorer().score({"expect_number": spec}, _outcome(answer))["answer_correct"] is expected


@pytest.mark.parametrize("spec", [True, {"value": 1, "abs_tol": -1}, {"value": float("nan")}, {"value": 1, "typo": 2}])
def test_invalid_numeric_task_contract_is_a_scorer_error(spec):
    with pytest.raises(ValueError):
        RuleScorer().score({"expect_number": spec}, _outcome("1"))


def test_structured_fields_and_numeric_paths_use_exact_types_and_tolerance():
    task = {"expect_fields": {"/answer/verified": True, "/answer/label": "total"},
            "expect_number": {"path": "/answer/value", "value": 391, "abs_tol": 0.1}}
    good = '{"answer":{"value":391.05,"verified":true,"label":"total"}}'
    assert RuleScorer().score(task, _outcome(good))["answer_correct"] is True
    assert RuleScorer().score(task, _outcome(good.replace('true', '1')))["answer_correct"] is False
    assert RuleScorer().score(task, _outcome("not JSON"))["answer_correct"] is False
    assert RuleScorer().score(task, _outcome('{"answer":{"verified":true,"label":"total"}}'))["answer_correct"] is False


@pytest.mark.parametrize("answer,expected", [
    ('{"items":["b","a"]}', True),
    ('{"items":["a","b","a"]}', True),
    ('{"items":["a"]}', False),
    ('{"items":["a","b","c"]}', False),
    ('{"items":"a,b"}', False),
])
def test_set_contract_checks_membership_without_requiring_order(answer, expected):
    task = {"expect_set": {"path": "/items", "values": ["a", "b"]}}
    assert RuleScorer().score(task, _outcome(answer))["answer_correct"] is expected


def test_empty_sets_are_real_assertions_and_booleans_are_not_numbers():
    assert RuleScorer().score({"expect_set": {"values": []}}, _outcome("[]"))["answer_correct"] is True
    assert RuleScorer().score({"expect_set": {"values": [1]}}, _outcome("[true]"))["answer_correct"] is False


def test_required_parameters_and_forbidden_calls_are_checked_across_the_whole_batch():
    required = {"required": [{"name": "lookup", "arguments": {"filter": {"key": "budget"}}}],
                "forbidden": ["send_email"]}
    correct = _call("lookup", {"filter": {"key": "budget", "extra": 1}}, observation="200")
    assert _contract_pass(required, [correct]) is True
    assert _contract_pass(required, [_call("lookup", {"filter": {"key": "wrong"}})]) is False
    assert _contract_pass(required, [correct, _call("send_email")]) is False
    assert _contract_pass(required, []) is False


def test_pending_or_failed_invocations_do_not_satisfy_a_required_success():
    contract = {"required": [{"name": "lookup"}]}
    assert _contract_pass(contract, [_call("lookup", observation=None, status="pending")]) is False
    assert _contract_pass(contract, [_call("lookup", observation="ERROR unavailable", ok=False)]) is False
    assert _contract_pass({"required": [{"name": "lookup", "successful": False}]},
                          [_call("lookup", observation="ERROR unavailable", ok=False)]) is True


def test_call_limit_counts_wrong_argument_attempts_as_well_as_the_successful_call():
    contract = {"required": [{"name": "lookup", "arguments": {"key": "budget"}, "max_calls": 1}]}
    calls = [_call("lookup", {"key": "wrong"}), _call("lookup", {"key": "budget"})]
    assert _contract_pass(contract, calls) is False
    assert _contract_pass(contract, calls[1:]) is True


def test_call_order_requires_the_source_before_the_target_but_allows_unrelated_work():
    contract = {"order": [["lookup", "calculator"]]}
    assert _contract_pass(contract, [_call("lookup"), _call("irrelevant"), _call("calculator")]) is True
    assert _contract_pass(contract, [_call("calculator"), _call("lookup")]) is False
    assert _contract_pass(contract, [_call("lookup")]) is False


def _dependency():
    return {"dependencies": [{"source": "lookup", "target": "calculator", "argument": "/expression",
                              "result_path": "/value", "match": "contains"}]}


def test_dependencies_require_the_latest_observed_result_in_a_later_model_step():
    first = _call("lookup", observation='{"value":200}')
    latest = _call("lookup", observation='{"value":250}')
    stale = _call("calculator", {"expression": "200 * 3"})
    current = _call("calculator", {"expression": "250 * 3"})
    assert _contract_pass(_dependency(), [first, latest, stale]) is False
    assert _contract_pass(_dependency(), [first, latest, current]) is True
    assert _contract_pass(_dependency(), [first, _call("calculator", {"expression": "1200 * 3"})]) is False
    assert _contract_pass(_dependency(), [first, stale],
                          trajectory=[TrajectoryStep(tool_calls=[first, stale])]) is False


def test_dependencies_can_compare_structured_values_exactly():
    contract = {"dependencies": [{"source": "lookup", "target": "save", "argument": "/payload",
                                  "result_path": "/record"}]}
    source = _call("lookup", observation='{"record":{"value":200}}')
    assert _contract_pass(contract, [source, _call("save", {"payload": {"value": 200}})]) is True
    assert _contract_pass(contract, [source, _call("save", {"payload": {"value": "200"}})]) is False
    assert _contract_pass(contract, [_call("save", {"payload": {"value": 200}})]) is False


def test_a_successful_retry_recovers_process_health_without_erasing_the_error():
    failed = _call("calculator", observation="ERROR bad input", ok=False)
    recovered = _call("calculator", observation="391", ok=True)
    result = TrajectoryScorer().score({"expect_tool": "calculator"}, _outcome(calls=[failed, recovered]))
    assert result == {"trajectory_score": 1.0, "tool_error_free": False, "tool_recovery_pass": True}
    unresolved = TrajectoryScorer().score({}, _outcome(calls=[failed, _call("other")]))
    assert unresolved["tool_recovery_pass"] is False
    assert unresolved["trajectory_score"] < 1.0


def test_declared_expected_errors_require_a_completed_run_and_do_not_excuse_fatal_failures():
    task = {"tool_contract": {"allow_errors": ["calculator"]}}
    failed = _call("calculator", observation="ERROR division by zero", ok=False)
    assert TrajectoryScorer().score(task, _outcome("undefined", calls=[failed]))["tool_recovery_pass"] is True
    assert TrajectoryScorer().score(task, _outcome(calls=[failed], success=False))["tool_recovery_pass"] is False
    fatal = _call("calculator", observation=None, status="fatal", error="invariant failure")
    assert TrajectoryScorer().score(task, _outcome(calls=[fatal]))["tool_recovery_pass"] is False


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, {}])
def test_binary_judges_must_return_actual_booleans(value):
    with pytest.raises(ValueError):
        LLMJudgeScorer(lambda task, outcome: value).score({}, _outcome())


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True, "1"])
def test_judge_wrappers_validate_injected_callbacks_too(value):
    with pytest.raises(ValueError):
        TrajectoryJudgeScorer(lambda task, outcome: {"quality": value}, dimensions=("quality",)).score({}, _outcome())
    with pytest.raises(ValueError):
        AnswerRelevancyScorer(lambda task, outcome: {"answer_relevancy": value}).score({}, _outcome())


def test_metric_names_are_available_before_execution_and_never_include_rationales():
    for scorer in (RuleScorer(), ToolUsageScorer(), TrajectoryScorer(), AnswerRelevancyScorer(lambda task, outcome: {}),
                   LLMJudgeScorer(lambda task, outcome: True)):
        assert scorer.metric_names
        assert not any("rationale" in key for key in scorer.metric_names)

    def judge(task, outcome):
        return {"quality": 0.5, "rationale": "checked"}

    judge.metric_names = ("quality",)
    assert TrajectoryJudgeScorer(judge).metric_names == ("judge_quality",)


def test_legacy_judge_dimensions_are_inferred_once_and_then_required():
    replies = iter([{"quality": 0.5}, {}])
    scorer = TrajectoryJudgeScorer(lambda task, outcome: next(replies))
    assert scorer.metric_names == ()
    assert scorer.score({}, _outcome())["judge_quality"] == 0.5
    assert scorer.metric_names == ("judge_quality",)
    with pytest.raises(ValueError):
        scorer.score({}, _outcome())
