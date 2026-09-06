from dataclasses import replace

from agent_eval.execution import decode_outcome, encode_outcome, execute
from agent_eval.types import AgentOutcome


def adapt_metadata(value):
    return replace(value, metadata={"api_calls": [{"usage": {"prompt_tokens": 5, "completion_tokens": 2}}]})


def adapt_provider_error(value):
    return replace(value, success=False, metadata={"execution_error": {
        "stage": "provider", "type": "HTTPError", "message": "HTTP 429"},
        "api_calls": [{"usage": {"prompt_tokens": 5, "completion_tokens": 2}}, {"usage": None}]})


def test_normalized_provider_usage_survives_spawn_and_offline_decode():
    result = execute({"kind": "agent", "prompt": "x", "condition": {
        "agent": "tests.experiment_fixtures:build_agent",
        "outcome_adapter": "tests.test_provider_metadata:adapt_metadata"}}, timeout_seconds=10)
    assert result["execution_error"] is None
    usage = result["outcome"]["metadata"]["api_calls"][0]["usage"]
    assert usage == {"prompt_tokens": 5, "completion_tokens": 2}
    assert encode_outcome(decode_outcome(result["outcome"]))["metadata"] == result["outcome"]["metadata"]


def test_provider_failure_keeps_prior_usage_and_normalized_evidence():
    result = execute({"kind": "agent", "prompt": "x", "condition": {
        "agent": "tests.experiment_fixtures:build_agent",
        "outcome_adapter": "tests.test_provider_metadata:adapt_provider_error"}}, timeout_seconds=10)
    assert result["execution_error"]["message"] == "HTTP 429"
    assert result["outcome"]["metadata"]["api_calls"][0]["usage"]["prompt_tokens"] == 5
    assert result["outcome"]["answer"] == "42"
