"""Recorded requests retain the actual per-call cap without changing defaults."""

import json

import pytest

from adapters import deepseek_public_eval as adapter
from tests.test_deepseek_public_eval import Response, client, fake_harness, response


def test_recorded_client_sends_and_records_cap_without_mutating_default(monkeypatch):
    requests = []
    monkeypatch.setattr(adapter, "urlopen", lambda req, **kwargs: (requests.append(json.loads(req.data)), Response(response()))[1])
    recorded = client()
    recorded.chat([], max_output_tokens=77)
    recorded.chat([])
    assert [r["max_tokens"] for r in requests] == [77, 2048]
    assert [r["requested_max_output_tokens"] for r in recorded.calls] == [77, 2048]
    assert recorded.max_output_tokens == 2048


def test_native_recorded_llm_preserves_trace_chat_wrapper_and_per_call_limit(fake_harness, monkeypatch):
    requests = []
    monkeypatch.setattr(adapter, "urlopen", lambda req, **kwargs: (requests.append(json.loads(req.data)), Response(response()))[1])
    built = adapter.build_agent(harness_path=str(fake_harness), max_output_tokens=1024)
    llm = built.agent.llm
    traced = []
    original = llm.chat

    def traced_chat(messages, tools=None):
        traced.append(messages)
        return original(messages, tools=tools)

    llm.chat = traced_chat
    llm.chat_with_budget([], max_output_tokens=53)
    llm.chat([])
    assert len(traced) == 2
    assert [r["max_tokens"] for r in requests] == [53, 1024]
    assert llm.max_output_tokens == built.client.max_output_tokens == 1024


def test_recorded_cap_cannot_raise_the_configured_maximum(monkeypatch):
    requests = []
    monkeypatch.setattr(adapter, "urlopen", lambda req, **kwargs: (requests.append(json.loads(req.data)), Response(response()))[1])
    recorded = client()
    recorded.chat([], max_output_tokens=4096)
    assert requests[0]["max_tokens"] == 2048


def test_recorded_length_failure_carries_validated_billing_usage(monkeypatch):
    monkeypatch.setattr(adapter, "urlopen", lambda *args, **kwargs: Response(response(reason="length")))
    recorded = client()
    with pytest.raises(adapter.ProviderCallError) as caught:
        recorded.chat([])
    assert caught.value.usage["total_tokens"] == 120
    assert recorded.calls[0]["usage_complete"]
    assert caught.value.kind == "length"


def test_native_recorded_length_failure_preserves_usage_when_normalizing_error(fake_harness, monkeypatch):
    monkeypatch.setattr(adapter, "urlopen", lambda *args, **kwargs: Response(response(reason="length")))
    built = adapter.build_agent(harness_path=str(fake_harness))
    with pytest.raises(Exception) as caught:
        built.agent.llm.chat_with_budget([], max_output_tokens=30)
    assert type(caught.value).__name__ == "TransientLLMError"
    assert caught.value.usage.prompt_tokens == 100
    assert caught.value.usage.completion_tokens == 20
    assert built.client.calls[0]["usage_complete"]


def test_malformed_provider_usage_is_not_attached_to_an_error_as_known_billing(monkeypatch):
    monkeypatch.setattr(adapter, "urlopen", lambda *args, **kwargs: Response(response(usage={
        "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 119,
    })))
    with pytest.raises(adapter.ProviderCallError) as caught:
        client().chat([])
    assert caught.value.usage is None


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_invalid_recorded_cap_does_not_make_request(monkeypatch, limit):
    monkeypatch.setattr(adapter, "urlopen", lambda *args, **kwargs: pytest.fail("must not send"))
    with pytest.raises(ValueError):
        client().chat([], max_output_tokens=limit)
