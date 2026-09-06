"""Offline checks: no test in this file contacts the provider."""
import copy
import io
import json
import sys
import types
from urllib.error import HTTPError, URLError

import pytest

from adapters import deepseek_public_eval as adapter
from agent_eval.execution import decode_outcome, encode_outcome


def response(*, content="42", reason="stop", calls=None, usage=None):
    return {"id": "response-1", "model": "deepseek-v4-flash", "system_fingerprint": "fixture",
            "choices": [{"finish_reason": reason,
                         "message": {"content": content, "tool_calls": calls}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                      "prompt_cache_hit_tokens": 40, "prompt_cache_miss_tokens": 60,
                      "completion_tokens_details": {"reasoning_tokens": 0}} if usage is None else usage}


class Response:
    status = 200

    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, limit):
        return json.dumps(self.data).encode()[:limit]


def client():
    return adapter._RecordedClient(api_key="test-key-do-not-log", model="deepseek-v4-flash",
                                   temperature=0, max_output_tokens=2048, timeout_seconds=60,
                                   max_calls=4, max_tokens=20000)


def test_request_explicitly_disables_thinking_and_records_provider_usage(monkeypatch):
    sent = []

    def fake_open(request, timeout):
        sent.append((request, timeout))
        return Response(response())

    monkeypatch.setattr(adapter, "urlopen", fake_open)
    llm = client()
    message, usage = llm.chat([{"role": "user", "content": "6*7"}], [{"type": "function"}])
    request, timeout = sent[0]
    body = json.loads(request.data)
    assert request.full_url == "https://api.deepseek.com/chat/completions"
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == 2048 and body["temperature"] == 0
    assert body["stream"] is False and body["tool_choice"] == "auto"
    assert timeout == 60 and message["content"] == "42"
    assert usage["prompt_cache_hit_tokens"] == 40
    assert usage["prompt_cache_miss_tokens"] == 60
    assert usage["reasoning_tokens"] == 0
    assert llm.calls[0]["finish_reason"] == "stop"
    assert llm.calls[0]["usage_complete"] is True
    assert "test-key-do-not-log" not in json.dumps(llm.calls)


@pytest.mark.parametrize("error,kind", [
    (HTTPError(adapter.ENDPOINT, 429, "test-key-do-not-log", {}, io.BytesIO(b"secret")), "http_429"),
    (TimeoutError("test-key-do-not-log"), "timeout"),
    (URLError("test-key-do-not-log"), "connection_error"),
])
def test_failed_request_has_no_retry_no_secrets_and_unknown_usage(monkeypatch, error, kind):
    attempts = []

    def fake_open(*args, **kwargs):
        attempts.append(1)
        raise error

    monkeypatch.setattr(adapter, "urlopen", fake_open)
    llm = client()
    with pytest.raises(adapter.ProviderCallError) as caught:
        llm.chat([{"role": "user", "content": "test"}])
    assert caught.value.kind == kind
    assert len(attempts) == 1
    assert all(value is None for value in llm.calls[0]["usage"].values())
    assert llm.calls[0]["usage_complete"] is False
    assert "test-key-do-not-log" not in str(caught.value) + json.dumps(llm.calls)


@pytest.mark.parametrize("usage,kind", [
    ({}, "missing_usage"),
    ({"prompt_tokens": True, "completion_tokens": 1, "total_tokens": 2}, "missing_usage"),
    ({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 2}, "invalid_usage"),
    ({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3,
      "prompt_cache_hit_tokens": 1, "prompt_cache_miss_tokens": 1}, "invalid_usage"),
])
def test_malformed_usage_stops_before_another_paid_request(monkeypatch, usage, kind):
    monkeypatch.setattr(adapter, "urlopen", lambda *args, **kwargs: Response(response(usage=usage)))
    llm = client()
    with pytest.raises(adapter.ProviderCallError) as caught:
        llm.chat([{"role": "user", "content": "test"}])
    assert caught.value.kind == kind
    assert len(llm.calls) == 1


@pytest.mark.parametrize("reason", ["length", "content_filter", "insufficient_system_resource"])
def test_abnormal_finish_keeps_known_billed_usage(monkeypatch, reason):
    monkeypatch.setattr(adapter, "urlopen", lambda *args, **kwargs: Response(response(reason=reason)))
    llm = client()
    with pytest.raises(adapter.ProviderCallError):
        llm.chat([{"role": "user", "content": "test"}])
    assert llm.calls[0]["usage"]["total_tokens"] == 120
    assert llm.calls[0]["finish_reason"] == reason
    assert llm.calls[0]["error"]["type"] == reason


def test_unexpected_reasoning_is_not_used_as_final_answer(monkeypatch):
    payload = response()
    payload["choices"][0]["message"]["reasoning_content"] = "hidden reasoning"
    monkeypatch.setattr(adapter, "urlopen", lambda *args, **kwargs: Response(payload))
    llm = client()
    with pytest.raises(adapter.ProviderCallError) as caught:
        llm.chat([{"role": "user", "content": "test"}])
    assert caught.value.kind == "unexpected_thinking"
    assert "hidden reasoning" not in json.dumps(llm.calls)


@pytest.mark.parametrize("expression,expected", [("(100-16)/2", "42.0"), ("2**10", "1024"), ("-3+5%2", "-2")])
def test_calculator_evaluates_arithmetic(expression, expected):
    assert adapter.calculator(expression) == expected


@pytest.mark.parametrize("expression", ["__import__('os').getcwd()", "True+1", "9**9**9", "2**10000", "'abc'", "[1]*5"])
def test_calculator_rejects_code_and_resource_exhaustion(expression):
    with pytest.raises((ValueError, OverflowError)):
        adapter.calculator(expression)


@pytest.fixture
def fake_harness(tmp_path, monkeypatch):
    package = tmp_path / "agent"
    package.mkdir()
    (package / "agent.py").write_text("# fake sibling\n")
    native = types.ModuleType("agent")
    native.__file__ = str(package / "__init__.py")
    native.__path__ = [str(package)]
    native.ToolRegistry = lambda tools: tools
    native.tool = lambda *args, **kwargs: lambda function: function
    native.ReActAgent = lambda **kwargs: types.SimpleNamespace(**kwargs)
    llm = types.ModuleType("agent.llm")
    llm.BaseLLM = object
    llm.LLMResponse = lambda **kwargs: types.SimpleNamespace(**kwargs)
    llm.ToolCall = lambda **kwargs: types.SimpleNamespace(**kwargs)
    llm.Usage = lambda **kwargs: types.SimpleNamespace(**kwargs)
    llm.parse_tool_arguments = lambda text: json.loads(text)
    retry = types.ModuleType("agent.retry")
    retry.TransientLLMError = type("TransientLLMError", (Exception,), {})
    state = types.ModuleType("agent.state")
    state.__path__ = []
    memory = types.ModuleType("agent.state.memory")
    memory.ShortTermMemory = lambda **kwargs: types.SimpleNamespace(**kwargs)
    for module in [native, llm, retry, state, memory]:
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-do-not-log")
    return tmp_path


def test_factory_uses_real_agent_contract_and_preserves_context(fake_harness, monkeypatch):
    monkeypatch.setattr(adapter, "urlopen", lambda *args, **kwargs: pytest.fail("construction must not call API"))
    built = adapter.build_agent(harness_path=str(fake_harness))
    assert built.agent.max_steps == 4
    assert built.agent.max_tokens == 20000
    assert built.agent.short_term.max_tokens == 20000
    assert built.agent.short_term.window == 24
    assert len(built.agent.tools) == 1 and built.agent.tools[0] is adapter.calculator
    assert built.agent.max_tool_retries == 0
    assert built.client.calls == []


def test_factory_does_not_fallback_to_other_provider_key(fake_harness, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider")
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        adapter.build_agent(harness_path=str(fake_harness))


def test_adapter_preserves_trajectory_and_metadata_through_persistence(monkeypatch):
    monkeypatch.setattr(adapter, "urlopen", lambda *args, **kwargs: Response(response()))
    llm = client()
    llm.chat([{"role": "user", "content": "test"}])
    trajectory = [{"thought": "Checked arithmetic", "tool_calls": [
        {"id": "call-1", "name": "calculator", "arguments": {"expression": "6*7"},
         "observation": "42", "status": "completed", "ok": True}]}]
    original = copy.deepcopy(trajectory)
    result = types.SimpleNamespace(answer="42", success=True, stop_reason="finished", steps=2,
                                   tokens=999, trajectory=trajectory)
    wrapper = adapter._RecordedAgent(types.SimpleNamespace(run=lambda prompt: result), client())
    wrapper.agent.run = lambda prompt: (wrapper.client.chat([{"role": "user", "content": prompt}]), result)[1]
    normalized = adapter.adapt(wrapper.run("test"))
    persisted = decode_outcome(encode_outcome(normalized))
    assert normalized.tokens == 120
    assert trajectory == original
    assert persisted.tool_calls[0].id == "call-1"
    assert persisted.metadata["api_calls"][0]["usage"]["prompt_cache_hit_tokens"] == 40
    assert persisted.metadata["execution_error"] is None
    with pytest.raises(RuntimeError, match="single-use"):
        wrapper.run("second sample")


@pytest.mark.parametrize("reason,execution_error", [("length", False), ("content_filter", False),
                                                     ("insufficient_system_resource", True)])
def test_generation_stop_and_provider_outage_remain_distinct(monkeypatch, reason, execution_error):
    monkeypatch.setattr(adapter, "urlopen", lambda *args, **kwargs: Response(response(reason=reason)))
    llm = client()
    fake = types.SimpleNamespace(run=lambda prompt: llm.chat([{"role": "user", "content": prompt}]))
    normalized = adapter.adapt(adapter._RecordedAgent(fake, llm).run("test"))
    assert normalized.success is False
    assert normalized.tokens == 120
    assert normalized.metadata["provider_error"]["type"] == reason
    assert bool(normalized.metadata["execution_error"]) is execution_error
    if not execution_error:
        assert normalized.answer == "42"


@pytest.mark.parametrize("overrides", [{"model": "deepseek-v4-pro"}, {"temperature": True},
                                       {"max_steps": 0}, {"timeout_seconds": float("nan")}])
def test_invalid_configuration_is_rejected_before_api_use(fake_harness, overrides):
    with pytest.raises(ValueError):
        adapter.build_agent(harness_path=str(fake_harness), **overrides)
