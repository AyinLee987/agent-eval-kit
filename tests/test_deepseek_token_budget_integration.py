"""Provider prompt estimates must survive tracing and drive the actual request cap."""
from pathlib import Path
import json

import pytest

from adapters import deepseek_public_eval as adapter
from adapters.trace_agent import _LLMProxy
from agent_eval.tracing import TraceRecorder, read_trace
from benchmarks.agent_design.live import AccountedLLM
from tests.test_deepseek_public_eval import Response, response
from tests.test_trace_agent import native


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_official_estimate_controls_recorded_request_and_keeps_failure_usage(native, monkeypatch, tmp_path, finish_reason):
    from agent.retry import TransientLLMError
    from agent.token_budget import TokenBudget, budget_scope, budgeted_chat
    from agent.token_estimation import DeepSeekV4TokenEstimator

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-do-not-log")
    messages = [{"role": "user", "content": "读取订单 ORD-测试，核对状态。"}]
    tools = [{"type": "function", "function": {"name": "read_order", "description": "Read one order.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}}]
    estimate = DeepSeekV4TokenEstimator().estimate(messages, tools, thinking_mode="chat")
    sent = []
    billed = {"prompt_tokens": estimate.raw_tokens, "completion_tokens": 50, "total_tokens": estimate.raw_tokens + 50}
    payload = response(reason=finish_reason, usage=billed)
    monkeypatch.setattr(adapter, "urlopen", lambda request, **kwargs: (sent.append(json.loads(request.data)), Response(payload))[1])
    built = adapter.build_agent(harness_path=str(Path(native.__file__).resolve().parents[1]), max_output_tokens=1024)
    recorder = TraceRecorder(tmp_path, task_id="deepseek-budget")
    wrapped = _LLMProxy(AccountedLLM(built.agent.llm, built.client, recorder), recorder)
    budget = TokenBudget(estimate.tokens + 300)
    try:
        with budget_scope(budget):
            if finish_reason == "length":
                with pytest.raises(TransientLLMError):
                    budgeted_chat(wrapped, messages, tools)
            else:
                budgeted_chat(wrapped, messages, tools)
    finally:
        recorder.close()

    assert len(sent) == 1 and sent[0]["max_tokens"] == 300
    assert sent[0]["thinking"] == {"type": "disabled"}
    assert built.client.calls[0]["input_token_estimate"] == estimate.as_dict()
    assert built.client.calls[0]["requested_max_output_tokens"] == 300
    assert budget.snapshot()["spent_tokens"] == billed["total_tokens"]
    assert budget.snapshot()["uncertain_tokens"] == 0
    trace = read_trace(recorder.path)
    assert trace["complete"] and not trace["diagnostics"]
    endings = [event for event in trace["events"] if event["event"] == "span.end" and event.get("kind") == "llm"]
    assert len(endings) == 1
    assert endings[0]["usage"]["total_tokens"] == billed["total_tokens"]
    assert endings[0]["status"] == ("error" if finish_reason == "length" else "ok")
