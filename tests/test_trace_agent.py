"""Offline integration checks exercise the actual companion state machine."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from adapters.trace_agent import adapt_traced, trace_agent
from agent_eval.tracing import TraceRecorder, read_trace
from agent_eval.types import AgentOutcome


@pytest.fixture
def native(monkeypatch):
    root = Path(__file__).resolve().parents[2] / "agent" / "agent-harness-from-scratch"
    if root.is_dir():
        monkeypatch.syspath_prepend(str(root))
    monkeypatch.setenv("AGENT_LOG_PER_RUN", "false")
    return pytest.importorskip("agent")


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def chat(self, messages, tools=None):
        self.requests.append(copy.deepcopy({"messages": messages, "tools": tools}))
        result = next(self.responses)
        if isinstance(result, BaseException):
            raise result
        return result


def response(content=None, calls=None):
    from agent.llm import LLMResponse, Usage
    return LLMResponse(content=content, tool_calls=calls or [], usage=Usage(10, 2))


def call(name="lookup", arguments=None, call_id="call-1"):
    from agent.llm import ToolCall
    return ToolCall(call_id, name, arguments or {})


def events(recorder, event_type, kind=None):
    loaded = read_trace(recorder.path)
    return [event for event in loaded["events"]
            if event["event"] == event_type and (kind is None or event.get("kind") == kind)]


def test_real_calls_preserve_outcome_and_record_inputs_usage_parentage(native, tmp_path):
    @native.tool
    def lookup() -> str:
        return "42"

    llm = ScriptedLLM([response(calls=[call()]), response("42")])
    agent = native.ReActAgent(llm, native.ToolRegistry([lookup]))
    original_dispatcher = agent._loop._dispatcher
    recorder = TraceRecorder(tmp_path, task_id="real-calls")
    wrapped = trace_agent(agent, recorder)
    raw = wrapped.run("Find the answer")
    outcome = adapt_traced(raw)
    recorder.close()
    assert outcome.answer == "42" and outcome.success
    assert outcome.raw is raw and outcome.used_tool("lookup")
    assert outcome.metadata["trace_id"] == recorder.trace_id
    assert agent.llm is llm and agent._loop._dispatcher is original_dispatcher
    assert read_trace(recorder.path)["complete"]
    starts = events(recorder, "span.start", "llm")
    assert [event["input"] for event in starts] == llm.requests
    assert all(event["usage"]["total_tokens"] == 12 for event in events(recorder, "span.end", "llm"))
    tool_start = events(recorder, "span.start", "tool")[0]
    attempt_start = events(recorder, "span.start", "tool_attempt")[0]
    assert tool_start["metadata"]["tool_call_id"] == "call-1"
    assert attempt_start["parent_span_id"] == tool_start["span_id"]
    assert all(event["duration_ms"] >= 0 for event in events(recorder, "span.end"))


@pytest.mark.parametrize("backend", ["legacy", "fixed", "bounded"])
@pytest.mark.parametrize("proxy_depth", [1, 2])
def test_trace_proxy_preserves_actual_output_shrink_capability(native, tmp_path, backend, proxy_depth):
    from adapters.trace_agent import _LLMProxy
    from agent.llm import LLMResponse, Usage
    from agent.token_budget import BudgetExceeded, TokenBudget, budget_scope, budgeted_chat

    class Legacy:
        max_output_tokens = 1024

        def __init__(self):
            self.calls = 0
            self.requested_caps = []

        def estimate_input_tokens(self, messages, tools=None):
            return 757

        def chat(self, messages, tools=None):
            self.calls += 1
            return LLMResponse(content="done", usage=Usage(757, 150))

    class Bounded(Legacy):
        def chat_with_budget(self, messages, tools=None, *, max_output_tokens):
            self.requested_caps.append(max_output_tokens)
            return self.chat(messages, tools=tools)

    class Fixed(Bounded):
        supports_output_shrink = False

    target = {"legacy": Legacy, "fixed": Fixed, "bounded": Bounded}[backend]()
    recorder = TraceRecorder(tmp_path, task_id="shrink-capability")
    proxy = target
    for _ in range(proxy_depth):
        proxy = _LLMProxy(proxy, recorder)
    budget = TokenBudget(6000, initial_tokens=4443)
    try:
        assert proxy.supports_output_shrink is (backend == "bounded")
        with budget_scope(budget):
            if backend == "bounded":
                assert budgeted_chat(proxy, []).content == "done"
                assert target.requested_caps == [800]
                assert budget.total_tokens == 5350
            else:
                with pytest.raises(BudgetExceeded):
                    budgeted_chat(proxy, [])
                assert target.calls == 0
                assert target.requested_caps == []
                assert budget.total_tokens == 4443
        assert budget.snapshot()["unknown_requests"] == 0
        assert budget.snapshot()["reserved_tokens"] == 0
    finally:
        recorder.close()


def test_recoverable_failure_reflection_and_retry_are_distinct_real_calls(native, tmp_path):
    from agent.errors import RecoverableToolError
    attempts = []

    @native.tool
    def lookup() -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise RecoverableToolError("temporary outage")
        return "42"

    llm = ScriptedLLM([response(calls=[call()]), response("The outage may be temporary; retry once."),
                       response(calls=[call(call_id="retry-1")]), response("42")])
    agent = native.ReActAgent(llm, native.ToolRegistry([lookup]), max_steps=6)
    recorder = TraceRecorder(tmp_path, task_id="retry")
    result = trace_agent(agent, recorder).run("Look up the answer")
    recorder.close()
    assert result.success and attempts == [1, 1]
    assert llm.requests[1]["tools"] is None
    tool_ends = events(recorder, "span.end", "tool")
    assert [event["status"] for event in tool_ends] == ["error", "ok"]
    assert "temporary outage" in tool_ends[0]["output"]
    attempt_ends = events(recorder, "span.end", "tool_attempt")
    assert attempt_ends[0]["metadata"]["exception"]["type"] == "RecoverableToolError"
    assert len(events(recorder, "span.start", "llm")) == 4
    assert read_trace(recorder.path)["complete"]


def test_fatal_tool_failure_is_retained_without_changing_run_semantics(native, tmp_path):
    from agent.errors import FatalToolError

    @native.tool
    def lookup() -> str:
        raise FatalToolError("unrecoverable failure")

    agent = native.ReActAgent(ScriptedLLM([response(calls=[call()])]), native.ToolRegistry([lookup]))
    recorder = TraceRecorder(tmp_path, task_id="fatal-tool")
    result = trace_agent(agent, recorder).run("Lookup")
    recorder.close()
    assert not result.success and result.stop_reason == "fatal_tool_error"
    assert events(recorder, "span.end", "tool")[0]["metadata"]["exception"]["type"] == "FatalToolError"
    assert read_trace(recorder.path)["complete"]


def test_budget_stop_keeps_computed_result_and_stopping_reason(native, tmp_path):
    @native.tool
    def lookup() -> str:
        return "1056"

    agent = native.ReActAgent(ScriptedLLM([response(calls=[call()])]), native.ToolRegistry([lookup]), max_steps=1)
    recorder = TraceRecorder(tmp_path, task_id="budget")
    result = trace_agent(agent, recorder).run("Lookup")
    recorder.close()
    assert "max_steps (1)" in result.stop_reason and not result.success
    assert len(events(recorder, "span.start", "llm")) == 1
    assert events(recorder, "span.end", "tool")[0]["output"] == "1056"
    assert events(recorder, "span.end", "agent")[0]["output"]["stop_reason"] == result.stop_reason


def test_provider_exception_propagates_unchanged_and_dependencies_restore(native, tmp_path):
    from agent.retry import PermanentLLMError
    error = PermanentLLMError("invalid model")
    llm = ScriptedLLM([error])
    agent = native.ReActAgent(llm, native.ToolRegistry())
    memory = agent.short_term
    dispatcher = agent._loop._dispatcher
    registry = dispatcher.registry
    recorder = TraceRecorder(tmp_path, task_id="provider-failure")
    with pytest.raises(PermanentLLMError) as caught:
        trace_agent(agent, recorder).run("Hello")
    recorder.close(status="error")
    assert caught.value is error
    assert agent.llm is llm and agent.short_term is memory and memory._llm is llm
    assert agent._loop._dispatcher is dispatcher and dispatcher.registry is registry
    assert events(recorder, "span.end", "llm")[0]["status"] == "error"
    assert read_trace(recorder.path)["complete"]


def test_failed_model_span_keeps_usage_returned_before_generation_error(native, tmp_path):
    from agent.llm import Usage
    from agent.retry import TransientLLMError
    from agent.token_budget import TokenBudget, budget_scope

    failure = TransientLLMError("generation stopped at length limit")
    failure.usage = Usage(892, 333)

    class FailedLLM:
        max_output_tokens = 333

        def estimate_input_tokens(self, messages, tools=None):
            return 830

        def chat(self, messages, tools=None):
            raise failure

    recorder = TraceRecorder(tmp_path, task_id="known-failed-usage")
    root = TokenBudget(6000, initial_tokens=4837)
    agent = native.ReActAgent(FailedLLM(), native.ToolRegistry(), max_tokens=3000)
    with budget_scope(root):
        result = trace_agent(agent, recorder).run("Finish the response")
    recorder.close()
    assert not result.success
    assert result.stop_reason.startswith("budget:")
    assert root.total_tokens == 6062
    llm_end = events(recorder, "span.end", "llm")
    assert len(llm_end) == 1
    assert llm_end[0]["status"] == "error"
    assert llm_end[0]["usage"]["total_tokens"] == 1225
    assert llm_end[0]["metadata"]["exception"]["type"] == "TransientLLMError"
    assert read_trace(recorder.path)["complete"]


def test_short_term_summary_call_is_captured_under_context_span(native, tmp_path):
    from agent.state.memory import ShortTermMemory

    @native.tool
    def lookup() -> str:
        return "The project is ALPHA"

    llm = ScriptedLLM([response(calls=[call()]), response("The user requests project ALPHA."), response("ALPHA")])
    memory = ShortTermMemory(llm, window=2)

    class EarlierContext:
        def prepare(self, task):
            return [{"role": "assistant", "content": "An earlier inspection identified the project workspace."}]

    agent = native.ReActAgent(llm, native.ToolRegistry([lookup]), short_term=memory,
                             context_providers=[EarlierContext()])
    recorder = TraceRecorder(tmp_path, task_id="context-summary")
    result = trace_agent(agent, recorder).run("Find project ALPHA")
    recorder.close()
    assert result.success and len(llm.requests) == 3
    llm_starts = events(recorder, "span.start", "llm")
    contexts = events(recorder, "span.start", "context")
    assert "Summarize" in llm_starts[1]["input"]["messages"][0]["content"]
    assert "earlier inspection" in llm_starts[1]["input"]["messages"][1]["content"]
    assert llm_starts[1]["parent_span_id"] == contexts[1]["span_id"]
    assert "Summary of earlier conversation" in str(llm_starts[2]["input"])
    assert memory._llm is llm


def test_unknown_tool_has_dispatch_error_without_fabricated_execution(native, tmp_path):
    llm = ScriptedLLM([response(calls=[call("missing")])])
    agent = native.ReActAgent(llm, native.ToolRegistry(), max_steps=1)
    recorder = TraceRecorder(tmp_path, task_id="unknown-tool")
    trace_agent(agent, recorder).run("Lookup")
    recorder.close()
    assert len(events(recorder, "span.end", "tool")) == 1
    assert events(recorder, "span.end", "tool")[0]["status"] == "error"
    assert not events(recorder, "span.start", "tool_attempt")


def test_existing_outcome_metadata_and_close_delegation_are_preserved(tmp_path):
    original = AgentOutcome("42", True, "finished", 1, 12, metadata={"provider": "fake"})

    class ExistingAgent:
        closed = False

        def run(self, prompt):
            return original

        def close(self):
            self.closed = True

    agent = ExistingAgent()
    recorder = TraceRecorder(tmp_path, task_id="existing-outcome")
    wrapped = trace_agent(agent, recorder)
    outcome = adapt_traced(wrapped.run("Hello"))
    wrapped.close()
    assert agent.closed and outcome.metadata["provider"] == "fake"
    assert outcome.metadata["trace_coverage"] == "run_only"
    assert "trace_id" not in original.metadata
    assert read_trace(recorder.path)["manifest"]["closed"] is False
    recorder.close()


def test_existing_provider_wrapper_keeps_result_identity_and_metadata(native, tmp_path):
    agent = native.ReActAgent(ScriptedLLM([response("42")]), native.ToolRegistry())

    class ProviderWrapper:
        def __init__(self, child):
            self.agent = child

        def run(self, prompt):
            self.result = SimpleNamespace(result=self.agent.run(prompt), metadata={"api_calls": ["existing"]})
            return self.result

    provider = ProviderWrapper(agent)
    recorder = TraceRecorder(tmp_path, task_id="provider-wrapper")
    result = trace_agent(provider, recorder).run("Hello")
    recorder.close()
    assert result is provider.result and result.metadata["api_calls"] == ["existing"]
    assert result.metadata["trace_coverage"] == "react_sync_calls"
    assert len(events(recorder, "span.start", "llm")) == 1


def test_nested_agents_with_separate_recorders_do_not_cross_trace_ids(tmp_path):
    inner_recorder = TraceRecorder(tmp_path, task_id="inner")
    outer_recorder = TraceRecorder(tmp_path, task_id="outer")
    outcome = AgentOutcome("42", True, "finished", 1, 1)

    class Inner:
        def run(self, prompt):
            return outcome

    inner = trace_agent(Inner(), inner_recorder)

    class Outer:
        def run(self, prompt):
            return inner.run(prompt)

    result = trace_agent(Outer(), outer_recorder).run("Nested")
    inner_recorder.close()
    outer_recorder.close()
    assert result.metadata["trace_id"] == outer_recorder.trace_id
    assert read_trace(inner_recorder.path)["complete"]
    assert read_trace(outer_recorder.path)["complete"]


def test_frozen_third_party_result_metadata_does_not_turn_success_into_failure(tmp_path):
    @dataclass(frozen=True)
    class FrozenResult:
        answer: str = "42"
        success: bool = True
        metadata: dict = field(default_factory=lambda: {"provider": "fake"})

    expected = FrozenResult()

    class ExistingAgent:
        def run(self, prompt):
            return expected

    recorder = TraceRecorder(tmp_path, task_id="frozen-result")
    result = trace_agent(ExistingAgent(), recorder).run("Hello")
    recorder.close()
    assert result is expected and result.metadata == {"provider": "fake"}
    assert read_trace(recorder.path)["complete"]


def test_five_smokes_run_real_mechanisms_without_provider_calls(native, tmp_path):
    from benchmarks.agent_design.trace_smoke import run_smoke
    directory = tmp_path / "smoke"
    harness = Path(native.__file__).resolve().parent.parent
    report = run_smoke(directory, harness)
    assert report["passed"] == report["case_count"] == 5
    assert report["model_api_calls"] == 0 and report["smoke_only"]
    assert all(record["smoke_only"] and record["scores"] == {} for record in report["records"])
    context = next(row for row in report["cases"] if row["id"] == "smoke.context_summary")
    context_trace = read_trace(context["trace_path"])
    calls = [event for event in context_trace["events"] if event["event"] == "span.start" and event.get("kind") == "llm"]
    contexts = [event for event in context_trace["events"] if event["event"] == "span.start" and event.get("kind") == "context"]
    assert len(calls) == 3 and "Summarize" in calls[1]["input"]["messages"][0]["content"]
    assert calls[1]["parent_span_id"] == contexts[1]["span_id"]
    assert context["evidence"]["summary_and_final_prompts_recorded"]
    a2a = next(row for row in report["cases"] if row["id"] == "smoke.a2a_failure_isolation")
    loaded = read_trace(a2a["trace_path"])
    workers = [event for event in loaded["events"] if event["event"] == "span.start"
               and event.get("name") in {"broken.run", "healthy.run"}]
    assert len(workers) == 2
    assert all(event["metadata"]["runtime_context"]["task_id"] for event in workers)
    original_report = (directory / "report.json").read_bytes()
    with pytest.raises(FileExistsError, match="choose a new --out"):
        run_smoke(directory, harness)
    assert (directory / "report.json").read_bytes() == original_report
    assert len(list((directory / "traces").glob("*.jsonl"))) == 5
    assert all(record["condition"] == read_trace(record["trace"]["path"])["manifest"]["condition"]
               for record in report["records"])
