"""Run the native budget paths offline and keep failures as benchmark evidence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time

import pytest

from adapters.trace_agent import adapt_traced, trace_agent
from agent_eval.tracing import TraceRecorder, read_trace
from agent_eval.types import AgentOutcome
from benchmarks.agent_design.live_budget import build_case


@pytest.fixture
def native(monkeypatch):
    root = Path(__file__).resolve().parents[2] / "agent" / "agent-harness-from-scratch"
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.setenv("AGENT_LOG_PER_RUN", "false")
    return pytest.importorskip("agent")


def catalog():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "agent_design" / "cases.json"
    return {case["id"]: case for case in json.loads(path.read_text(encoding="utf-8"))["cases"]}


class NoProvider:
    def chat(self, *args, **kwargs):
        raise AssertionError("Deterministic budget cases must not call the supplied live provider")


class ScriptedLLM:
    def __init__(self, script):
        self.script = iter(script)

    def chat(self, messages, tools=None):
        from agent.llm import LLMResponse, ToolCall, Usage
        item = next(self.script)
        if isinstance(item, list):
            return LLMResponse(tool_calls=[ToolCall(str(index), name, args) for index, (name, args) in enumerate(item)], usage=Usage(20, 5))
        return LLMResponse(content=item, usage=Usage(20, 5))


@pytest.mark.parametrize("identifier,expected", [
    ("BUDGET-03", True), ("BUDGET-04", True), ("BUDGET-05", True),
    ("BUDGET-06", True), ("BUDGET-07", True), ("BUDGET-08", True),
    ("BUDGET-09", True), ("BUDGET-10", True),
])
def test_deterministic_cases_execute_real_runtime_and_report_observed_contract(native, tmp_path, identifier, expected):
    recorder = TraceRecorder(tmp_path, task_id=identifier)
    scenario = build_case(catalog()[identifier], native, NoProvider(), recorder)
    began = time.monotonic()
    try:
        assert scenario.execution_mode == "deterministic_runtime"
        outcome = adapt_traced(trace_agent(scenario.agent, recorder).run(scenario.prompt, **scenario.run_kwargs))
        elapsed = time.monotonic() - began
        scores = scenario.evaluate(outcome)
        evidence = scenario.evidence()
        assert scores["fault_triggered"] is True
        assert scores["contract_pass"] is expected, (scores, outcome.answer, outcome.stop_reason)
        if identifier == "BUDGET-03":
            assert "loop_detected" in outcome.stop_reason and scores["business_calls"] <= 3
        elif identifier == "BUDGET-04":
            assert scores["business_calls"] == 4 and scores["loop_guard_triggered"]
        elif identifier == "BUDGET-05":
            assert outcome.tokens == 750 and scores["model_calls"] == 1
            assert outcome.stop_reason.startswith("budget:")
            assert sum(e["event"] == "scripted_model" for e in evidence["events"]) == 1
            assert sum(e["total_tokens"] for e in evidence["events"] if e["event"] == "controlled_usage") == 750
        elif identifier == "BUDGET-06":
            assert elapsed <= 3.2 and 0 < scores["tool_elapsed_seconds"] <= 3.2
            assert outcome.stop_reason == "tool_timeout" and scores["process_reaped"]
            assert any(e["event"] == "tool.process.started" for e in evidence["events"])
        elif identifier == "BUDGET-07":
            assert outcome.stop_reason == "cancelled" and scores["committed_count"] == 1
        elif identifier == "BUDGET-08":
            assert sorted(evidence["state"]["worker_tokens"]) in ([0, 1200], [0, 1500])
            assert outcome.tokens in (1901, 2201)
            assert scores["budget_rejected_worker"] and scores["task_success"] is False
        elif identifier == "BUDGET-09":
            assert scores["attempt_count"] == 2
            attempts = [e for e in evidence["events"] if e["event"] == "attempt.started"]
            assert attempts[0]["timeout"] == 2 and 0 < attempts[1]["timeout"] < 1.1
            assert "deadline" in outcome.answer and "TEMPORARY_FAILURE" in outcome.answer
        elif identifier == "BUDGET-10":
            assert outcome.stop_reason == "fatal_tool_error" and scores["unauthorized_effect_count"] == 0
    finally:
        scenario.close()
        recorder.close()
    trace = read_trace(recorder.path)
    assert trace["complete"], trace["diagnostics"]
    llms = [span for span in trace["spans"] if span["kind"] == "llm"]
    assert llms
    if identifier == "BUDGET-08":
        assert sum(span.get("usage", {}).get("total_tokens", 0) for span in llms) == outcome.tokens
        workers = [span for span in trace["spans"] if span["kind"] == "worker"]
        assert len(workers) == 2 and all(span["parent_id"] for span in workers)


@pytest.mark.parametrize("answer,expected", [
    ('{"T1":"open","T2":"closed"}', True),
    ('{"T1":"closed","T2":"open"}', False),
])
def test_normal_ticket_answer_must_bind_status_to_correct_ticket(native, tmp_path, answer, expected):
    recorder = TraceRecorder(tmp_path, task_id="BUDGET-01")
    llm = ScriptedLLM([[('tickets_read', {'ticket_id': 'T1'}), ('tickets_read', {'ticket_id': 'T2'})], answer])
    scenario = build_case(catalog()["BUDGET-01"], native, llm, recorder)
    try:
        outcome = adapt_traced(trace_agent(scenario.agent, recorder).run(scenario.prompt))
        assert scenario.evaluate(outcome)["contract_pass"] is expected
    finally:
        scenario.close()
        recorder.close()


@pytest.mark.parametrize("identifier", ["BUDGET-03", "BUDGET-04"])
def test_no_business_call_means_no_fault_exercised(native, tmp_path, identifier):
    recorder = TraceRecorder(tmp_path, task_id=identifier)
    case = deepcopy(catalog()[identifier])
    case["setup"]["limits"]["max_total_tokens"] = 1
    scenario = build_case(case, native, NoProvider(), recorder)
    try:
        outcome = adapt_traced(trace_agent(scenario.agent, recorder).run(scenario.prompt))
        assert not scenario.evidence()["state"]["fault_triggered"]
        assert scenario.evaluate(outcome)["fault_triggered"] is False
        assert scenario.evaluate(outcome)["contract_pass"] is False
    finally:
        scenario.close()
        recorder.close()


def test_shared_budget_oracle_does_not_trust_an_underreported_root_counter(native, tmp_path):
    recorder = TraceRecorder(tmp_path, task_id="BUDGET-08")
    scenario = build_case(catalog()["BUDGET-08"], native, NoProvider(), recorder)
    try:
        outcome = adapt_traced(trace_agent(scenario.agent, recorder).run(scenario.prompt))
        # Only the native aggregate is corrupted; the independent sum of actual
        # model responses still includes the admitted Worker's real usage.
        scenario.evidence()["state"]["root_tokens"] = 700
        underreported = replace(outcome, tokens=700)
        assert scenario.evaluate(underreported)["contract_pass"] is False
    finally:
        scenario.close()
        recorder.close()


@pytest.mark.parametrize("missing", ["run_elapsed", "process_reaped", "process_outcome"])
def test_deadline_oracle_does_not_assume_missing_cleanup_or_timing_evidence(native, tmp_path, missing):
    recorder = TraceRecorder(tmp_path, task_id="BUDGET-06")
    scenario = build_case(catalog()["BUDGET-06"], native, NoProvider(), recorder)
    try:
        state = scenario.evidence()["state"]
        state.update(fault_triggered=True, run_elapsed=2.9, tool_elapsed=2.9,
                     process_reaped=True, process_outcome="timed_out")
        outcome = AgentOutcome(answer="timeout", success=False, stop_reason="tool_timeout", steps=1, tokens=12)
        assert scenario.evaluate(outcome)["contract_pass"] is True
        del state[missing]
        assert scenario.evaluate(outcome)["contract_pass"] is False
    finally:
        scenario.close()
        recorder.close()


def test_deadline_oracle_requires_child_readiness_and_actual_timeout(native, tmp_path):
    recorder = TraceRecorder(tmp_path, task_id="BUDGET-06")
    scenario = build_case(catalog()["BUDGET-06"], native, NoProvider(), recorder)
    try:
        state = scenario.evidence()["state"]
        state.update(run_elapsed=2.9, tool_elapsed=2.9, process_reaped=True, process_outcome="timed_out")
        outcome = AgentOutcome(answer="timeout", success=False, stop_reason="tool_timeout", steps=1, tokens=12)
        assert scenario.evaluate(outcome)["fault_triggered"] is False
        assert scenario.evaluate(outcome)["contract_pass"] is False
        state["fault_triggered"] = True
        assert scenario.evaluate(replace(outcome, stop_reason="fatal_tool_error"))["contract_pass"] is False
    finally:
        scenario.close()
        recorder.close()
