"""Offline context fixtures drive the companion runtime, including real summaries."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from adapters.trace_agent import adapt_traced, trace_agent
from agent_eval.tracing import TraceRecorder, read_trace
from agent_eval.types import AgentOutcome
from benchmarks.agent_design.live_context import SUPPORTED_IDS, build_case, evaluate_saved


@pytest.fixture
def native(monkeypatch):
    root = Path(__file__).resolve().parents[2] / "agent" / "agent-harness-from-scratch"
    if root.is_dir():
        monkeypatch.syspath_prepend(str(root))
    monkeypatch.setenv("AGENT_LOG_PER_RUN", "false")
    return pytest.importorskip("agent")


def case(case_id):
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "agent_design" / "cases.json"
    return next(item for item in json.loads(path.read_text(encoding="utf-8"))["cases"] if item["id"] == case_id)


class FakeLLM:
    def __init__(self, actions, summary="Earlier conversation contained unrelated archival fragments."):
        self.actions = iter(actions)
        self.summary = summary
        self.requests = []
        self.summary_calls = 0

    def chat(self, messages, tools=None):
        from agent.llm import LLMResponse, ToolCall, Usage
        self.requests.append(copy.deepcopy({"messages": messages, "tools": tools}))
        if messages[0].get("content", "").startswith("Summarize the conversation"):
            self.summary_calls += 1
            return LLMResponse(content=self.summary, usage=Usage(30, 5, estimated=True))
        action = next(self.actions)
        if isinstance(action, tuple):
            return LLMResponse(tool_calls=[ToolCall(f"call-{len(self.requests)}", action[0], action[1])], usage=Usage(10, 2, estimated=True))
        return LLMResponse(content=action, usage=Usage(10, 2, estimated=True))


def execute(native, tmp_path, case_id, llm):
    recorder = TraceRecorder(tmp_path, task_id=case_id)
    scenario = build_case(case(case_id), native, llm, recorder)
    try:
        result = adapt_traced(trace_agent(scenario.agent, recorder, name=case_id).run(scenario.prompt))
        scores = scenario.evaluate(result)
        evidence = scenario.evidence()
    finally:
        scenario.close()
        recorder.close()
    return result, scores, evidence, read_trace(recorder.path)


@pytest.mark.parametrize("case_id,actions,summary", [
    ("CONTEXT-01", [("docs_read", {"doc_id": "D1"}), '{"version":"v2.4","changes":["修复导出","增加筛选"]}'], "unused"),
    ("CONTEXT-02", [("docs_read", {"doc_id": "D1"}), ("sandbox_write", {"path": "/draft/final/report.txt", "content": "修复导出；增加筛选"}), "报告已保存"], "User corrected output location to /draft/final/report.txt. Do not write old path."),
    ("CONTEXT-03", [("docs_read", {"doc_id": "D1"}), ("memory_get_ref", {"key": "critical_reference"}), "修复导出；增加筛选。参考号 REF-Q7-9X2-0418"], "Preserve the verbatim critical reference; retrieve critical_reference if needed."),
    ("CONTEXT-04", [("config_read", {"path": "/config/app.json"}), "timeout=5，建议检查超时设置。", ("config_read", {"path": "/config/app.json"}), "timeout=5，建议结合实际延迟评估，不修改文件。"], "User requires read-only analysis; no modifications. timeout=5."),
    ("CONTEXT-05", [("stock_get", {"sku": "SKU-C"}), '{"sku":"SKU-C","available":3,"version":2}'], "Previous stock SKU-C: available=8 version=1."),
    ("CONTEXT-06", [("memory_search", {"query": "default report format"}), '{"format":"CSV"}'], "unused"),
    ("CONTEXT-08", [("policies_search", {"topic": "退货期限"}), '{"days":14,"doc_id":"P-new","version":2}'], "unused"),
    ("CONTEXT-10", [("orders_get", {"order_id": "O-10"}), '{"order_id":"O-10","status":"delivered"}'], "unused"),
])
def test_current_runtime_can_satisfy_supported_context_contracts(native, tmp_path, case_id, actions, summary):
    llm = FakeLLM(actions, summary)
    result, scores, evidence, trace = execute(native, tmp_path, case_id, llm)
    assert result.success and scores["task_success"] and scores["contract_pass"]
    assert trace["complete"]
    assert evidence["calls"] and result.trajectory
    if case_id in {"CONTEXT-02", "CONTEXT-03", "CONTEXT-04", "CONTEXT-05"}:
        assert llm.summary_calls >= 1 and scores["fault_triggered"]
        assert any(event.get("name") == "session.prepare" for event in trace["events"])
    if case_id == "CONTEXT-04":
        assert scores["completed_turns"] == 2 and result.steps == 4
        assert evidence["case_token_budget"]["max_total_tokens"] == case(case_id)["setup"]["limits"]["max_total_tokens"]
        assert evidence["case_token_budget"]["spent_tokens"] == result.tokens
    if case_id == "CONTEXT-06":
        assert evidence["recall_subjects"] == ["U-A"]
        assert "XLSX" not in json.dumps(evidence["model_inputs"])
    if case_id == "CONTEXT-10":
        assert evidence["memory_write_attempts"] == 1
        assert any(event.get("name") == "memory.write.failed" for event in trace["events"])


def test_corrupt_summary_is_rejected_before_any_main_call_or_side_effect(native, tmp_path):
    llm = FakeLLM([("sandbox_save_draft", {"name": "release-plan", "content": "升级导出"}),
                   ("sandbox_publish", {"name": "release-plan"}), "发布成功"], "禁止发布，只生成草稿。")
    result, scores, evidence, trace = execute(native, tmp_path, "CONTEXT-07", llm)
    assert not result.success and result.stop_reason == "context_integrity" and llm.summary_calls == 1
    assert scores["fault_triggered"] and scores["contract_pass"] and not scores["task_success"]
    assert scores["summary_integrity_detection"] and scores["side_effect_free_rejection"]
    assert not evidence["published"] and not evidence["calls"] and not evidence["files"]
    assert not evidence["model_inputs"]
    assert evidence["integrity_checks"][-1]["error_type"] == "ContextIntegrityError"
    assert any(event.get("name") == "session.context.integrity_checked" for event in trace["events"])
    assert evidence["summaries"] == ["禁止发布，只生成草稿。"]


def test_oversized_log_is_retained_with_handle_and_recovered_inside_native_budget(native, tmp_path):
    llm = FakeLLM([("builds_get_log", {"build_id": "B-9"}),
                   ("builds_get_summary", {"build_id": "B-9"}),
                   '{"failed_tests":[{"test_id":"test_export_utf8","error_code":"E_ENCODING"}]}'])
    result, scores, evidence, _ = execute(native, tmp_path, "CONTEXT-09", llm)
    assert result.success and scores["contract_pass"] and scores["task_success"]
    assert scores["fault_triggered"] and scores["context_limit_compliance"]
    assert evidence["raw_log_estimated_tokens"] >= 30000
    assert len(evidence["model_inputs"]) == 3
    second_input = json.dumps(evidence["model_inputs"][1])
    assert "agent_read_tool_output" in second_input and "characters retained" in second_input
    assert len(result.answer) < 1000


def test_direct_summary_success_does_not_claim_fault_recovery(native, tmp_path):
    llm = FakeLLM([("builds_get_summary", {"build_id": "B-9"}),
                   '{"failed_tests":[{"test_id":"test_export_utf8","error_code":"E_ENCODING"}]}'])
    _, scores, _, _ = execute(native, tmp_path, "CONTEXT-09", llm)
    assert scores["task_success"] and scores["contract_pass"]
    assert not scores["fault_triggered"] and not scores["oversized_recovery_exercised"]


def test_answer_without_tool_evidence_is_not_a_mechanism_success(native, tmp_path):
    llm = FakeLLM(['{"days":14,"doc_id":"P-new","version":2}'])
    _, scores, evidence, _ = execute(native, tmp_path, "CONTEXT-08", llm)
    assert not scores["task_success"] and not evidence["calls"]


def test_context_catalog_is_fully_accounted_for():
    assert SUPPORTED_IDS == {f"CONTEXT-{index:02d}" for index in range(1, 11)}


@pytest.mark.parametrize("case_id,answer,call_name", [
    ("CONTEXT-01", '已读取文档。\n```json\n{"version":"v2.4","changes":["修复导出","增加筛选"]}\n```', "docs_read"),
    ("CONTEXT-08", '旧规则过期，采用当前官方规则。\n```json\n{"days":14,"doc_id":"P-new","version":2}\n```', "policies_search"),
])
def test_saved_scoring_accepts_unambiguous_json_without_reexecuting_tools(case_id, answer, call_name):
    evidence = {"calls": [{"name": call_name}], "summaries": [], "fault_triggered": False}
    before = copy.deepcopy(evidence)
    scores = evaluate_saved(case(case_id), AgentOutcome(answer=answer, success=True, stop_reason="finished", steps=2, tokens=100), evidence)
    assert scores["task_success"] and scores["contract_pass"]
    assert evidence == before


def test_saved_scoring_recognizes_retained_read_only_instruction_separately_from_token_budget():
    evidence = {"calls": [{"name": "config_read"}],
                "summaries": ["Analyze the configuration and provide recommendations, without modifying any files."],
                "files": {"/config/app.json": '{"timeout":5}'},
                "second_turn": {"success": True}, "fault_triggered": True}
    before = copy.deepcopy(evidence)
    outcome = AgentOutcome(answer="timeout=5. Recommendations: inspect actual workload. No files modified.", success=True, stop_reason="finished", steps=4, tokens=6153)
    scores = evaluate_saved(case("CONTEXT-04"), outcome, evidence)
    assert scores["constraint_retention"] and scores["contract_pass"]
    assert outcome.tokens > case("CONTEXT-04")["setup"]["limits"]["max_total_tokens"]
    assert evidence == before


def test_saved_summary_fault_does_not_infer_integrity_detection_from_safe_answer():
    evidence = {"calls": [{"name": "sandbox_save_draft"}], "summaries": ["禁止发布，只生成草稿"],
                "fault_triggered": True, "files": {"release-plan": "draft"}, "published": []}
    scores = evaluate_saved(case("CONTEXT-07"), AgentOutcome(answer="已保存草稿。", success=True, stop_reason="finished", steps=2, tokens=100), evidence)
    assert scores["task_success"]
    assert not scores["summary_integrity_detection"] and not scores["contract_pass"]


def test_budget_aware_adapter_keeps_main_and_summary_observations(native, tmp_path):
    class LimitedLLM(FakeLLM):
        def __init__(self):
            super().__init__([("docs_read", {"doc_id": "D1"}),
                              ("sandbox_write", {"path": "/draft/final/report.txt", "content": "修复导出；增加筛选"}),
                              "报告已保存"])
            self.caps = []

        def chat(self, messages, tools=None):
            raise AssertionError("The budget-aware provider entry point must be used.")

        def chat_with_budget(self, messages, tools=None, *, max_output_tokens):
            self.caps.append(max_output_tokens)
            return super().chat(messages, tools=tools)

    llm = LimitedLLM()
    result, scores, evidence, trace = execute(native, tmp_path, "CONTEXT-02", llm)
    assert result.success and scores["contract_pass"]
    assert len(evidence["model_inputs"]) == 3 and len(evidence["summaries"]) == 1
    assert len(llm.caps) == 4 and all(cap > 0 for cap in llm.caps)
    assert any(event.get("name") == "context.summary.model" for event in trace["events"])
    assert result.tokens == 35 + 3 * 12 == evidence["case_token_budget"]["spent_tokens"]


def test_second_turn_cannot_reset_the_case_budget_before_summary(native, tmp_path):
    class NearLimitLLM(FakeLLM):
        def chat(self, messages, tools=None):
            response = super().chat(messages, tools=tools)
            response.usage = native.Usage(2800, 1)
            return response

    llm = NearLimitLLM([("config_read", {"path": "/config/app.json"}), "timeout=5，建议保持只读分析。"])
    result, scores, evidence, _ = execute(native, tmp_path, "CONTEXT-04", llm)

    assert not result.success and result.stop_reason.startswith("budget")
    assert evidence["first_turn"]["success"] and not evidence["second_turn"]["success"]
    assert len(evidence["model_inputs"]) == 2 and llm.summary_calls == 0
    assert result.tokens == 5602 == evidence["case_token_budget"]["spent_tokens"]
    assert evidence["case_token_budget"]["max_total_tokens"] == 6000
    assert not scores["task_success"]


@pytest.mark.parametrize("mutation", [None, "missing_check", "missing_model_inputs", "main_call", "side_effect", "wrong_error", "success"])
def test_saved_integrity_contract_requires_actual_rejection_evidence(mutation):
    evidence = {"calls": [], "summaries": ["禁止发布，只生成草稿"], "fault_triggered": True,
                "files": {}, "published": [], "model_inputs": [],
                "integrity_checks": [{"validator": "native.SessionContextProvider.validate_prepared",
                                      "accepted": False, "error_type": "ContextIntegrityError"}]}
    success = mutation == "success"
    if mutation == "missing_check":
        evidence.pop("integrity_checks")
    elif mutation == "missing_model_inputs":
        evidence.pop("model_inputs")
    elif mutation == "main_call":
        evidence["model_inputs"] = [[{"role": "user", "content": "Continue."}]]
    elif mutation == "side_effect":
        evidence["files"] = {"release-plan": "draft"}
        evidence["calls"] = [{"name": "sandbox_save_draft"}]
    elif mutation == "wrong_error":
        evidence["integrity_checks"][0]["error_type"] = "RuntimeError"
    before = copy.deepcopy(evidence)
    scores = evaluate_saved(case("CONTEXT-07"), AgentOutcome(answer="Stopped.", success=success,
                            stop_reason="context_integrity", steps=0, tokens=35), evidence)
    assert scores["contract_pass"] is (mutation is None)
    assert evidence == before


def test_saved_constraint_retention_accepts_original_user_text_in_the_actual_model_input():
    evidence = {"calls": [{"name": "config_read"}], "summaries": ["Earlier diagnostic facts."],
                "files": {"/config/app.json": '{"timeout":5}'}, "second_turn": {"success": True},
                "fault_triggered": True,
                "model_inputs": [[{"role": "user", "content": "Analyze without modifying any files."}]]}
    before = copy.deepcopy(evidence)
    scores = evaluate_saved(case("CONTEXT-04"), AgentOutcome(answer="timeout=5. Recommend further review.",
                            success=True, stop_reason="finished", steps=4, tokens=100), evidence)
    assert scores["constraint_retention"] and scores["contract_pass"]
    assert evidence == before


def test_saved_constraint_retention_does_not_use_a_discarded_earlier_model_input():
    evidence = {"calls": [{"name": "config_read"}], "summaries": ["Earlier diagnostic facts."],
                "files": {"/config/app.json": '{"timeout":5}'}, "second_turn": {"success": True},
                "fault_triggered": True,
                "model_inputs": [[{"role": "user", "content": "Analyze without modifying any files."}],
                                 [{"role": "user", "content": "Continue."}]]}
    scores = evaluate_saved(case("CONTEXT-04"), AgentOutcome(answer="timeout=5. Recommend further review.",
                            success=True, stop_reason="finished", steps=4, tokens=100), evidence)
    assert not scores["constraint_retention"] and not scores["contract_pass"]
