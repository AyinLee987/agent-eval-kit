"""Independent state oracles reject plausible answers with missing real effects."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import threading

import pytest

from adapters.trace_agent import adapt_traced, trace_agent
from agent_eval.tracing import TraceRecorder, read_trace
from agent_eval.types import AgentOutcome
from benchmarks.agent_design.live_state import SUPPORTED_IDS, UNSUPPORTED, build_case, evaluate_saved


@pytest.fixture
def native(monkeypatch):
    root = Path(__file__).resolve().parents[2] / "agent" / "agent-harness-from-scratch"
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.setenv("AGENT_LOG_PER_RUN", "false")
    return pytest.importorskip("agent")


def cases():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "agent_design" / "cases.json"
    return {case["id"]: case for case in json.loads(path.read_text(encoding="utf-8"))["cases"]}


SCRIPTS = {
    "STATE-01": [
        ("labels_put", {"customer": "C-1", "label": "VIP", "operation_key": "label-C1-VIP"}),
        ("labels_get", {"customer": "C-1"}),
        {"status": "saved", "customer": "C-1", "labels": ["VIP"]},
    ],
    "STATE-02": [
        ("tickets_create", {"title": "月报", "operation_key": "CREATE-42"}),
        ("tickets_find_by_operation", {"operation_key": "CREATE-42"}),
        {"status": "created", "ticket_id": "T-42"},
    ],
    "STATE-04": [
        ("stock_read", {"sku": "SKU-A"}),
        ("stock_reserve", {"sku": "SKU-A", "quantity": 1, "expected_version": 8, "operation_key": "RESERVE-O4"}),
        ("stock_read", {"sku": "SKU-A"}),
        {"status": "blocked", "reserved": False, "reason": "insufficient_stock"},
    ],
    "STATE-05": [
        ("jobs_start", {"job_id": "J-5"}),
        ("jobs_cancel", {"job_id": "J-5"}),
        {"job_id": "J-5", "state": "cancelled"},
    ],
    "STATE-06": [
        ("seats_reserve", {"seat": "S1", "operation_key": "P-6"}),
        ("vouchers_issue", {"voucher": "V1", "operation_key": "P-6"}),
        ("seats_release", {"seat": "S1", "operation_key": "P-6"}),
        {"order_status": "failed", "seat": "free", "voucher": "not_issued"},
    ],
    "STATE-07": [
        ("reminders_create", {"title": "周报", "operation_key": "REMIND-7"}),
        {"status": "conflict", "created": False},
    ],
}


class ScriptedLLM:
    def __init__(self, script):
        self.script = iter(deepcopy(script))
        self.requests = []

    def chat(self, messages, tools=None):
        from agent.llm import LLMResponse, ToolCall, Usage
        self.requests.append(deepcopy(messages))
        if tools is None:
            return LLMResponse(content="Reconsider the error and verify the actual state before continuing.", usage=Usage(12, 3))
        item = next(self.script)
        if isinstance(item, tuple):
            return LLMResponse(tool_calls=[ToolCall("call-" + str(len(self.requests)), item[0], item[1])], usage=Usage(12, 3))
        return LLMResponse(content=json.dumps(item, ensure_ascii=False), usage=Usage(12, 3))


class SessionLLM:
    def __init__(self):
        self.cursors = {"A": 0, "B": 0}
        self.requests = []
        self.lock = threading.Lock()

    def chat(self, messages, tools=None):
        from agent.llm import LLMResponse, ToolCall, Usage
        task = next(message["content"] for message in messages if message["role"] == "user")
        session = "A" if "当前会话为 A" in task else "B"
        with self.lock:
            self.requests.append(deepcopy(messages))
            cursor = self.cursors[session]
            self.cursors[session] += 1
        args = {"session_id": session, "draft_id": "D"}
        if cursor == 0:
            args["title"] = session + "-周报"
            return LLMResponse(tool_calls=[ToolCall(session + "-put", "drafts_put", args)], usage=Usage(12, 3))
        if cursor == 1:
            return LLMResponse(tool_calls=[ToolCall(session + "-get", "drafts_get", args)], usage=Usage(12, 3))
        return LLMResponse(content=json.dumps({"session_id": session, "title": session + "-周报"}, ensure_ascii=False), usage=Usage(12, 3))


@pytest.mark.parametrize("identifier", sorted(SUPPORTED_IDS))
def test_actual_agent_effects_match_oracle_and_are_fully_traced(native, tmp_path, identifier):
    llm = SessionLLM() if identifier == "STATE-08" else ScriptedLLM(SCRIPTS[identifier])
    recorder = TraceRecorder(tmp_path, task_id=identifier)
    scenario = build_case(cases()[identifier], native, llm, recorder)
    try:
        outcome = adapt_traced(trace_agent(scenario.agent, recorder).run(scenario.prompt, **scenario.run_kwargs))
        scores = scenario.evaluate(outcome)
        assert scores["contract_pass"], (scores, outcome.answer, scenario.evidence())
        assert scores["invariant_pass"]
        saved_evidence = scenario.evidence()
        unchanged = deepcopy(saved_evidence)
        assert evaluate_saved(cases()[identifier], outcome, saved_evidence) == scores
        assert saved_evidence == unchanged
        assert scores["fault_triggered"] is (identifier not in {"STATE-01", "STATE-08"})
        expected_task_success = identifier not in {"STATE-04", "STATE-06", "STATE-07"}
        assert scores["task_success"] is expected_task_success
        assert scenario.evidence()["events"]
        if identifier != "STATE-08":
            bad_answer = replace(outcome, answer='{"status":"made_up"}')
            assert not scenario.evaluate(bad_answer)["contract_pass"]
    finally:
        scenario.close()
        recorder.close()
    trace = read_trace(recorder.path)
    assert trace["complete"]
    assert any(span["kind"] == "llm" for span in trace["spans"])
    assert any(span["kind"] == "tool" for span in trace["spans"])
    if identifier == "STATE-08":
        sessions = [span for span in trace["spans"] if span["kind"] == "session"]
        assert len(sessions) == 2 and all(span["parent_id"] for span in sessions)


@pytest.mark.parametrize("identifier", sorted(SUPPORTED_IDS))
def test_answer_only_cannot_pass_without_state_effects_or_fault_exercise(native, tmp_path, identifier):
    recorder = TraceRecorder(tmp_path, task_id=identifier)
    scenario = build_case(cases()[identifier], native, ScriptedLLM([]), recorder)
    try:
        final = SCRIPTS.get(identifier, [{}])[-1]
        outcome = AgentOutcome(json.dumps(final, ensure_ascii=False), True, "finished", 1, 10)
        scores = scenario.evaluate(outcome)
        assert scores["contract_pass"] is False
        assert scores["task_success"] is False
        assert scores["fault_triggered"] is False
    finally:
        scenario.close()
        recorder.close()


def test_response_lost_can_be_reconciled_by_same_key_replay(native, tmp_path):
    script = deepcopy(SCRIPTS["STATE-02"])
    script[1] = deepcopy(script[0])
    recorder = TraceRecorder(tmp_path, task_id="same-key-retry")
    scenario = build_case(cases()["STATE-02"], native, ScriptedLLM(script), recorder)
    try:
        outcome = adapt_traced(trace_agent(scenario.agent, recorder).run(scenario.prompt))
        assert scenario.evaluate(outcome)["contract_pass"]
        assert scenario.evaluate(outcome)["physical_commits"] == 1
    finally:
        scenario.close()
        recorder.close()


def test_new_key_bypass_and_false_success_fail_conflict_contract(native, tmp_path):
    script = [SCRIPTS["STATE-07"][0], ("reminders_create", {"title": "周报", "operation_key": "NEW-KEY"}),
              {"status": "created", "created": True}]
    recorder = TraceRecorder(tmp_path, task_id="key-bypass")
    scenario = build_case(cases()["STATE-07"], native, ScriptedLLM(script), recorder)
    try:
        outcome = adapt_traced(trace_agent(scenario.agent, recorder).run(scenario.prompt))
        scores = scenario.evaluate(outcome)
        assert not scores["contract_pass"] and not scores["invariant_pass"]
        assert scores["new_key_bypass_count"] == 1
    finally:
        scenario.close()
        recorder.close()


def test_missing_compensation_fails_even_when_final_claims_released(native, tmp_path):
    script = deepcopy(SCRIPTS["STATE-06"])
    del script[2]
    recorder = TraceRecorder(tmp_path, task_id="missing-compensation")
    scenario = build_case(cases()["STATE-06"], native, ScriptedLLM(script), recorder)
    try:
        outcome = adapt_traced(trace_agent(scenario.agent, recorder).run(scenario.prompt))
        scores = scenario.evaluate(outcome)
        assert not scores["contract_pass"] and not scores["invariant_pass"]
        assert not scores["compensation_success"]
        assert scenario.evidence()["state"]["seat"]["S1"] == "reserved"
    finally:
        scenario.close()
        recorder.close()


def test_all_ten_catalog_cases_have_explicit_support_or_missing_runtime_reason(native, tmp_path):
    identifiers = {identifier for identifier in cases() if identifier.startswith("STATE-")}
    assert SUPPORTED_IDS | set(UNSUPPORTED) == identifiers
    assert not SUPPORTED_IDS & set(UNSUPPORTED)
    recorder = TraceRecorder(tmp_path, task_id="unsupported")
    try:
        for identifier, reason in UNSUPPORTED.items():
            with pytest.raises(ValueError, match="."):
                build_case(cases()[identifier], native, ScriptedLLM([]), recorder)
            assert len(reason) > 60
    finally:
        recorder.close()


def saved_run(native, tmp_path, identifier):
    """Return only serialized observations after closing every live dependency."""
    recorder = TraceRecorder(tmp_path, task_id=identifier)
    scenario = build_case(cases()[identifier], native, ScriptedLLM(SCRIPTS[identifier]), recorder)
    try:
        outcome = adapt_traced(trace_agent(scenario.agent, recorder).run(scenario.prompt))
        return outcome, json.loads(json.dumps(scenario.evidence(), ensure_ascii=False))
    finally:
        scenario.close()
        recorder.close()


@pytest.mark.parametrize("identifier", ["STATE-02", "STATE-04", "STATE-05", "STATE-06"])
def test_saved_oracle_scores_fenced_or_prefaced_answer_without_rerunning_agent(native, tmp_path, identifier):
    outcome, evidence = saved_run(native, tmp_path, identifier)
    baseline = deepcopy(evidence)
    for answer in ("The observed result follows.\n```json\n" + outcome.answer + "\n```",
                   "The observed result follows.\n" + outcome.answer):
        scores = evaluate_saved(cases()[identifier], replace(outcome, answer=answer), evidence)
        assert scores["contract_pass"]
        assert scores["format_compliant"] is False
        assert scores["task_success"] is (identifier in {"STATE-02", "STATE-05"})
    assert evidence == baseline


@pytest.mark.parametrize("suffix", ['\n{"status":"failed"}', '\n```json\n{"status":"failed"}\n```'])
def test_saved_oracle_rejects_ambiguous_answer_even_when_database_is_correct(native, tmp_path, suffix):
    outcome, evidence = saved_run(native, tmp_path, "STATE-02")
    answer = "```json\n" + outcome.answer + "\n```" + suffix
    assert not evaluate_saved(cases()["STATE-02"], replace(outcome, answer=answer), evidence)["contract_pass"]


def test_saved_oracle_does_not_repair_truncated_json(native, tmp_path):
    outcome, evidence = saved_run(native, tmp_path, "STATE-02")
    for answer in (outcome.answer[:-1], "```json\n" + outcome.answer[:-1] + "\n```"):
        assert not evaluate_saved(cases()["STATE-02"], replace(outcome, answer=answer), evidence)["contract_pass"]


def test_fenced_content_cannot_substitute_for_reconciliation_or_actual_commit(native, tmp_path):
    outcome, evidence = saved_run(native, tmp_path, "STATE-02")
    outcome = replace(outcome, answer="```json\n" + outcome.answer + "\n```")
    missing_lookup = deepcopy(evidence)
    missing_lookup["events"] = [e for e in missing_lookup["events"] if e["name"] != "lookup"]
    assert not evaluate_saved(cases()["STATE-02"], outcome, missing_lookup)["contract_pass"]
    duplicate = deepcopy(evidence)
    duplicate["state"]["tickets"].append(deepcopy(duplicate["state"]["tickets"][0]))
    assert not evaluate_saved(cases()["STATE-02"], outcome, duplicate)["contract_pass"]
    untriggered = deepcopy(evidence)
    untriggered["fault_triggered"] = False
    assert not evaluate_saved(cases()["STATE-02"], outcome, untriggered)["contract_pass"]


def test_compensation_aliases_require_real_failed_order_and_ordered_rollback(native, tmp_path):
    outcome, evidence = saved_run(native, tmp_path, "STATE-06")
    answer = 'The order failed; the seat was released.\n```json\n{"order_status":"failed","seat":"released","voucher":"none"}\n```'
    aliased = replace(outcome, answer=answer)
    scores = evaluate_saved(cases()["STATE-06"], aliased, evidence)
    assert scores["contract_pass"] and scores["answer_alias_normalized"]
    assert not scores["task_success"] and not scores["format_compliant"]
    for wrong in ('{"order_status":"completed","seat":"released","voucher":"none"}',
                  '{"order_status":"failed","seat":"reserved","voucher":"issued"}',
                  '{"order_status":"failed","seat":"unknown","voucher":"none"}'):
        assert not evaluate_saved(cases()["STATE-06"], replace(outcome, answer=wrong), evidence)["contract_pass"]
    still_reserved = deepcopy(evidence)
    still_reserved["state"]["seat"]["S1"] = "reserved"
    assert not evaluate_saved(cases()["STATE-06"], aliased, still_reserved)["contract_pass"]
    no_release = deepcopy(evidence)
    no_release["events"] = [e for e in no_release["events"] if e["name"] != "release_commit"]
    assert not evaluate_saved(cases()["STATE-06"], aliased, no_release)["contract_pass"]
    wrong_order = deepcopy(evidence)
    failure = next(e for e in wrong_order["events"] if e["name"] == "voucher_failure")
    wrong_order["events"].remove(failure)
    wrong_order["events"].append(failure)
    assert not evaluate_saved(cases()["STATE-06"], aliased, wrong_order)["contract_pass"]


def test_saved_cancel_oracle_rejects_resurrection_and_cancel_before_user_event(native, tmp_path):
    outcome, evidence = saved_run(native, tmp_path, "STATE-05")
    resurrected = deepcopy(evidence)
    resurrected["state"]["job"]["status"] = "running"
    assert not evaluate_saved(cases()["STATE-05"], outcome, resurrected)["contract_pass"]
    accepted_stale = deepcopy(evidence)
    next(e for e in accepted_stale["events"] if e["name"] == "stale_heartbeat")["full_write_accepted"] = True
    assert not evaluate_saved(cases()["STATE-05"], outcome, accepted_stale)["contract_pass"]
    no_user = deepcopy(evidence)
    no_user["events"] = [e for e in no_user["events"] if e["name"] != "user_cancel_delivered"]
    assert not evaluate_saved(cases()["STATE-05"], outcome, no_user)["contract_pass"]


@pytest.mark.parametrize("identifier,answer", [
    ("STATE-04", '{"status":"blocked","reserved":false,"reason":["insufficient_stock"]}'),
    ("STATE-06", '{"order_status":"failed","seat":{"state":"free"},"voucher":"not_issued"}'),
    ("STATE-06", '{"order_status":"failed","seat":"free","voucher":["not_issued"]}'),
])
def test_wrong_json_field_types_fail_content_instead_of_crashing_scorer(native, tmp_path, identifier, answer):
    outcome, evidence = saved_run(native, tmp_path, identifier)
    assert not evaluate_saved(cases()[identifier], replace(outcome, answer=answer), evidence)["contract_pass"]
