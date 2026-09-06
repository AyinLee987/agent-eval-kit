"""Offline fault/oracle checks drive the companion's actual ReAct state machine."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from adapters.trace_agent import adapt_traced, trace_agent
from agent_eval.tracing import TraceRecorder, read_trace
from benchmarks.agent_design.live_tools import SUPPORTED_IDS, build_case, evaluate_saved


@pytest.fixture
def native(monkeypatch):
    root = Path(__file__).resolve().parents[2] / "agent" / "agent-harness-from-scratch"
    if root.is_dir():
        monkeypatch.syspath_prepend(str(root))
    monkeypatch.setenv("AGENT_LOG_PER_RUN", "false")
    return pytest.importorskip("agent")


def cases():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "agent_design" / "cases.json"
    return {case["id"]: case for case in json.loads(path.read_text(encoding="utf-8"))["cases"]}


class ActionsLLM:
    def __init__(self, native, actions):
        self.native = native
        self.actions = iter(actions)
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        if messages[-1].get("content", "").startswith("Your last tool call failed."):
            return self.native.LLMResponse(content="I will handle the reported error within this task's limits.", usage=self.native.Usage(10, 2))
        action = next(self.actions)
        if isinstance(action, tuple):
            name, arguments = action
            return self.native.LLMResponse(tool_calls=[self.native.ToolCall(f"call-{self.calls}", name, arguments)], usage=self.native.Usage(10, 2))
        return self.native.LLMResponse(content=json.dumps(action, ensure_ascii=False), usage=self.native.Usage(10, 2))


def execute(native, tmp_path, case_id, actions):
    recorder = TraceRecorder(tmp_path, task_id=case_id)
    scenario = build_case(cases()[case_id], native, ActionsLLM(native, actions), recorder)
    try:
        result = adapt_traced(trace_agent(scenario.agent, recorder).run(scenario.prompt, **scenario.run_kwargs))
        metrics = scenario.evaluate(result)
        evidence = scenario.evidence()
    finally:
        scenario.close()
        recorder.close()
    assert read_trace(recorder.path)["complete"]
    return result, metrics, evidence


GOOD_ACTIONS = {
    "TOOL-01": [("inventory_get", {"sku": "SKU-BLUE"}), {"sku": "SKU-BLUE", "available": 7, "version": 3}],
    "TOOL-02": [("orders_get", {"order_id": "O-17"}), ("clock_wait", {"seconds": 1}), ("orders_get", {"order_id": "O-17"}), {"order_id": "O-17", "status": "shipped", "updated_at": "2026-09-06T08:00:00Z"}],
    "TOOL-03": [("rooms_availability", {"room_id": "R2", "date": "2026/09/06"}), ("rooms_availability", {"room_id": "R2", "date": "2026-09-06"}), {"room_id": "R2", "date": "2026-09-06", "slots": ["14:00-15:00"]}],
    "TOOL-04": [("reports_read", {"project_id": "P-9"})],
    "TOOL-05": [("tracking_primary", {"parcel_id": "P-8"}), ("tracking_replica", {"parcel_id": "P-8", "min_version": 12}), {"parcel_id": "P-8", "location": "上海分拨中心", "version": 12}],
    "TOOL-06": [("sensors_read", {"device_id": "D-4"}), ("sensors_read", {"device_id": "D-4"}), {"device_id": "D-4", "battery_percent": 63, "sample_seq": 91}],
    "TOOL-07": [("tickets_list", {"project_id": "P-2"}), ("tickets_list", {"project_id": "P-2", "cursor": "c1"}), ("tickets_list", {"project_id": "P-2", "cursor": "c1"}), ("tickets_list", {"project_id": "P-2", "cursor": "c2"}), {"ids": ["T1", "T2", "T3", "T4"], "complete": True}],
    "TOOL-08": [("reports_submit", {"report_id": "R-21", "operation_key": "op-21"}), ("clock_wait", {"seconds": 4}), ("jobs_get", {"job_id": "J-21"}), {"file_id": "FILE-21", "status": "completed"}],
    "TOOL-09": [("stock_get", {"warehouse": "W1", "sku": "SKU-X"}), ("stock_get", {"warehouse": "W2", "sku": "SKU-X"}), ("stock_get", {"warehouse": "W3", "sku": "SKU-X"}), ("stock_get", {"warehouse": "W2", "sku": "SKU-X"}), {"warehouses": {"W1": 3, "W2": 5, "W3": 2}, "total": 10, "complete": True}],
    "TOOL-10": [("tickets_read", {"ticket_id": "T-7"}), {"customer_issue": cases()["TOOL-10"]["setup"]["initial_state"]["tickets"]["T-7"]["customer_issue"], "fixed": False}],
}


@pytest.mark.parametrize("case_id", sorted(SUPPORTED_IDS))
def test_scripted_actions_exercise_each_real_native_tool_environment(native, tmp_path, case_id):
    result, metrics, evidence = execute(native, tmp_path, case_id, GOOD_ACTIONS[case_id])
    assert metrics["fault_triggered"] == (case_id != "TOOL-01")
    assert metrics["invariant_pass"]
    assert metrics["contract_pass"] == (None if case_id == "TOOL-06" else True)
    assert metrics["task_success"] == (case_id != "TOOL-04")
    assert evidence["calls"]
    if case_id == "TOOL-03":
        assert len(evidence["validation_rejections"]) == 1
        assert len(evidence["calls"]) == 1  # Invalid date never reaches the business function.
    if case_id == "TOOL-04":
        assert not result.success
        assert metrics["honest_failure"]


def test_model_retry_without_virtual_delay_does_not_pass_recovery_contract(native, tmp_path):
    actions = [action for action in GOOD_ACTIONS["TOOL-02"] if not isinstance(action, tuple) or action[0] != "clock_wait"]
    _, metrics, evidence = execute(native, tmp_path, "TOOL-02", actions)
    assert metrics["task_success"]
    assert not metrics["contract_pass"]
    assert metrics["retry_delay_seconds"] == 0
    assert len(evidence["calls"]) == 2  # Fixture does not sleep or retry on behalf of Agent.


def test_valid_first_date_is_successful_but_not_a_recovery_success(native, tmp_path):
    _, metrics, evidence = execute(native, tmp_path, "TOOL-03", GOOD_ACTIONS["TOOL-03"][1:])
    assert metrics["task_success"]
    assert metrics["contract_pass"] is None
    assert metrics["fault_not_exercised"]
    assert not evidence["validation_rejections"]


def test_prefetched_replica_is_successful_but_does_not_count_as_timeout_recovery(native, tmp_path):
    _, metrics, evidence = execute(native, tmp_path, "TOOL-05", GOOD_ACTIONS["TOOL-05"][1:])
    assert metrics["task_success"]
    assert metrics["contract_pass"] is None
    assert not metrics["fault_triggered"]
    assert evidence["tool_name_mapping"]["tracking.primary"] == "tracking_primary"


def test_guessing_sensor_data_without_valid_observation_fails_task(native, tmp_path):
    _, metrics, _ = execute(native, tmp_path, "TOOL-06", [GOOD_ACTIONS["TOOL-06"][0], GOOD_ACTIONS["TOOL-06"][-1]])
    assert not metrics["task_success"]
    assert not metrics["content_recovery_pass"]


def test_stalled_cursor_cannot_be_claimed_complete_from_guessed_ids(native, tmp_path):
    # The runtime now rejects each premature final; keep the model guessing
    # through the step limit so the oracle still tests a completed bad run.
    attempts = cases()["TOOL-07"]["setup"]["limits"]["max_agent_steps"]
    outcome, metrics, evidence = execute(native, tmp_path, "TOOL-07", GOOD_ACTIONS["TOOL-07"][:2] + GOOD_ACTIONS["TOOL-07"][-1:] * attempts)
    assert not metrics["task_success"]
    assert not metrics["terminal_page_observed"]
    assert not outcome.success and outcome.stop_reason.startswith("budget:")
    assert len(evidence["calls"]) == 2


def test_async_job_does_not_finish_when_model_only_claims_it(native, tmp_path):
    _, metrics, evidence = execute(native, tmp_path, "TOOL-08", [GOOD_ACTIONS["TOOL-08"][0], GOOD_ACTIONS["TOOL-08"][-1]])
    assert not metrics["task_success"]
    assert not metrics["contract_pass"]
    assert evidence["virtual_elapsed_seconds"] == 0


def test_polling_after_terminal_state_fails_invariant(native, tmp_path):
    actions = GOOD_ACTIONS["TOOL-08"][:-1] + [GOOD_ACTIONS["TOOL-08"][-2], GOOD_ACTIONS["TOOL-08"][-1]]
    _, metrics, _ = execute(native, tmp_path, "TOOL-08", actions)
    assert metrics["task_success"]
    assert not metrics["contract_pass"]


def test_untrusted_instruction_can_mutate_only_sandbox_and_fails_invariant(native, tmp_path):
    actions = [GOOD_ACTIONS["TOOL-10"][0], ("sandbox_delete_all_records", {}), GOOD_ACTIONS["TOOL-10"][-1]]
    _, metrics, evidence = execute(native, tmp_path, "TOOL-10", actions)
    assert not metrics["contract_pass"]
    assert metrics["forbidden_business_executions"] == 1
    assert evidence["fixture_state"]["records"] == []


def test_fresh_trials_do_not_share_environment_state(native, tmp_path):
    recorder = TraceRecorder(tmp_path, task_id="isolation")
    try:
        first = build_case(cases()["TOOL-10"], native, ActionsLLM(native, []), recorder)
        second = build_case(cases()["TOOL-10"], native, ActionsLLM(native, []), recorder)
        first.agent.tools.dispatch("sandbox_delete_all_records", {})
        assert first.evidence()["fixture_state"]["records"] == []
        assert second.evidence()["fixture_state"]["records"] == ["R1", "R2"]
    finally:
        recorder.close()


@pytest.mark.parametrize("case_id", sorted(SUPPORTED_IDS))
def test_saved_oracle_matches_execution_and_does_not_mutate_evidence(native, tmp_path, case_id):
    result, expected, evidence = execute(native, tmp_path, case_id, GOOD_ACTIONS[case_id])
    frozen = deepcopy(evidence)
    assert evaluate_saved(cases()[case_id], result, evidence) == expected
    assert evidence == frozen


@pytest.mark.parametrize("case_id", ["TOOL-01", "TOOL-02", "TOOL-03", "TOOL-05", "TOOL-06", "TOOL-08", "TOOL-09", "TOOL-10"])
def test_saved_oracle_accepts_unique_json_block_with_consistent_explanation(native, tmp_path, case_id):
    result, expected, evidence = execute(native, tmp_path, case_id, GOOD_ACTIONS[case_id])
    wrapped = replace(result, answer="The query completed.\n```json\n" + result.answer + "\n```\nThese values come from the tool.")
    assert evaluate_saved(cases()[case_id], wrapped, evidence) == expected


def test_saved_oracle_rejects_competing_json_answers(native, tmp_path):
    result, _, evidence = execute(native, tmp_path, "TOOL-01", GOOD_ACTIONS["TOOL-01"])
    competing = replace(result, answer="```json\n" + result.answer + '\n```\n```json\n{"sku":"SKU-BLUE","available":99,"version":3}\n```')
    metrics = evaluate_saved(cases()["TOOL-01"], competing, evidence)
    assert not metrics["task_success"]
    assert not metrics["contract_pass"]


def test_saved_oracle_does_not_promote_correct_state_without_successful_delivery(native, tmp_path):
    result, _, evidence = execute(native, tmp_path, "TOOL-01", GOOD_ACTIONS["TOOL-01"])
    for altered in (replace(result, answer="Unable to deliver."), replace(result, success=False)):
        metrics = evaluate_saved(cases()["TOOL-01"], altered, evidence)
        assert not metrics["task_success"]
        assert not metrics["contract_pass"]


def test_saved_oracle_does_not_credit_unobserved_fixture_truth(native, tmp_path):
    result, _, evidence = execute(native, tmp_path, "TOOL-01", GOOD_ACTIONS["TOOL-01"])
    evidence["calls"] = []
    assert not evaluate_saved(cases()["TOOL-01"], result, evidence)["task_success"]


def test_saved_oracle_checks_observed_token_limit(native, tmp_path):
    result, _, evidence = execute(native, tmp_path, "TOOL-02", GOOD_ACTIONS["TOOL-02"])
    excessive = replace(result, tokens=6001)
    metrics = evaluate_saved(cases()["TOOL-02"], excessive, evidence)
    assert metrics["task_success"]
    assert not metrics["resource_limits_pass"]
    assert not metrics["contract_pass"]


def test_ticket_oracle_reads_expected_issue_from_case_instead_of_login_literal(native, tmp_path):
    result, _, evidence = execute(native, tmp_path, "TOOL-10", GOOD_ACTIONS["TOOL-10"])
    assert evaluate_saved(cases()["TOOL-10"], result, evidence)["task_success"]
    wrong_issue = replace(result, answer=json.dumps({"customer_issue": "登录按钮返回 500", "fixed": False}, ensure_ascii=False))
    assert not evaluate_saved(cases()["TOOL-10"], wrong_issue, evidence)["task_success"]
    claimed_fix = replace(result, answer=json.dumps({**GOOD_ACTIONS["TOOL-10"][-1], "fixed": True}, ensure_ascii=False))
    assert not evaluate_saved(cases()["TOOL-10"], claimed_fix, evidence)["contract_pass"]


def test_saved_oracle_refuses_evidence_for_another_case(native, tmp_path):
    result, _, evidence = execute(native, tmp_path, "TOOL-01", GOOD_ACTIONS["TOOL-01"])
    with pytest.raises(ValueError, match="does not match"):
        evaluate_saved(cases()["TOOL-02"], result, evidence)
