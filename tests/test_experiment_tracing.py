"""Trace files survive execution failures and remain associated with trials."""
from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

from agent_eval import experiments as e
from agent_eval.cli import main
from agent_eval.execution import execute
from agent_eval.tracing import read_trace
from agent_eval.types import AgentOutcome


class SlowAgent:
    def run(self, prompt):
        time.sleep(30)
        return AgentOutcome("42", True, "finished", 1, 1)


def slow_agent():
    return SlowAgent()


def adapt(value):
    return value


def cfg(**changes):
    return {"trials": 1, "trace": {"enabled": True}, "conditions": [
        {"id": "baseline", "agent": "tests.experiment_fixtures:build_agent",
         "outcome_adapter": "tests.experiment_fixtures:adapt"}], **changes}


@pytest.fixture(autouse=True)
def stable_source(monkeypatch):
    monkeypatch.setattr(e, "source_snapshot", lambda roots: [{"root": "fixture", "sha256": "fixture"}])


def test_trace_links_survive_resume_and_rescore_without_duplicate_calls(tmp_path):
    directory = tmp_path / "exp"
    e.create_experiment(directory, [{"id": "x", "prompt": "compute", "expect_number": 42}], cfg(trials=2))
    first = e.run_experiment(directory, max_runs=1)["records"][0]
    report = e.run_experiment(directory)
    traces = list((directory / "traces").glob("*.jsonl"))
    assert len(traces) == 2
    assert len({r["trace"]["trace_id"] for r in report["records"]}) == 2
    assert report["records"][0]["trace"] == first["trace"]
    for record in report["records"]:
        assert record["trace"]["complete"] is True
        assert record["outcome"]["metadata"]["trace_id"] == record["trace"]["trace_id"]
        trace = read_trace(record["trace"]["path"])
        assert any(event.get("name") == "evaluation.run" for event in trace["events"])
        assert trace["complete"]
    previous = {p.name: p.read_bytes() for p in traces}
    e.run_experiment(directory)
    e.rescore_experiment(directory, tmp_path / "rescored.json")
    assert {p.name: p.read_bytes() for p in traces} == previous
    rescored = json.loads((tmp_path / "rescored.json").read_text(encoding="utf-8"))
    assert rescored["records"][0]["trace"] == first["trace"]


def test_execution_failure_is_complete_trace_but_failed_task(tmp_path):
    directory = tmp_path / "exp"
    config = cfg()
    config["conditions"][0]["config"] = {"fail": True}
    e.create_experiment(directory, [{"id": "error", "prompt": "fail"}], config)
    record = e.run_experiment(directory)["records"][0]
    assert record["execution_error"]["type"] == "RuntimeError"
    trace = read_trace(record["trace"]["path"])
    assert trace["complete"] is True
    assert any(event.get("event") == "span.end" and event.get("status") == "error" for event in trace["events"])


def test_timeout_retains_started_trace_and_reports_incomplete(tmp_path):
    directory = tmp_path / "exp"
    config = cfg(timeout_seconds=2)
    config["conditions"][0]["agent"] = "tests.test_experiment_tracing:slow_agent"
    e.create_experiment(directory, [{"id": "timeout", "prompt": "wait"}], config)
    record = e.run_experiment(directory)["records"][0]
    assert record["execution_error"]["type"] == "TimeoutError"
    trace = read_trace(record["trace"]["path"])
    assert any(event.get("event") == "span.start" for event in trace["events"])
    assert trace["complete"] is False
    assert record["trace"]["complete"] is False


def test_trace_setup_failure_does_not_change_agent_result(tmp_path):
    bad_directory = tmp_path / "file"
    bad_directory.write_text("occupied", encoding="utf-8")
    request = {"kind": "agent", "condition": cfg()["conditions"][0], "prompt": "ok",
               "trace": {"directory": str(bad_directory), "task_id": "x", "condition": "a", "trial_id": 0, "trace_id": "abc123"}}
    result = execute(request, timeout_seconds=10)
    assert result["outcome"]["success"]
    assert result["outcome"]["answer"] == "42"
    assert result["trace"]["complete"] is False
    assert result["trace"]["diagnostics"][0]["code"] == "trace_setup_failed"


@pytest.mark.parametrize("trace", [True, {}, {"enabled": 1}, {"enabled": True, "unknown": 1}, {"enabled": True, "instrumentation": "bad"}])
def test_invalid_trace_configuration_fails_before_execution(tmp_path, trace):
    with pytest.raises(ValueError, match="trace"):
        e.create_experiment(tmp_path / "exp", [{"id": "x", "prompt": "ok"}], cfg(trace=trace))
    assert not (tmp_path / "exp").exists()


def test_comparison_protocol_distinguishes_trace_overhead(tmp_path):
    on = e.normalize_config(cfg(), [{"id": "x", "prompt": "x"}])
    off = e.normalize_config(cfg(trace={"enabled": False}), [{"id": "x", "prompt": "x"}])
    assert e.measurement_protocol(on) != e.measurement_protocol(off)


def test_cli_exports_linked_trace_and_scores(tmp_path):
    directory = tmp_path / "exp"
    e.create_experiment(directory, [{"id": "x", "prompt": "ok", "expect_number": 42}], cfg())
    e.run_experiment(directory)
    page = tmp_path / "view.html"
    assert main(["trace-view", str(directory), "--out", str(page)]) == 0
    html = page.read_text(encoding="utf-8")
    assert "answer_correct" in html and "evaluation.run" in html
