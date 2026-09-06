"""The paid runner must preserve failure/cost evidence without paid requests."""
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_eval.experiments import create_experiment, run_experiment, rescore_experiment
from agent_eval.tracing import TraceRecorder, read_trace
from benchmarks.agent_design import live


HARNESS = Path(__file__).resolve().parents[2] / "agent" / "agent-harness-from-scratch"


def test_declared_coverage_separates_live_controlled_and_missing_protocol():
    entries = [live.coverage(case) for case in live.load_catalog()["cases"]]
    assert Counter(entry["status"] for entry in entries) == {"planned": 37, "unsupported": 13}
    assert Counter(entry.get("mode") for entry in entries) == {"live_model": 29, "deterministic_runtime": 8, None: 13}


def test_controlled_runtime_goes_through_durable_worker_scoring_and_trace(tmp_path, monkeypatch):
    if not HARNESS.exists():
        pytest.skip("companion checkout unavailable")
    monkeypatch.setenv("AGENT_LOG_PER_RUN", "false")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    case = next(c for c in live.load_catalog()["cases"] if c["id"] == "BUDGET-10")
    root = tmp_path / "experiment"
    create_experiment(root, [{"id": case["id"], "prompt": case["prompt"]}], {
        "trials": 1, "timeout_seconds": 30,
        "conditions": [{"id": "current_agent", "agent": "benchmarks.agent_design.live:build_agent",
                        "outcome_adapter": "benchmarks.agent_design.live:adapt",
                        "config": {"case": case, "harness_path": str(HARNESS)}}],
        "trace": {"enabled": True, "instrumentation": "benchmarks.agent_design.live:instrument"},
        "scorers": [{"factory": "benchmarks.agent_design.live:DesignScorer"}],
    })
    run_experiment(root)
    record = live.read_records(root)[0]
    assert not record["execution_error"]
    assert not record["scorer_errors"]
    assert record["scores"]["contract_pass"] is True
    assert record["outcome"]["tokens"] == 0
    assert record["outcome"]["metadata"]["runtime_reported_tokens"] > 0
    assert record["outcome"]["metadata"]["api_calls"] == []
    trace = read_trace(record["trace"]["path"])
    assert trace["complete"]
    assert any(event.get("name") == "design.oracle" for event in trace["events"])
    run_experiment(root)
    assert len(live.read_records(root)) == 1
    saved_before = json.dumps(live.read_records(root), sort_keys=True)
    revised = rescore_experiment(root, tmp_path / "rescored.json", scorers=[{
        "factory": "benchmarks.agent_design.rescore_live:EvidenceScorer", "config": {"case": case}}])
    reviewed = revised["records"][0]
    assert reviewed["scores"]["contract_pass"] is True
    assert not reviewed["scorer_errors"]
    assert reviewed["trace"]["trace_id"] == record["trace"]["trace_id"]
    assert reviewed["outcome"]["metadata"]["api_calls"] == []
    assert json.dumps(live.read_records(root), sort_keys=True) == saved_before


def test_accounting_flushes_even_when_model_adapter_raises(tmp_path):
    client = SimpleNamespace(calls=[], failure=None, max_calls=24, max_tokens=6000)
    class FailingModel:
        def chat(self, messages, tools=None):
            client.calls.append({"call_index": 0, "usage": {"total_tokens": 17}, "usage_complete": False})
            raise RuntimeError("decoder failed")
    recorder = TraceRecorder(tmp_path, task_id="accounting")
    with pytest.raises(RuntimeError, match="decoder"):
        live.AccountedLLM(FailingModel(), client, recorder).chat([])
    recorder.close()
    events = read_trace(recorder.path)["events"]
    assert [e["name"] for e in events if e["name"].startswith("provider.")] == ["provider.request.started", "provider.request.finished"]
    assert next(e for e in events if e["name"] == "provider.request.finished")["data"]["usage"]["total_tokens"] == 17


@pytest.mark.parametrize("scoring_failure", [False, True])
def test_summary_preserves_unknown_cost_and_distinguishes_broken_scorer(tmp_path, monkeypatch, scoring_failure):
    recorder = TraceRecorder(tmp_path / "source", task_id="TOOL-01")
    recorder.event("provider.request.started", {"call_index": 0})
    recorder.event("provider.request.finished", {"call_index": 0, "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17}, "usage_complete": True, "model": "deepseek-v4-flash"})
    recorder.event("provider.request.started", {"call_index": 1})
    recorder.close()
    record = {"task_id": "TOOL-01", "outcome": {"answer": "", "tokens": 0, "stop_reason": "error", "metadata": {}},
              "scores": {"contract_pass": None}, "execution_error": None if scoring_failure else {"type": "timeout"},
              "scorer_errors": [{"type": "score_timeout"}] if scoring_failure else [],
              "trace": {"path": str(recorder.path), "trace_id": recorder.trace_id}}
    (tmp_path / "run_manifest.json").write_text(json.dumps({"model": "deepseek-v4-flash", "entries": [
        {"task_id": "TOOL-01", "title": "fixture", "category": "tools", "status": "planned", "mode": "live_model", "experiment": "cases/TOOL-01"}]}), encoding="utf-8")
    monkeypatch.setattr(live, "read_records", lambda _: [record])
    monkeypatch.setattr(live, "export_trace_html", lambda *args: None)
    report = live.summarize(tmp_path)
    assert report["counts"] == {"scoring_error" if scoring_failure else "execution_error": 1}
    assert report["known_usage"]["total_tokens"] == 17
    assert report["real_api_attempts"] == 2
    assert report["requests_with_usage"] == 1
    assert not report["usage_complete"]
    assert report["unknown_accounting_runs"] == ["TOOL-01"]
