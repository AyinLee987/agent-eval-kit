"""The portable viewer treats model output as data, including hostile markup."""
from __future__ import annotations

import json
import re

import pytest

from agent_eval.trace_viewer import export_trace_html, load_traces
from agent_eval.tracing import TraceRecorder


def test_viewer_escapes_script_breakout_and_keeps_complete_inputs(tmp_path):
    recorder = TraceRecorder(tmp_path / "traces", task_id="case")
    hostile = '</script><script>window.bad=true</script>&<img src="https://example.invalid/leak">'
    with recorder.span("llm", "llm", input={"prompt": hostile}) as span:
        span.finish(output="x" * 20000)
    recorder.close()
    path = export_trace_html(recorder.path, tmp_path / "out.html")
    html = path.read_text(encoding="utf-8")
    assert hostile not in html
    match = re.search(r'<script id="trace-data" type="application/json">(.*?)</script>', html, re.S)
    data = json.loads(match.group(1))
    assert any(event.get("input", {}).get("prompt") == hostile for event in data[0]["events"])
    assert "x" * 20000 in html
    assert "connect-src 'none'" in html


def test_viewer_reports_incomplete_span_without_inventing_end(tmp_path):
    recorder = TraceRecorder(tmp_path, task_id="pending")
    recorder.span("still running", "tool")
    traces = load_traces(recorder.path)
    assert traces[0]["complete"] is False
    assert not any(event.get("event") == "span.end" for event in traces[0]["events"])
    recorder.close(status="interrupted")


def test_empty_directory_is_not_an_empty_success_report(tmp_path):
    with pytest.raises(ValueError, match="No JSONL"):
        export_trace_html(tmp_path, tmp_path / "empty.html")


def test_missing_prior_attempt_is_visible_without_current_attempt_scores(tmp_path):
    (tmp_path / "traces").mkdir()
    recorder = TraceRecorder(tmp_path / "traces", task_id="case", trace_id="current")
    recorder.close()
    (tmp_path / "report.json").write_text(json.dumps({"records": [{
        "task_id": "case", "trace": {"trace_id": "current"},
        "trace_attempts": [{"trace_id": "previous"}], "scores": {"answer_correct": 1},
    }]}), encoding="utf-8")
    traces = {trace["trace_id"]: trace for trace in load_traces(tmp_path)}
    assert set(traces) == {"current", "previous"}
    assert traces["previous"]["complete"] is False
    assert traces["previous"]["evaluation"]["historical_attempt"] is True
    assert "scores" not in traces["previous"]["evaluation"]
    assert traces["current"]["evaluation"]["scores"] == {"answer_correct": 1}


def test_linked_report_cannot_reintroduce_credentials_into_export(tmp_path):
    recorder = TraceRecorder(tmp_path / "traces", task_id="case", trace_id="current")
    recorder.close()
    (tmp_path / "report.json").write_text(json.dumps({"records": [{
        "trace": {"trace_id": "current"},
        "execution_error": {"message": "api_key=original-secret Bearer secret-header"},
    }]}), encoding="utf-8")
    exported = export_trace_html(tmp_path, tmp_path / "index.html").read_text(encoding="utf-8")
    assert "original-secret" not in exported and "secret-header" not in exported
    assert "[REDACTED]" in exported
