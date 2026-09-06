import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import threading

import pytest

from agent_eval.tracing import TraceRecorder, read_trace, sanitize_trace_value


def _codes(trace):
    return {item["code"] for item in trace["diagnostics"]}


def test_public_sanitizer_redacts_associated_report_errors():
    original = {"execution_error": {"message": "api_key=report-secret Bearer abcdef123456"}, "tokens": 12}
    sanitized = sanitize_trace_value(original)
    assert sanitized == {"execution_error": {"message": "api_key=[REDACTED] Bearer [REDACTED]"}, "tokens": 12}
    assert "report-secret" in original["execution_error"]["message"]


def test_full_trace_preserves_inputs_outputs_usage_and_hierarchy(tmp_path):
    recorder = TraceRecorder(tmp_path, task_id="task-1", condition="recovery", trial_id=2)
    with recorder.span("task", "agent", input={"prompt": "calculate"}) as root:
        with recorder.span("completion", "llm", input={"messages": [{"role": "user", "content": "2+2"}]}) as llm:
            llm.finish(output={"tool_calls": [{"name": "calculator", "expression": "2+2"}]},
                       usage={"input_tokens": 8, "output_tokens": 4, "total_tokens": 12})
        with recorder.span("calculator", "tool", input={"expression": "2+2"}) as tool:
            tool.event("result_received", {"result": 4})
            tool.finish(output=4)
        root.finish(output="4")
    recorder.close(metadata={"successful": True})
    recorder.close()

    trace = read_trace(recorder.path)
    assert trace["complete"] and not trace["diagnostics"]
    assert not recorder.errors
    assert len(trace["spans"]) == 3
    root_data, llm_data, tool_data = trace["spans"]
    assert root_data["input"] == {"prompt": "calculate"}
    assert root_data["output"] == "4"
    assert llm_data["parent_id"] == root.span_id
    assert llm_data["usage"]["total_tokens"] == 12
    assert tool_data["output"] == 4 and tool_data["events"][0]["data"] == {"result": 4}
    assert all(span["duration_ms"] >= 0 for span in trace["spans"])
    assert [event["sequence"] for event in trace["events"]] == list(range(1, len(trace["events"]) + 1))
    assert all(event["task_id"] == "task-1" and event["condition"] == "recovery" and event["trial_id"] == 2
               for event in trace["events"])
    assert trace["manifest"]["written_event_count"] == len(trace["events"])


def test_incremental_writes_are_visible_before_close(tmp_path):
    recorder = TraceRecorder(tmp_path, task_id="stream")
    recorder.event("checkpoint", {"state": "visible"})
    trace = read_trace(recorder.path)
    assert [event["event"] for event in trace["events"]] == ["trace.start", "event"]
    assert not trace["complete"]
    assert "missing_trace_end" in _codes(trace)
    recorder.close()
    assert read_trace(recorder.path)["complete"]


def test_exception_propagates_and_context_is_restored(tmp_path):
    recorder = TraceRecorder(tmp_path, task_id="exception")
    expected = ValueError("request failed api_key=secret-value")
    with recorder.span("root", "agent") as root:
        with pytest.raises(ValueError) as caught:
            with recorder.span("bad tool", "tool"):
                raise expected
        assert caught.value is expected
        with recorder.span("recovery", "tool") as recovery:
            assert recovery.parent_span_id == root.span_id
    with recorder.span("next root", "agent") as new_root:
        assert new_root.parent_span_id is None
    recorder.close(status="error")
    trace = read_trace(recorder.path)
    failed = trace["spans"][1]
    assert trace["complete"]  # Completeness describes trace integrity, not task success.
    assert failed["status"] == "error"
    assert failed["end_metadata"]["exception"]["type"] == "ValueError"
    assert "secret-value" not in recorder.path.read_text(encoding="utf-8")


def test_exception_with_broken_string_does_not_replace_business_exception(tmp_path):
    class BrokenException(Exception):
        def __str__(self):
            raise RuntimeError("no message")

    recorder = TraceRecorder(tmp_path, task_id="broken-exception")
    error = BrokenException()
    with pytest.raises(BrokenException) as caught:
        with recorder.span("failing", "tool"):
            raise error
    recorder.close(status="error")
    assert caught.value is error
    assert read_trace(recorder.path)["spans"][0]["status"] == "error"


def test_recursive_redaction_keeps_token_counters_and_full_content(tmp_path):
    recorder = TraceRecorder(tmp_path, task_id="redaction")
    long_text = "complete observable content " * 10000
    recorder.event("payload", {
        "headers": {"Authorization": "Basic not-for-log", "X-API-Key": "nested-key"},
        "credentials": {"name": "sensitive"}, "PASSWORD": "abc123", "OPENAI_API_KEY": "provider-key",
        "access_token": "access-value", "refresh_token": "refresh-value", "client_secret": "client-value",
        "messages": [{"content": "Bearer abc.def-123 sk-123456789abcdef api_key=text-value password=hidden"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                  "cached_tokens": 3, "tokens": 15, "token_count": 15},
        "long": long_text,
    })
    recorder.close()
    raw = recorder.path.read_text(encoding="utf-8")
    for secret in ["not-for-log", "nested-key", "abc123", "provider-key", "access-value", "refresh-value",
                   "client-value", "abc.def-123", "sk-123456789abcdef", "text-value", "hidden"]:
        assert secret not in raw
    payload = read_trace(recorder.path)["events"][1]["data"]
    assert payload["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                                "cached_tokens": 3, "tokens": 15, "token_count": 15}
    assert payload["long"] == long_text


def test_json_safe_snapshot_handles_cycles_bad_repr_and_mutation(tmp_path):
    class BrokenRepr:
        def __repr__(self):
            raise ValueError("cannot represent")

    @dataclass
    class Credentials:
        api_key: str
        count: int

    recorder = TraceRecorder(tmp_path, task_id="serialize")
    cyclic = [1]
    cyclic.append(cyclic)
    mutable = {"items": [1]}
    recorder.event("values", {"cycle": cyclic, "broken": BrokenRepr(), "data": Credentials("secret", 3),
                              "path": Path("example"), "nan": float("nan"), "mutable": mutable})
    mutable["items"].append(2)
    recorder.close()
    trace = read_trace(recorder.path)
    assert trace["complete"]
    value = trace["events"][1]["data"]
    assert value["cycle"] == [1, {"__cycle__": "list"}]
    assert value["broken"]["__serialization_error__"] == "ValueError"
    assert value["data"] == {"api_key": "[REDACTED]", "count": 3}
    assert value["nan"] == {"__nonfinite_float__": "nan"}
    assert value["mutable"] == {"items": [1]}


def test_threads_are_isolated_and_explicit_worker_parent_is_supported(tmp_path):
    recorder = TraceRecorder(tmp_path, task_id="parallel")
    barrier = threading.Barrier(4)
    with recorder.span("root", "agent") as root:
        def worker(index):
            barrier.wait()
            with recorder.span(f"worker-{index}", "worker", parent_span_id=root.span_id) as parent:
                for number in range(10):
                    with recorder.span(f"tool-{number}", "tool") as child:
                        assert child.parent_span_id == parent.span_id
                        child.finish(output={"worker": index, "number": number})
            return parent.span_id

        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(worker, range(4)))
        recorder.event("workers_joined", ids)
    recorder.close()
    trace = read_trace(recorder.path)
    assert trace["complete"] and len(trace["spans"]) == 45
    assert all(span["parent_id"] == root.span_id for span in trace["spans"] if span["kind"] == "worker")
    assert trace["events"][-3]["span_id"] == root.span_id


def test_async_contexts_preserve_separate_parent_stacks(tmp_path):
    recorder = TraceRecorder(tmp_path, task_id="async")

    async def work():
        async def worker(index):
            with recorder.span(f"worker-{index}", "worker") as parent:
                await asyncio.sleep(0)
                with recorder.span("tool", "tool") as child:
                    assert child.parent_span_id == parent.span_id
        with recorder.span("root", "agent"):
            await asyncio.gather(*(worker(index) for index in range(5)))

    asyncio.run(work())
    recorder.close()
    assert read_trace(recorder.path)["complete"]


def test_cross_trace_context_and_explicit_parents_are_not_mixed(tmp_path):
    first = TraceRecorder(tmp_path / "one", task_id="one", trace_id="same-external-id")
    second = TraceRecorder(tmp_path / "two", task_id="two", trace_id="same-external-id")
    with first.span("root", "agent") as root:
        with second.span("other root", "agent") as other:
            assert other.parent_span_id is None
            with pytest.raises(ValueError, match="parent_span_id"):
                second.span("wrong parent", "worker", parent_span_id=root.span_id)
            with pytest.raises(ValueError, match="span_id"):
                second.event("wrong span", span_id=root.span_id)
        with first.span("next", "tool") as child:
            assert child.parent_span_id == root.span_id
    first.close()
    second.close()
    assert read_trace(first.path)["complete"] and read_trace(second.path)["complete"]


def test_different_recorders_in_parallel_have_unique_files(tmp_path):
    def run(index):
        recorder = TraceRecorder(tmp_path, task_id="same-task", condition="same", trial_id=index)
        with recorder.span("agent", "agent"):
            recorder.event("trial", index)
        recorder.close()
        return recorder.path

    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(run, range(16)))
    assert len(set(paths)) == 16
    for path in paths:
        trace = read_trace(path)
        assert trace["complete"]
        assert len({event["trial_id"] for event in trace["events"]}) == 1


def test_logging_failure_does_not_fail_agent_and_is_in_inventory(tmp_path):
    recorder = TraceRecorder(tmp_path, task_id="disk-error")
    original = recorder._file

    class FailingWrite:
        def write(self, value):
            raise OSError("disk full api_key=must-redact")

        def flush(self):
            pass

    recorder._file = FailingWrite()
    with recorder.span("tool", "tool") as span:
        result = 42
        span.finish(output=result)
    recorder._file = original
    recorder.close()
    assert result == 42
    assert len(recorder.errors) == 2
    trace = read_trace(recorder.path)
    assert not trace["complete"] and "logging_errors" in _codes(trace)
    assert "sequence_gap" in _codes(trace) and "event_count_mismatch" in _codes(trace)
    assert trace["manifest"]["dropped_sequences"] == [2, 3]
    assert "must-redact" not in recorder.manifest_path.read_text(encoding="utf-8")


def test_manifest_write_failure_is_reported_without_stopping_work(tmp_path, monkeypatch):
    recorder = TraceRecorder(tmp_path, task_id="manifest-error")
    original = Path.write_text

    def fail_manifest(path, *args, **kwargs):
        if path.name.endswith(".manifest.json.tmp"):
            raise OSError("inventory unavailable")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", fail_manifest)
        recorder.event("still-running", {"result": 7})
    recorder.close()
    assert recorder.errors[0]["operation"] == "manifest.write"
    trace = read_trace(recorder.path)
    assert len(trace["events"]) == 3
    assert "logging_errors" in _codes(trace)


def test_partial_write_retains_prior_events_and_reports_corruption(tmp_path):
    recorder = TraceRecorder(tmp_path, task_id="partial-write")
    recorder.event("saved", {"result": "keep"})
    original = recorder._file

    class PartialWrite:
        def write(self, value):
            original.write(value[:20])
            original.flush()
            return 20

        def flush(self):
            original.flush()

    recorder._file = PartialWrite()
    recorder.event("partial", {"result": "incomplete"})
    trace = read_trace(recorder.path)
    assert len(trace["events"]) == 2
    assert trace["events"][1]["data"] == {"result": "keep"}
    assert "incomplete_tail" in _codes(trace)
    recorder._file = original
    recorder.close()
    trace = read_trace(recorder.path)
    assert "corrupt_line" in _codes(trace) and "logging_errors" in _codes(trace)


def test_write_failure_never_replaces_original_exception(tmp_path):
    recorder = TraceRecorder(tmp_path, task_id="both-fail")
    span = recorder.span("tool", "tool")
    recorder._file.close()
    expected = RuntimeError("business failure")
    with pytest.raises(RuntimeError) as caught:
        with span:
            raise expected
    recorder.close(status="error")
    assert caught.value is expected
    assert recorder.errors


def test_unfinished_span_and_post_close_events_are_explicit(tmp_path):
    recorder = TraceRecorder(tmp_path, task_id="unfinished")
    span = recorder.span("in-flight", "tool")
    recorder.close(status="cancelled")
    trace = read_trace(recorder.path)
    assert "unfinished_span" in _codes(trace)
    assert trace["spans"][0]["status"] == "incomplete"
    assert trace["manifest"]["status"] == "incomplete"
    span.finish(output="late")
    assert recorder.errors[-1]["operation"] == "event.after_close"
    assert "logging_errors" in _codes(read_trace(recorder.path))


@pytest.mark.parametrize("tail", [b'{"event":"span.', b'{"event":"event"}', b'\xff\xfe'])
def test_reader_keeps_valid_prefix_of_incomplete_tail(tmp_path, tail):
    recorder = TraceRecorder(tmp_path, task_id="tail")
    recorder.close()
    with recorder.path.open("ab") as file:
        file.write(tail)
    trace = read_trace(recorder.path)
    assert not trace["complete"]
    assert "incomplete_tail" in _codes(trace)
    assert trace["events"][0]["event"] == "trace.start"


def test_reader_reports_middle_corruption_and_keeps_following_valid_records(tmp_path):
    recorder = TraceRecorder(tmp_path, task_id="corrupt")
    recorder.event("middle", 1)
    recorder.close()
    lines = recorder.path.read_bytes().splitlines(keepends=True)
    lines[1] = b"not json\n"
    recorder.path.write_bytes(b"".join(lines))
    trace = read_trace(recorder.path)
    assert not trace["complete"]
    assert {"corrupt_line", "sequence_gap", "event_count_mismatch"} <= _codes(trace)
    assert trace["events"][-1]["event"] == "trace.end"


def test_inventory_detects_missing_trace_file(tmp_path):
    recorder = TraceRecorder(tmp_path, task_id="missing")
    recorder.close()
    recorder.path.unlink()
    trace = read_trace(recorder.path)
    assert not trace["complete"] and "missing_trace" in _codes(trace)
    assert trace["trace_id"] == recorder.trace_id
    assert trace["manifest"]["intended_event_count"] == 2


def test_missing_inventory_and_invalid_record_types_are_reported(tmp_path):
    recorder = TraceRecorder(tmp_path, task_id="malformed")
    recorder.close()
    recorder.manifest_path.unlink()
    with recorder.path.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"schema_version": 1, "trace_id": recorder.trace_id,
                               "event": "span.end", "span_id": [], "sequence": 3}) + "\n")
    trace = read_trace(recorder.path)
    assert not trace["complete"]
    assert {"missing_manifest", "orphan_span_end", "missing_trace_end"} <= _codes(trace)


def test_corrupt_json_identifiers_and_nonfinite_data_cannot_break_export(tmp_path):
    recorder = TraceRecorder(tmp_path, task_id="invalid-json")
    recorder.close()
    manifest = json.loads(recorder.manifest_path.read_text(encoding="utf-8"))
    manifest["trace_id"] = ["invalid"]
    recorder.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with recorder.path.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"schema_version": 1, "trace_id": {"invalid": True},
                               "event": [], "sequence": 3}) + "\n")
        file.write('{"schema_version":1,"data":NaN}\n')
    trace = read_trace(recorder.path)
    assert not trace["complete"]
    assert {"invalid_manifest", "invalid_trace_id", "unknown_event", "corrupt_line"} <= _codes(trace)
    assert trace["trace_id"] == recorder.trace_id
    json.dumps(trace, allow_nan=False)


def test_invalid_directory_and_duplicate_explicit_id_fail_before_overwrite(tmp_path):
    occupied = tmp_path / "file"
    occupied.write_text("existing", encoding="utf-8")
    with pytest.raises(OSError):
        TraceRecorder(occupied, task_id="bad-directory")
    with pytest.raises(ValueError):
        TraceRecorder(tmp_path, task_id="bad-id", trace_id="../escape")
    recorder = TraceRecorder(tmp_path, task_id="one", trace_id="fixed")
    recorder.close()
    original = recorder.path.read_bytes()
    with pytest.raises(FileExistsError):
        TraceRecorder(tmp_path, task_id="two", trace_id="fixed")
    assert recorder.path.read_bytes() == original
