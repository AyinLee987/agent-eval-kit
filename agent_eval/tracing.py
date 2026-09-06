"""Incremental, dependency-free traces of observable agent execution.

Traces contain calls, messages, results and usage supplied by an adapter. They do
not request or reconstruct a model's private chain of thought. Each trace has an
independent JSONL file and an inventory manifest; neither needs an external
observability service. Log failures are retained in ``recorder.errors`` and never
replace the exception or return value of the instrumented operation.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
import json
import math
from pathlib import Path
import re
import threading
import time
import traceback
from typing import Any
from uuid import uuid4


_CURRENT_SPAN: ContextVar[tuple[object, str] | None] = ContextVar("agent_eval_current_span", default=None)
_REDACTED = "[REDACTED]"
_SECRET_KEYS = {
    "apikey", "password", "passwd", "authorization", "proxyauthorization",
    "accesstoken", "refreshtoken", "sessiontoken", "clientsecret", "privatekey",
    "secret", "secrets", "credential", "credentials", "cookie", "setcookie",
    "token", "auth", "bearer",
}
_SECRET_SUFFIXES = ("apikey", "password", "passwd", "accesstoken", "refreshtoken", "clientsecret", "privatekey")
_BEARER = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_SK_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")
_TEXT_SECRET = re.compile(
    r"(?P<prefix>\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|client[_-]?secret)"
    r"[\"']?\s*[:=]\s*[\"']?)(?P<value>[^\s\"',;}]+)", re.IGNORECASE,
)


def _redact_text(value: str) -> str:
    value = _BEARER.sub("Bearer " + _REDACTED, value)
    value = _SK_KEY.sub(_REDACTED, value)
    return _TEXT_SECRET.sub(lambda match: match.group("prefix") + _REDACTED, value)


def _secret_key(key: str, value: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    # Usage counters are observable metrics, including a numeric bare `tokens`.
    if normalized in {"token", "tokens"} and isinstance(value, (int, float)):
        return False
    return normalized in _SECRET_KEYS or normalized.endswith(_SECRET_SUFFIXES)


def _safe_value(value: Any, seen: set[int] | None = None) -> Any:
    """Snapshot arbitrary values without invoking JSON's unsafe default fallback."""
    if seen is None:
        seen = set()
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else {"__nonfinite_float__": str(value)}
    if isinstance(value, (date, datetime, Path)):
        return _redact_text(str(value))
    if isinstance(value, bytes):
        return {"__bytes_utf8__": _redact_text(value.decode("utf-8", errors="replace"))}
    if isinstance(value, Enum):
        return _safe_value(value.value, seen)
    identity = id(value)
    if identity in seen:
        return {"__cycle__": type(value).__name__}
    seen.add(identity)
    try:
        if isinstance(value, Mapping):
            result = {}
            for key, item in value.items():
                key_text = str(key)
                result[_redact_text(key_text)] = _REDACTED if _secret_key(key_text, item) else _safe_value(item, seen)
            return result
        if is_dataclass(value) and not isinstance(value, type):
            return _safe_value({field.name: getattr(value, field.name) for field in fields(value)}, seen)
        if isinstance(value, (list, tuple, set, frozenset)):
            return [_safe_value(item, seen) for item in value]
        if isinstance(value, BaseException):
            return {"type": type(value).__name__, "message": _redact_text(str(value))}
        return {"__type__": type(value).__name__, "repr": _redact_text(repr(value))}
    except Exception as error:
        return {"__serialization_error__": type(error).__name__, "value_type": type(value).__name__}
    finally:
        seen.discard(identity)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_trace_value(value: Any) -> Any:
    """Return a JSON-safe, recursively redacted snapshot for trace attachments.

    Use this for associated evaluation metadata as well as recorded events so
    an unredacted error message cannot reintroduce a secret in an HTML export.
    """
    return _safe_value(value)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Nonstandard JSON constant: {value}")


class TraceRecorder:
    """Record one task/condition/trial; use a separate recorder for each run.

    ``path`` is opened exclusively so an explicit existing trace ID is never
    overwritten. Directory creation and initial file opening fail fast. Once
    initialized, I/O and serialization failures are reported through ``errors``.
    Full supplied content is retained, with recursive secret redaction and
    explicit markers for values that cannot be represented as JSON.
    """

    def __init__(self, directory: str | Path, *, task_id: str, condition: str = "default",
                 trial_id: int = 0, trace_id: str | None = None):
        self.trace_id = trace_id or uuid4().hex
        if not isinstance(self.trace_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.trace_id):
            raise ValueError("trace_id must be a safe, nonempty file identifier")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"{self.trace_id}.jsonl"
        self.manifest_path = self.directory / f"{self.trace_id}.manifest.json"
        self.task_id, self.condition, self.trial_id = task_id, condition, trial_id
        self.errors: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._sequence = 0
        self._written = 0
        self._dropped: list[int] = []
        self._spans: dict[str, TraceSpan] = {}
        self._closed = False
        self._started = time.monotonic_ns()
        self._created_at = _now()
        self._file = self.path.open("x", encoding="utf-8", newline="\n")
        self._emit("trace.start", name="agent evaluation", kind="trace")

    def _record_error(self, operation: str, error: BaseException, sequence: int | None = None) -> None:
        self.errors.append({"operation": operation, "sequence": sequence,
                            "type": type(error).__name__, "message": _safe_value(error).get("message", "logging failed")})

    def _write_manifest(self) -> None:
        open_spans = [span.span_id for span in self._spans.values() if not span._finished]
        document = {
            "schema_version": 1, "trace_id": self.trace_id, "task_id": self.task_id,
            "condition": self.condition, "trial_id": self.trial_id, "file": self.path.name,
            "created_at": self._created_at, "closed": self._closed,
            "status": "degraded" if self.errors else ("incomplete" if self._closed and open_spans else
                                                      "complete" if self._closed else "open"),
            "open_span_ids": open_spans,
            "intended_event_count": self._sequence, "written_event_count": self._written,
            "dropped_sequences": list(self._dropped), "errors": list(self.errors),
        }
        temporary = self.manifest_path.with_suffix(".json.tmp")
        try:
            temporary.write_text(json.dumps(_safe_value(document), ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(self.manifest_path)
        except Exception as error:
            self._record_error("manifest.write", error)

    def _emit(self, event: str, *, span_id: str | None = None, parent_span_id: str | None = None,
              name: str | None = None, kind: str | None = None, **payload: Any) -> None:
        with self._lock:
            if self._closed and event != "trace.end":
                self._record_error("event.after_close", RuntimeError("trace is already closed"))
                self._write_manifest()
                return
            self._sequence += 1
            record = {"schema_version": 1, "trace_id": self.trace_id, "task_id": self.task_id,
                      "condition": self.condition, "trial_id": self.trial_id,
                      "sequence": self._sequence, "event": event, "timestamp": _now(),
                      "monotonic_ns": time.monotonic_ns(), "span_id": span_id,
                      "parent_span_id": parent_span_id, "name": name, "kind": kind, **payload}
            try:
                line = json.dumps(_safe_value(record), ensure_ascii=False, allow_nan=False) + "\n"
                written = self._file.write(line)
                if written != len(line):
                    raise OSError("incomplete trace write")
                self._file.flush()
                self._written += 1
            except Exception as error:
                self._dropped.append(self._sequence)
                self._record_error("trace.write", error, self._sequence)
            self._write_manifest()

    def span(self, name: str, kind: str, *, parent_span_id: str | None = None,
             input: Any = None, metadata: Any = None) -> TraceSpan:
        with self._lock:
            if parent_span_id is None:
                current = _CURRENT_SPAN.get()
                if current is not None and current[0] is self:
                    parent_span_id = current[1]
            if parent_span_id is not None and parent_span_id not in self._spans:
                raise ValueError("parent_span_id does not belong to this recorder")
            span = TraceSpan(self, name, kind, parent_span_id, input, metadata)
            self._spans[span.span_id] = span
            self._emit("span.start", span_id=span.span_id, parent_span_id=parent_span_id,
                       name=name, kind=kind, input=input, metadata=metadata)
            return span

    def event(self, name: str, data: Any = None, *, span_id: str | None = None) -> None:
        with self._lock:
            if span_id is None:
                current = _CURRENT_SPAN.get()
                if current is not None and current[0] is self:
                    span_id = current[1]
            if span_id is not None and span_id not in self._spans:
                raise ValueError("span_id does not belong to this recorder")
            parent = self._spans[span_id].parent_span_id if span_id else None
            self._emit("event", span_id=span_id, parent_span_id=parent, name=name, kind="event", data=data)

    def close(self, status: str = "ok", metadata: Any = None) -> None:
        with self._lock:
            if self._closed:
                return
            # Preserve genuinely unfinished spans instead of inventing success.
            open_spans = [span.span_id for span in self._spans.values() if not span._finished]
            self._closed = True
            self._emit("trace.end", name="agent evaluation", kind="trace", status=status,
                       duration_ms=(time.monotonic_ns() - self._started) / 1_000_000,
                       open_span_ids=open_spans, metadata=metadata)
            try:
                self._file.close()
            except Exception as error:
                self._record_error("trace.close", error)
                self._write_manifest()


class TraceSpan:
    """A started span. ``finish`` and context-manager exit are idempotent."""

    def __init__(self, recorder: TraceRecorder, name: str, kind: str, parent_span_id: str | None,
                 input: Any, metadata: Any):
        self.recorder = recorder
        self.span_id = uuid4().hex
        self.parent_span_id = parent_span_id
        self.name, self.kind = name, kind
        self._started = time.monotonic_ns()
        self._finished = False
        self._context_token = None

    def __enter__(self) -> TraceSpan:
        if self._context_token is not None:
            raise RuntimeError("a span context cannot be entered twice")
        self._context_token = _CURRENT_SPAN.set((self.recorder, self.span_id))
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> bool:
        try:
            if exc_value is None:
                self.finish()
            else:
                exception = _safe_value(exc_value)
                try:
                    exception["traceback"] = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
                except Exception as error:
                    exception["traceback_error"] = type(error).__name__
                self.finish(status="error", metadata={"exception": exception})
        finally:
            if self._context_token is not None:
                _CURRENT_SPAN.reset(self._context_token)
                self._context_token = None
        return False

    def finish(self, output: Any = None, status: str = "ok", usage: Any = None, metadata: Any = None) -> None:
        with self.recorder._lock:
            if self._finished:
                return
            self._finished = True
            self.recorder._emit("span.end", span_id=self.span_id, parent_span_id=self.parent_span_id,
                                name=self.name, kind=self.kind, output=output, status=status, usage=usage,
                                metadata=metadata, duration_ms=(time.monotonic_ns() - self._started) / 1_000_000)

    def event(self, name: str, data: Any = None) -> None:
        self.recorder.event(name, data, span_id=self.span_id)


def read_trace(path: str | Path) -> dict[str, Any]:
    """Read a trace, returning all parseable records and explicit integrity issues.

    Interrupted trailing writes are recoverable. Corrupt middle records,
    sequence gaps, mismatched IDs, unfinished spans and missing inventory/file
    records always make ``complete`` false and are identified in diagnostics.
    """
    path = Path(path)
    result: dict[str, Any] = {"trace_id": None, "events": [], "spans": [],
                             "complete": False, "diagnostics": [], "manifest": None}
    diagnostics = result["diagnostics"]

    def issue(code: str, message: str, **details: Any) -> None:
        diagnostics.append({"code": code, "message": message, **details})

    manifest_path = path.with_suffix(".manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be an object")
        result["manifest"] = manifest
        if isinstance(manifest.get("trace_id"), str):
            result["trace_id"] = manifest["trace_id"]
        else:
            issue("invalid_manifest", "Inventory trace_id must be a string.")
        if manifest.get("schema_version") != 1:
            issue("invalid_manifest", "Inventory schema version is not 1.")
    except FileNotFoundError:
        issue("missing_manifest", "Trace inventory manifest is missing.")
    except (OSError, ValueError, UnicodeError) as error:
        issue("invalid_manifest", f"Cannot read trace inventory: {type(error).__name__}.")
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        issue("missing_trace", "Trace file listed in the inventory is missing.")
        return result
    except OSError as error:
        issue("unreadable_trace", f"Cannot read trace file: {type(error).__name__}.")
        return result

    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines, 1):
        try:
            record = json.loads(line, parse_constant=_reject_json_constant)
            if not isinstance(record, dict):
                raise ValueError("event must be an object")
        except (ValueError, UnicodeError):
            incomplete = index == len(lines) and not line.endswith(b"\n")
            issue("incomplete_tail" if incomplete else "corrupt_line", "Unparseable trace record.", line=index)
            continue
        if index == len(lines) and not line.endswith(b"\n"):
            issue("incomplete_tail", "Last record has no terminating newline.", line=index)
        if not isinstance(record.get("trace_id"), str):
            issue("invalid_trace_id", "Record trace_id must be a string.", line=index)
        elif result["trace_id"] is None:
            result["trace_id"] = record.get("trace_id")
        if record.get("trace_id") != result["trace_id"]:
            issue("trace_id_mismatch", "Record belongs to a different trace.", line=index)
        if record.get("schema_version") != 1:
            issue("unsupported_schema", "Record schema version is not 1.", line=index)
        if not isinstance(record.get("event"), str) or record.get("event") not in {
            "trace.start", "trace.end", "span.start", "span.end", "event"
        }:
            issue("unknown_event", "Record has an unsupported event type.", line=index)
        expected = len(result["events"]) + 1
        if record.get("sequence") != expected:
            issue("sequence_gap", "Trace event sequence is incomplete or out of order.", line=index,
                  expected=expected, actual=record.get("sequence"))
        result["events"].append(record)

    events = result["events"]
    if not events or events[0].get("event") != "trace.start":
        issue("missing_trace_start", "No initial trace.start record.")
    if not events or events[-1].get("event") != "trace.end":
        issue("missing_trace_end", "Trace has not been closed successfully.")
    if sum(event.get("event") == "trace.start" for event in events) > 1:
        issue("duplicate_trace_start", "Trace has more than one start record.")
    if sum(event.get("event") == "trace.end" for event in events) > 1:
        issue("duplicate_trace_end", "Trace has more than one end record.")
    spans: dict[str, dict[str, Any]] = {}
    for event in events:
        identifier = event.get("span_id")
        if event.get("event") == "span.start":
            if not isinstance(identifier, str):
                issue("invalid_span_id", "Span start has no string identifier.")
                continue
            if identifier in spans:
                issue("duplicate_span_start", "Span start is duplicated.", span_id=identifier)
                continue
            spans[identifier] = {**event, "id": identifier, "parent_id": event.get("parent_span_id"),
                                 "started_at": event.get("timestamp"), "start": event, "end": None,
                                 "status": "incomplete", "complete": False, "events": []}
        elif event.get("event") == "span.end":
            if not isinstance(identifier, str) or identifier not in spans:
                issue("orphan_span_end", "Span end has no preceding start.", span_id=identifier)
                continue
            span = spans[identifier]
            if span["end"] is not None:
                issue("duplicate_span_end", "Span end is duplicated.", span_id=identifier)
                continue
            if span["parent_span_id"] != event.get("parent_span_id"):
                issue("span_parent_mismatch", "Span parent changed between start and end.", span_id=identifier)
            span.update({"end": event, "ended_at": event.get("timestamp"), "complete": True,
                         "output": event.get("output"), "status": event.get("status"),
                         "usage": event.get("usage"), "duration_ms": event.get("duration_ms"),
                         "end_metadata": event.get("metadata")})
        elif event.get("event") == "event" and identifier:
            if isinstance(identifier, str) and identifier in spans:
                spans[identifier]["events"].append(event)
            else:
                issue("orphan_event", "Event has no preceding span start.", span_id=identifier)
    for span in spans.values():
        if not span["complete"]:
            issue("unfinished_span", "Span has no end record.", span_id=span["span_id"])
        if span["parent_id"] is not None and (not isinstance(span["parent_id"], str) or span["parent_id"] not in spans):
            issue("missing_parent_span", "Span references a missing parent.", span_id=span["span_id"])
    result["spans"] = list(spans.values())
    manifest = result["manifest"]
    if manifest is not None:
        if manifest.get("file") != path.name:
            issue("manifest_file_mismatch", "Inventory file name differs from this trace.")
        if manifest.get("closed") is not True:
            issue("open_manifest", "Inventory indicates an interrupted or open trace.")
        if manifest.get("intended_event_count") != len(events) or manifest.get("written_event_count") != len(events):
            issue("event_count_mismatch", "Inventory count differs from readable events.",
                  intended=manifest.get("intended_event_count"), written=manifest.get("written_event_count"), actual=len(events))
        if manifest.get("errors") or manifest.get("dropped_sequences") or manifest.get("status") == "degraded":
            issue("logging_errors", "Recorder reported logging failures.", errors=manifest.get("errors", []),
                  dropped_sequences=manifest.get("dropped_sequences", []))
    result["complete"] = not diagnostics
    return result
