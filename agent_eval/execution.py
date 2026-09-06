"""Process-isolated execution with bounded time and descendant cleanup.

This is lifecycle isolation, not a security sandbox. Only run trusted adapters.
The parent opens the start gate only after it has installed containment and
persisted the worker PID. Large results use a file so a full pipe cannot defeat
the timeout. Adapter ``raw`` objects are intentionally not persisted.
"""
from __future__ import annotations

import ctypes
import importlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict

from .types import AgentOutcome, TrajectoryStep
from .score_reporting import error_record, score_outcome


class ExecutionCleanupError(RuntimeError):
    """A worker may still be alive; its persisted claim must remain uncertain."""


def resolve(reference):
    module, sep, attr = reference.partition(":")
    if not sep or not module or not attr:
        raise ValueError(f"Expected module:attribute, got {reference!r}")
    return getattr(importlib.import_module(module), attr)


def encode_outcome(outcome):
    if not isinstance(outcome, AgentOutcome):
        raise TypeError("Adapter must return AgentOutcome")
    if not isinstance(outcome.answer, str) or not isinstance(outcome.success, bool):
        raise TypeError("Outcome answer/success must be str/bool")
    for name in ("steps", "tokens"):
        value = getattr(outcome, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Outcome {name} must be a nonnegative integer")
    return {"answer": outcome.answer, "success": outcome.success,
            "stop_reason": outcome.stop_reason, "steps": outcome.steps,
            "tokens": outcome.tokens,
            "trajectory": [asdict(step) for step in outcome.trajectory],
            "metadata": dict(outcome.metadata)}


def decode_outcome(data):
    return AgentOutcome(data["answer"], data["success"], data["stop_reason"],
                        data["steps"], data["tokens"],
                        [TrajectoryStep.from_dict(s) for s in data.get("trajectory", [])],
                        metadata=data.get("metadata", {}))


def build_scorers(specs):
    return [resolve(spec["factory"])(**spec.get("config", {})) for spec in specs]


def default_scorers(tasks):
    specs = [{"factory": f"agent_eval.scoring:{name}"} for name in
             ("RuleScorer", "ToolUsageScorer", "TrajectoryScorer")]
    if any("public_case" in task for task in tasks):
        specs.append({"factory": "agent_eval.dataset_scoring:PublicDatasetScorer"})
    return specs


def declared_metrics(specs):
    """Read explicit specs or built-in class declarations without custom imports."""
    names = []
    for spec in specs:
        declared = spec.get("metric_names")
        if declared is None and spec["factory"].startswith(("agent_eval.scoring:", "agent_eval.dataset_scoring:")):
            candidate = getattr(resolve(spec["factory"]), "metric_names", ())
            declared = candidate if isinstance(candidate, (tuple, list)) else ()
        for name in declared or ():
            if name not in names:
                names.append(name)
    return names


class _WindowsJob:
    """Kill all assigned processes when the owning parent handle closes."""
    def __init__(self, pid):
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                        ("flags", w.DWORD), ("min_ws", ctypes.c_size_t),
                        ("max_ws", ctypes.c_size_t), ("active", w.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]
        class Limits(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", IO), ("process_mem", ctypes.c_size_t),
                        ("job_mem", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, args, result in (
            ("CreateJobObjectW", [ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
            ("SetInformationJobObject", [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL),
            ("OpenProcess", [w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            ("AssignProcessToJobObject", [w.HANDLE, w.HANDLE], w.BOOL),
            ("CloseHandle", [w.HANDLE], w.BOOL),
        ):
            fn = getattr(self.kernel, name)
            fn.argtypes, fn.restype = args, result
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            limits = Limits()
            limits.basic.flags = 0x2000
            if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise ctypes.WinError(ctypes.get_last_error())
            process = self.kernel.OpenProcess(0x0100 | 0x0001, False, pid)
            if not process:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                if not self.kernel.AssignProcessToJobObject(self.handle, process):
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                self.kernel.CloseHandle(process)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def _worker(request, gate, path, max_bytes):
    if os.name != "nt":
        os.setsid()
    # A parent that fails before opening the gate must never start agent work.
    if not gate.wait(30):
        return
    stage = "build"
    started = time.monotonic()
    recorder = None
    trace_setup_error = None
    if request["kind"] == "agent" and request.get("trace"):
        try:
            from .tracing import TraceRecorder
            spec = request["trace"]
            recorder = TraceRecorder(spec["directory"], task_id=spec["task_id"],
                                     condition=spec["condition"], trial_id=spec["trial_id"], trace_id=spec["trace_id"])
        except Exception as exc:
            trace_setup_error = {"code": "trace_setup_failed", "type": type(exc).__name__}
    try:
        if request["kind"] == "agent":
            scope = recorder.span("evaluation.run", "evaluation", input={"prompt": request["prompt"]}) if recorder else nullcontext()
            with scope as span:
                condition = request["condition"]
                agent = resolve(condition["agent"])(**condition.get("config", {}))
                instrumentation = request.get("trace", {}).get("instrumentation")
                if recorder and instrumentation:
                    agent = resolve(instrumentation)(agent, recorder)
                stage = "run"
                raw = agent.run(request["prompt"])
                stage = "adapt"
                outcome = encode_outcome(resolve(condition["outcome_adapter"])(raw))
                provider_error = outcome.get("metadata", {}).get("execution_error")
                if provider_error is not None and (not isinstance(provider_error, dict) or
                        any(not isinstance(provider_error.get(key), str) for key in ("stage", "type", "message"))):
                    raise ValueError("Outcome execution_error metadata must declare stage/type/message")
                payload = {"outcome": outcome, "execution_error": provider_error}
                if recorder:
                    outcome["metadata"].update(trace_id=recorder.trace_id, trace_path=str(recorder.path))
                    span.finish(output=outcome, status="ok" if outcome["success"] and not provider_error else "error")
        else:
            stage = "score"
            scores, statuses, errors = score_outcome(
                build_scorers(request["scorers"]), request["task"],
                decode_outcome(request["outcome"]), unavailable=request.get("unavailable"))
            payload = {"scores": scores, "metric_statuses": statuses, "scorer_errors": errors}
        if len(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")) > max_bytes:
            raise ValueError(f"Normalized result exceeds {max_bytes} bytes")
    except BaseException as exc:
        payload = {"worker_error": error_record(stage, exc)}
    finally:
        if recorder:
            failed = payload.get("worker_error") or payload.get("execution_error") or not payload.get("outcome", {}).get("success", False)
            recorder.close(status="error" if failed else "ok")
    if recorder:
        payload["trace"] = {"logging_errors": list(recorder.errors)}
    elif trace_setup_error:
        payload["trace"] = {"complete": False, "diagnostics": [trace_setup_error]}
    payload["worker_elapsed_seconds"] = time.monotonic() - started
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(encoded) > max_bytes:
        encoded = json.dumps({"worker_error": error_record(stage, ValueError(f"Normalized result exceeds {max_bytes} bytes"))}, ensure_ascii=False).encode("utf-8")
    Path(path).write_bytes(encoded)


def execute(request, *, timeout_seconds=120, on_start=None, max_result_bytes=16 * 1024 * 1024):
    """Return a JSON result or worker_error; reap the worker before returning.

    Timeout covers child startup, factory, run, and adaptation/scoring. Cleanup
    latency is included in elapsed_seconds. External remote requests cannot be
    rolled back by terminating a local process.
    """
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    context = multiprocessing.get_context("spawn")
    gate = context.Event()
    with tempfile.TemporaryDirectory(prefix="agent-eval-worker-") as temp:
        path = str(Path(temp) / "result.json")
        process = context.Process(target=_worker, args=(request, gate, path, max_result_bytes))
        job = None
        started = time.monotonic()
        did_start = False
        try:
            process.start()
            did_start = True
            if os.name == "nt":
                job = _WindowsJob(process.pid)
            if on_start:
                on_start(process.pid)
            gate.set()
            process.join(max(0, timeout_seconds - (time.monotonic() - started)))
            if process.is_alive():
                result = {"worker_error": {"stage": request["kind"], "type": "TimeoutError", "message": f"Exceeded {timeout_seconds}s process deadline"}}
            elif not Path(path).is_file():
                result = {"worker_error": {"stage": request["kind"], "type": "WorkerExit", "message": f"Worker exited {process.exitcode} without a result"}}
            elif Path(path).stat().st_size > max_result_bytes:
                result = {"worker_error": {"stage": request["kind"], "type": "ResultTooLarge", "message": "Worker result exceeded configured limit"}}
            else:
                try:
                    result = json.loads(Path(path).read_text(encoding="utf-8"))
                    if not isinstance(result, dict):
                        raise ValueError("Worker result must be an object")
                    if "worker_error" not in result:
                        if request["kind"] == "agent":
                            encode_outcome(decode_outcome(result["outcome"]))
                        elif any(not isinstance(result.get(key), kind) for key, kind in
                                 (("scores", dict), ("metric_statuses", dict), ("scorer_errors", list))):
                            raise ValueError("Malformed scoring result")
                except (ValueError, KeyError, TypeError, UnicodeError, OSError) as exc:
                    result = {"worker_error": error_record("result", exc)}
        finally:
            if did_start:
                try:
                    if job:
                        job.close()
                    elif os.name != "nt":
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    if process.is_alive():
                        process.kill()
                    process.join(5)
                    if process.is_alive():
                        raise RuntimeError("Worker could not be reaped")
                    process.close()
                except BaseException as exc:
                    raise ExecutionCleanupError("Worker cleanup could not be verified; do not retry this trial") from exc
        result["elapsed_seconds"] = time.monotonic() - started
        return result
