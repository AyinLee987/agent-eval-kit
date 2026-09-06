"""Durable, paired experiments. No provider/network dependency in the core."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sqlite3
import subprocess
import sys
import uuid

from .execution import ExecutionCleanupError, declared_metrics, default_scorers, encode_outcome, execute
from .score_reporting import metric_summary, validate_tasks, write_json_report
from .types import AgentOutcome


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def fingerprint(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def load_tasks(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows and "question" in rows[0] and "dataset" in rows[0]:
            from .dataset_scoring import load_public_tasks
            return validate_tasks(load_public_tasks(path))
        return validate_tasks(rows)
    return validate_tasks(json.loads(path.read_text(encoding="utf-8")))


def _reject_secrets(value):
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = key.lower().replace("-", "_")
            if normalized in {"api_key", "apikey", "secret", "password", "authorization", "access_token", "auth_token"} or normalized.endswith("_api_key"):
                raise ValueError(f"Do not persist credentials in config ({key}); use environment variables")
            _reject_secrets(item)
    elif isinstance(value, list):
        for item in value:
            _reject_secrets(item)


def normalize_config(config, tasks):
    config = json.loads(_json(config))
    _reject_secrets(config)
    unknown = set(config) - {"trials", "seed", "timeout_seconds", "score_timeout_seconds", "conditions", "scorers", "source_roots", "metadata", "trace"}
    if unknown:
        raise ValueError(f"Unknown experiment config fields: {sorted(unknown)}")
    config.setdefault("trials", 3)
    config.setdefault("seed", 0)
    config.setdefault("timeout_seconds", 120)
    config.setdefault("score_timeout_seconds", 120)
    config.setdefault("scorers", default_scorers(tasks))
    config.setdefault("source_roots", [])
    config.setdefault("metadata", {})
    if "trace" in config:
        trace = config["trace"]
        if not isinstance(trace, dict) or set(trace) - {"enabled", "instrumentation"}:
            raise ValueError("trace requires enabled and optional instrumentation")
        if not isinstance(trace.get("enabled"), bool):
            raise ValueError("trace.enabled must be a boolean")
        if "instrumentation" in trace and (not isinstance(trace["instrumentation"], str) or ":" not in trace["instrumentation"]):
            raise ValueError("trace.instrumentation must be module:attribute")
    if isinstance(config["trials"], bool) or not isinstance(config["trials"], int) or config["trials"] < 1:
        raise ValueError("trials must be a positive integer")
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer")
    for name in ("timeout_seconds", "score_timeout_seconds"):
        v = config[name]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"{name} must be finite and positive")
    conditions = config.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("At least one condition is required")
    names = set()
    for condition in conditions:
        name = condition.get("id")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Condition ids must be nonempty and unique")
        names.add(name)
        for ref in ("agent", "outcome_adapter"):
            if not isinstance(condition.get(ref), str) or ":" not in condition[ref]:
                raise ValueError(f"Condition {name} needs {ref} as module:attribute")
        if not isinstance(condition.get("config", {}), dict):
            raise ValueError("Condition config must be an object of factory keyword arguments")
    if not isinstance(config["scorers"], list) or not config["scorers"]:
        raise ValueError("At least one scorer is required")
    for spec in config["scorers"]:
        if not isinstance(spec, dict) or not isinstance(spec.get("factory"), str) or ":" not in spec["factory"] or not isinstance(spec.get("config", {}), dict):
            raise ValueError("Scorers require factory module:attribute and optional config object")
        if "metric_names" in spec and (not isinstance(spec["metric_names"], list) or any(not isinstance(name, str) or not name for name in spec["metric_names"])):
            raise ValueError("Scorer metric_names must be a list of nonempty strings")
    config["source_roots"] = [str(Path(p).resolve(strict=True)) for p in config["source_roots"]]
    return config


def source_snapshot(roots):
    """Hash checkout code or installed package directories, plus explicit roots.

    A source checkout has both pyproject.toml and tests/. An installed module
    instead lives beneath site-packages: walking that parent would couple
    resume to every unrelated distribution in the environment. Only this
    distribution's existing package directories are implicit roots there.
    Git metadata is retained for every root; .env and raw diffs are not read.
    """
    base = Path(__file__).resolve().parents[1]
    if (base / "pyproject.toml").is_file() and (base / "tests").is_dir():
        implicit_roots = [base]
    else:
        implicit_roots = [base / name for name in ("agent_eval", "adapters", "benchmarks")
                          if (base / name).is_dir()]
    selected_roots = {str(path.resolve()) for path in implicit_roots}
    selected_roots.update(str(Path(root).resolve()) for root in roots)
    snapshots = []
    for root in sorted(selected_roots):
        path = Path(root)
        if not path.is_dir():
            raise ValueError(f"Source root must be a directory: {root}")
        files = {}
        excluded = {".git", ".venv", "venv", "node_modules", "__pycache__", ".cache", "results", "build", "dist", ".pytest-tmp"}
        for directory, subdirs, names in os.walk(path):
            subdirs[:] = sorted(d for d in subdirs if d not in excluded and not d.endswith(".egg-info"))
            for name in sorted(names):
                file = Path(directory) / name
                if file.suffix in {".py", ".toml"}:
                    files[file.relative_to(path).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
        def git(*args):
            try:
                result = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=10)
                return result.stdout.strip() if result.returncode == 0 else None
            except (OSError, subprocess.TimeoutExpired):
                return None
        status = git("status", "--porcelain", "--untracked-files=no")
        snapshots.append({"root": root, "sha256": fingerprint(files), "files": files,
                          "git_head": git("rev-parse", "HEAD"), "git_dirty": bool(status) if status is not None else None})
    return snapshots


@contextmanager
def _lock(directory):
    """OS lock releases on crash; a second process must not resume concurrently."""
    stream = open(Path(directory) / ".runner.lock", "a+b")
    try:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Experiment is already owned by another runner") from exc
        yield
    finally:
        stream.close()


@contextmanager
def _connect(directory):
    con = sqlite3.connect(Path(directory) / "runs.sqlite3")
    try:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=FULL")
        with con:
            yield con
    finally:
        con.close()


def measurement_protocol(config):
    # Adapter/model versions may differ in a comparison; scoring definitions
    # and the runner's time budgets must match. Custom scorer dependencies must
    # be versioned in their spec config and source_roots by the caller.
    root = Path(__file__).resolve().parent
    scorer_code = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                   ("scoring.py", "dataset_scoring.py", "score_reporting.py", "judge.py", "types.py")}
    protocol = {"timeout_seconds": config["timeout_seconds"], "score_timeout_seconds": config["score_timeout_seconds"],
                "scorers": config["scorers"], "builtin_scorer_code": scorer_code, "scoring_version": 2}
    if config.get("trace", {}).get("enabled"):
        protocol["trace"] = config["trace"]
    return protocol


def create_experiment(directory, tasks, config):
    tasks = validate_tasks(tasks)
    if not tasks:
        raise ValueError("An experiment needs at least one task")
    config = normalize_config(config, tasks)
    dataset_hash = fingerprint(tasks)
    task_meta = [{"id": t["id"], "group_id": t.get("group_id", t.get("family_id", t["id"])), "fingerprint": fingerprint(t)} for t in tasks]
    if any(not isinstance(t["group_id"], str) or not t["group_id"] for t in task_meta):
        raise ValueError("Task group_id/family_id must be a nonempty string")
    sources = source_snapshot(config["source_roots"])
    blocks = [(t["id"], trial) for trial in range(config["trials"]) for t in tasks]
    rng = random.Random(config["seed"])
    rng.shuffle(blocks)
    schedule = []
    for task_id, trial in blocks:
        conditions = [c["id"] for c in config["conditions"]]
        rng.shuffle(conditions)
        schedule.extend({"condition": c, "task_id": task_id, "trial_id": trial} for c in conditions)
    manifest = {"schema_version": 1, "scoring_version": 2, "experiment_id": str(uuid.uuid4()), "created_at": _now(),
                "dataset_fingerprint": dataset_hash, "config": config, "config_fingerprint": fingerprint(config),
                "tasks": task_meta, "schedule": schedule, "sources": sources,
                "measurement_protocol": measurement_protocol(config),
                "runtime": {"python": sys.version, "platform": platform.platform(),
                            "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions() if d.metadata["Name"]))}}
    manifest["manifest_fingerprint"] = fingerprint(manifest)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json_report(directory / "tasks.json", tasks)
    write_json_report(directory / "manifest.json", manifest)
    with _connect(directory) as con:
        con.execute("CREATE TABLE trials (position INTEGER PRIMARY KEY, condition TEXT NOT NULL, task_id TEXT NOT NULL, trial_id INTEGER NOT NULL, state TEXT NOT NULL, record TEXT, worker_pid INTEGER, UNIQUE(condition, task_id, trial_id))")
        con.executemany("INSERT INTO trials(position,condition,task_id,trial_id,state) VALUES (?,?,?,?, 'pending')",
                        [(i, r["condition"], r["task_id"], r["trial_id"]) for i, r in enumerate(schedule)])
    return manifest


def _load(directory, *, verify_sources=False):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest.pop("manifest_fingerprint")
    if fingerprint(manifest) != expected:
        raise ValueError("Experiment manifest was modified")
    manifest["manifest_fingerprint"] = expected
    tasks = load_tasks(directory / "tasks.json")
    if fingerprint(tasks) != manifest["dataset_fingerprint"]:
        raise ValueError("Saved task content changed; create a new experiment")
    if verify_sources:
        current = source_snapshot(manifest["config"]["source_roots"])
        if [(s["root"], s["sha256"]) for s in current] != [(s["root"], s["sha256"]) for s in manifest["sources"]]:
            raise ValueError("Source code changed; resume refused to avoid mixing implementations")
    return manifest, tasks


def read_records(directory):
    _load(directory)
    with _connect(directory) as con:
        return [json.loads(row[0]) for row in con.execute("SELECT record FROM trials WHERE state='completed' ORDER BY position")]


def _alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() != 87  # access denied is conservatively alive
        try:
            code = wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _save(con, position, state, record, pid=None):
    with con:
        con.execute("UPDATE trials SET state=?,record=?,worker_pid=? WHERE position=?", (state, _json(record) if record is not None else None, pid, position))


def _score(task, record, config, *, on_start=None):
    result = execute({"kind": "score", "task": task, "outcome": record["outcome"],
                      "scorers": config["scorers"], "unavailable": "execution_error" if record["execution_error"] else None},
                     timeout_seconds=config["score_timeout_seconds"], on_start=on_start)
    record["scoring_elapsed_seconds"] = result["elapsed_seconds"]
    if "worker_error" in result:
        names = declared_metrics(config["scorers"])
        state = "execution_error" if record["execution_error"] else "failed"
        record.update(scores={n: None for n in names}, metric_statuses={n: state for n in names},
                      scorer_errors=[result["worker_error"]], scoring_error=result["worker_error"])
    else:
        record["scoring_error"] = None
        record.update({key: result[key] for key in ("scores", "metric_statuses", "scorer_errors")})
    record["status"] = "completed"
    record["completed_at"] = _now()
    return record


def run_experiment(directory, *, recover_interrupted=False, max_runs=None, progress=None):
    """Resume pending work. A completed failure is a result, never silently retried.

    Clean interrupts are retriable after local workers are reaped. Following an
    abrupt crash, explicit recover_interrupted acknowledges possible external
    side effects; a still-live recorded worker always prevents recovery.
    """
    if max_runs is not None and (isinstance(max_runs, bool) or not isinstance(max_runs, int) or max_runs < 0):
        raise ValueError("max_runs must be a nonnegative integer")
    with _lock(directory):
        manifest, tasks = _load(directory, verify_sources=True)
        config = manifest["config"]
        tasks = {t["id"]: t for t in tasks}
        metadata = {t["id"]: t for t in manifest["tasks"]}
        conditions = {c["id"]: c for c in config["conditions"]}
        with _connect(directory) as con:
            abandoned = list(con.execute("SELECT * FROM trials WHERE state IN ('running','scoring')"))
            if abandoned and not recover_interrupted:
                raise RuntimeError("An abrupt interruption left an uncertain trial. Verify external side effects, then use recover_interrupted=True / --recover-interrupted")
            for row in abandoned:
                if _alive(row["worker_pid"]):
                    raise RuntimeError(f"Recorded worker {row['worker_pid']} is still alive; recovery refused")
            for row in abandoned:
                _save(con, row["position"], "executed" if row["state"] == "scoring" else "interrupted", json.loads(row["record"]) if row["record"] else None)
            processed = 0
            for row in list(con.execute("SELECT * FROM trials WHERE state != 'completed' ORDER BY position")):
                if max_runs is not None and processed >= max_runs:
                    break
                task, meta = tasks[row["task_id"]], metadata[row["task_id"]]
                record = json.loads(row["record"]) if row["record"] else None
                phase = "executed" if row["state"] == "executed" else "interrupted"
                try:
                    if phase != "executed":
                        previous_traces = list((record or {}).get("trace_attempts", []))
                        if (record or {}).get("trace"):
                            previous_traces.append(record["trace"])
                        record = {"schema_version": 1, "scoring_version": 2, "experiment_id": manifest["experiment_id"],
                                  "condition": row["condition"], "task_id": row["task_id"], "trial_id": row["trial_id"],
                                  "group_id": meta["group_id"], "task_fingerprint": meta["fingerprint"],
                                  "dataset_fingerprint": manifest["dataset_fingerprint"], "started_at": _now(), "status": "running"}
                        request = {"kind": "agent", "condition": conditions[row["condition"]], "prompt": task["prompt"]}
                        if config.get("trace", {}).get("enabled"):
                            trace_id = uuid.uuid4().hex
                            trace_directory = (Path(directory) / "traces").resolve()
                            request["trace"] = {"directory": str(trace_directory), "trace_id": trace_id,
                                                "task_id": row["task_id"], "condition": row["condition"],
                                                "trial_id": row["trial_id"],
                                                "instrumentation": config["trace"].get("instrumentation")}
                            record["trace"] = {"trace_id": trace_id, "path": str(trace_directory / (trace_id + ".jsonl"))}
                            record["trace_attempts"] = previous_traces
                        _save(con, row["position"], "running", record)
                        execution = execute(request,
                                            timeout_seconds=config["timeout_seconds"],
                                            on_start=lambda pid: _save(con, row["position"], "running", record, pid))
                        record["execution_error"] = execution.get("worker_error") or execution.get("execution_error")
                        record["tokens_observed"] = record["execution_error"] is None
                        record["outcome"] = execution.get("outcome") or encode_outcome(AgentOutcome("", False, "execution_error", 0, 0))
                        if "trace" in record:
                            from .tracing import read_trace
                            try:
                                observed = read_trace(record["trace"]["path"])
                                record["trace"].update(complete=observed["complete"], diagnostics=observed["diagnostics"])
                            except Exception as exc:
                                record["trace"].update(complete=False, diagnostics=[{"code": "trace_unavailable", "type": type(exc).__name__}])
                            record["trace"].update(execution.get("trace", {}))
                            record["outcome"]["metadata"].update(trace_id=record["trace"]["trace_id"], trace_path=record["trace"]["path"])
                        record["elapsed_seconds"] = execution["elapsed_seconds"]
                        record["outcome"]["elapsed_seconds"] = execution["elapsed_seconds"]
                        record["worker_elapsed_seconds"] = execution.get("worker_elapsed_seconds")
                        record["status"] = "executed"
                        _save(con, row["position"], "executed", record)
                        phase = "executed"
                    _save(con, row["position"], "scoring", record)
                    record = _score(task, record, config, on_start=lambda pid: _save(con, row["position"], "scoring", record, pid))
                    _save(con, row["position"], "completed", record)
                    phase = "completed"
                    processed += 1
                    if progress:
                        progress(record)
                except ExecutionCleanupError:
                    # Keep the PID and uncertain running/scoring claim for reconciliation.
                    raise
                except BaseException:
                    # execute's finally has reaped its child before control reaches here.
                    _save(con, row["position"], phase, record)
                    raise
        return export_report(directory)


def export_report(directory):
    manifest, _ = _load(directory)
    with _connect(directory) as con:
        rows = list(con.execute("SELECT * FROM trials ORDER BY position"))
    records = [json.loads(row["record"]) for row in rows if row["state"] == "completed"]
    names = set(declared_metrics(manifest["config"]["scorers"]))
    names.update(metric_summary((r.get("scores", {}), r.get("metric_statuses", {})) for r in records))
    conditions = {}
    for spec in manifest["config"]["conditions"]:
        selected = [r for r in records if r["condition"] == spec["id"]]
        planned = sum(r["condition"] == spec["id"] for r in rows)
        metric_rows = []
        for record in selected:
            statuses = {name: "missing" for name in names if name not in record.get("scores", {})}
            statuses.update(record.get("metric_statuses", {}))
            metric_rows.append((record.get("scores", {}), statuses))
        metric_rows.extend(({}, {name: "missing" for name in names}) for _ in range(planned - len(selected)))
        metrics = metric_summary(metric_rows)
        conditions[spec["id"]] = {"planned": planned, "completed": len(selected),
                                  "execution_errors": sum(bool(r["execution_error"]) for r in selected),
                                  "agent_failed": sum(not r["outcome"]["success"] and not r["execution_error"] for r in selected),
                                  "scorer_errors": sum(bool(r["scorer_errors"]) for r in selected), "metrics": metrics,
                                  "observed_tokens": sum(r["outcome"]["tokens"] for r in selected),
                                  "cost_unknown_runs": sum(bool(r["execution_error"]) for r in selected),
                                  "elapsed_seconds": sum(r["elapsed_seconds"] for r in selected)}
    report = {"schema_version": 1, "scoring_version": 2, "experiment_id": manifest["experiment_id"],
              "dataset_fingerprint": manifest["dataset_fingerprint"], "manifest_fingerprint": manifest["manifest_fingerprint"],
              "measurement_protocol": manifest["measurement_protocol"],
              "planned_keys": planned_keys(manifest),
              "states": dict(Counter(row["state"] for row in rows)), "conditions": conditions, "records": records}
    write_json_report(Path(directory) / "report.json", report)
    return report


def planned_keys(manifest):
    return [{"task_id": task["id"], "trial_id": trial, "group_id": task["group_id"],
             "task_fingerprint": task["fingerprint"], "dataset_fingerprint": manifest["dataset_fingerprint"]}
            for task in manifest["tasks"] for trial in range(manifest["config"]["trials"])]


def rescore_experiment(directory, output, *, scorers=None, score_timeout_seconds=120):
    """Derive a new report from stored trajectories, without creating any agent.

    Rule scorers are offline. A caller-supplied LLM scorer may incur API cost.
    The original experiment is immutable, and an existing output is refused.
    """
    output = Path(output).resolve()
    if output.exists() or Path(directory).resolve() in output.parents:
        raise ValueError("Rescoring output must be a new file outside the original experiment")
    manifest, tasks = _load(directory)
    task_map = {t["id"]: t for t in tasks}
    config = {**manifest["config"], "scorers": scorers if scorers is not None else default_scorers(tasks), "score_timeout_seconds": score_timeout_seconds}
    config = normalize_config(config, tasks)
    # Lock only while taking a stable snapshot; never hold it across judge calls.
    with _lock(directory):
        records = read_records(directory)
    for record in records:
        _score(task_map[record["task_id"]], record, config)
    # Recover the known metric universe without reporting a pooled A/B mean.
    metric_names = set(metric_summary((r["scores"], r["metric_statuses"]) for r in records))
    metric_names.update(declared_metrics(config["scorers"]))
    conditions = {}
    for spec in manifest["config"]["conditions"]:
        selected = [r for r in records if r["condition"] == spec["id"]]
        planned = sum(item["condition"] == spec["id"] for item in manifest["schedule"])
        metric_rows = []
        for record in selected:
            scores, statuses = record["scores"], dict(record["metric_statuses"])
            for name in metric_names:
                if name not in scores:
                    statuses.setdefault(name, "missing")
            metric_rows.append((scores, statuses))
        metric_rows.extend(({}, {name: "missing" for name in metric_names}) for _ in range(planned - len(selected)))
        conditions[spec["id"]] = {
            "planned": planned, "completed": len(selected), "missing_records": planned - len(selected),
            "execution_errors": sum(bool(r["execution_error"]) for r in selected),
            "agent_failed": sum(not r["outcome"]["success"] and not r["execution_error"] for r in selected),
            "scorer_errors": sum(bool(r["scorer_errors"]) for r in selected),
            "metrics": metric_summary(metric_rows),
            "observed_tokens": sum(r["outcome"]["tokens"] for r in selected),
            "cost_unknown_runs": sum(bool(r["execution_error"]) for r in selected),
            "elapsed_seconds": sum(r["elapsed_seconds"] for r in selected),
        }
    payload = {"schema_version": 1, "scoring_version": 2, "source_experiment_id": manifest["experiment_id"],
               "dataset_fingerprint": manifest["dataset_fingerprint"], "source_manifest_fingerprint": manifest["manifest_fingerprint"],
               "rescored_at": _now(), "scorers": config["scorers"],
               "scoring_sources": source_snapshot(config["source_roots"]),
               "measurement_protocol": measurement_protocol(config),
               "planned_keys": planned_keys(manifest), "records": records,
               "conditions": conditions}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_report(output, payload)
    return payload
