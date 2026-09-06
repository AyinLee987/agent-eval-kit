"""Read-only, coverage-aware summaries of a sharded public development run.

The root run_manifest.json declares dataset names, expected_ids and relative
shard paths. Child experiment manifests and committed SQLite rows are the
source of truth; report.json may be stale and is deliberately never read.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from agent_eval.dataset_scoring import METRIC_NAMES
from agent_eval.experiments import fingerprint
from agent_eval.score_reporting import metric_summary, write_json_report
from agent_eval.stats import wilson_ci


_STATES = {"pending", "running", "executed", "scoring", "interrupted", "completed"}
_BINARY = {"public_format_valid", "public_numeric_exact", "public_answer_em",
           "public_evidence_em", "public_joint_em", "run_completed"}
_PRIMARY = {"gsm8k": "public_numeric_exact", "hotpotqa": "public_answer_em"}


def _read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {value}")))


def _integer(value, label, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _seconds(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return value


def _instant(value, label):
    if not isinstance(value, str):
        raise ValueError(f"{label} requires an ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid {label}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def _child_path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Shard paths must be nonempty relative paths")
    child = (root / relative).resolve()
    if child == root or root not in child.parents:
        raise ValueError(f"Shard path escapes run directory: {relative}")
    return child


def _snapshot(directory):
    manifest = _read_json(directory / "manifest.json")
    body = {k: v for k, v in manifest.items() if k != "manifest_fingerprint"}
    if fingerprint(body) != manifest.get("manifest_fingerprint"):
        raise ValueError(f"Experiment manifest fingerprint mismatch: {directory}")
    if manifest.get("schema_version") != 1 or manifest.get("scoring_version") != 2:
        raise ValueError(f"Unsupported experiment/scoring version: {directory}")
    tasks = _read_json(directory / "tasks.json")
    if fingerprint(tasks) != manifest.get("dataset_fingerprint"):
        raise ValueError(f"Task dataset fingerprint mismatch: {directory}")
    if fingerprint(manifest["config"]) != manifest.get("config_fingerprint"):
        raise ValueError(f"Config fingerprint mismatch: {directory}")
    database = directory / "runs.sqlite3"
    # mode=ro refuses to create a database and does not change journal settings.
    con = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=10)
    try:
        con.row_factory = sqlite3.Row
        con.execute("BEGIN")
        rows = [dict(row) for row in con.execute("SELECT * FROM trials ORDER BY position")]
    finally:
        con.close()
    return manifest, tasks, rows


def _usage(records, planned):
    totals = dict.fromkeys(("prompt_tokens", "completion_tokens", "total_tokens",
                           "prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "reasoning_tokens"), 0)
    field_counts = dict.fromkeys(totals, 0)
    call_counts = dict.fromkeys(totals, 0)
    observed = 0
    token_count = 0
    total_calls = 0
    unknown_ledger_runs = 0
    for record in records:
        outcome = record["outcome"]
        token_count += _integer(outcome.get("tokens", 0), "outcome.tokens")
        if record.get("tokens_observed") is True and outcome.get("metadata", {}).get("usage_complete") is not False:
            observed += 1
        calls = outcome.get("metadata", {}).get("api_calls")
        if calls is None:
            unknown_ledger_runs += 1
            continue
        if not isinstance(calls, list):
            raise ValueError("outcome.metadata.api_calls must be a list")
        total_calls += len(calls)
        complete = {key: bool(calls) for key in totals}
        for call in calls:
            if not isinstance(call, dict):
                raise ValueError("Each api_calls entry must be an object")
            usage = call.get("usage", call)
            if usage is None:
                usage = {}
            if not isinstance(usage, dict):
                raise ValueError("API call usage must be an object or null")
            for field in totals:
                if usage.get(field) is not None:
                    totals[field] += _integer(usage[field], f"api_usage.{field}")
                    call_counts[field] += 1
                else:
                    complete[field] = False
        for key in totals:
            field_counts[key] += int(complete[key])
    return {"observed_tokens": token_count, "tokens_observed_runs": observed,
            "cost_unknown_runs": planned - observed,
            "token_observation_coverage": observed / planned if planned else None,
            "api_usage": {**{key: value if call_counts[key] else None for key, value in totals.items()},
                          "recorded_calls": total_calls,
                          "unknown_ledger_runs": unknown_ledger_runs + planned - len(records),
                          "observed_calls_by_field": call_counts,
                          "unknown_calls_by_field": {key: total_calls - count for key, count in call_counts.items()},
                          "observed_runs_by_field": field_counts,
                          "unknown_runs_by_field": {key: planned - count for key, count in field_counts.items()}}}


def _dataset_summary(name, expected_keys, planned_keys, records, states, tasks, trials):
    completed = {(record["task_id"], record["trial_id"]) for record in records}
    missing = sorted(expected_keys - completed)
    metric_names = ({"public_format_valid", "public_numeric_exact"} if name == "gsm8k"
                    else set(METRIC_NAMES) - {"public_numeric_exact"})
    metric_names.update(key for record in records for key in record.get("metric_statuses", {}))
    rows = []
    for record in records:
        scores = record.get("scores", {})
        status = {key: "missing" for key in metric_names if key not in scores}
        status.update(record.get("metric_statuses", {}))
        rows.append((scores, status))
    rows.extend(({}, {key: "missing" for key in metric_names}) for _ in missing)
    metrics = metric_summary(rows)
    for key, metric in metrics.items():
        metric["wilson_95"] = None
        if key in _BINARY and metric["valid"] and trials == 1:
            values = [float(score[key]) for score, status in rows if status.get(key) == "valid"]
            if any(value not in (0.0, 1.0) for value in values):
                raise ValueError(f"Binary metric {key} contains a nonbinary value")
            metric["wilson_95"] = asdict(wilson_ci(int(sum(values)), len(values)))
    errors, incorrect, api_errors = [], [], []
    failed = 0
    primary = _PRIMARY[name]
    for record in records:
        identity = {"task_id": record["task_id"], "trial_id": record["trial_id"], "shard": record["shard"]}
        execution_error = record.get("execution_error")
        scorer_errors = record.get("scorer_errors", [])
        agent_failed = record["outcome"].get("success") is not True
        if execution_error or scorer_errors or agent_failed:
            failed += 1
            errors.append({**identity, "execution_error": execution_error, "scorer_errors": scorer_errors,
                           "agent_failed": agent_failed, "stop_reason": record["outcome"].get("stop_reason")})
        if record.get("metric_statuses", {}).get(primary) == "valid" and record["scores"][primary] == 0:
            incorrect.append(identity)
        for index, call in enumerate(record["outcome"].get("metadata", {}).get("api_calls", [])):
            if call.get("error"):
                api_errors.append({**identity, "call_index": index, "error": call["error"],
                                   "finish_reason": call.get("finish_reason"),
                                   "elapsed_seconds": call.get("elapsed_seconds")})
    groups = Counter(task.get("group_id", task["id"]) for task in tasks.values())
    return {"planned": len(expected_keys), "planned_in_shards": len(planned_keys),
            "completed": len(records), "failed": failed,
            "execution_errors": sum(bool(r.get("execution_error")) for r in records),
            "agent_failed": sum(r["outcome"].get("success") is not True and not r.get("execution_error") for r in records),
            "scorer_error_runs": sum(any(error.get("type") != "UnavailableOutcome" for error in r.get("scorer_errors", [])) for r in records),
            "unavailable_scoring_runs": sum(any(error.get("type") == "UnavailableOutcome" for error in r.get("scorer_errors", [])) for r in records),
            "completion_coverage": len(records) / len(expected_keys) if expected_keys else None,
            "partial": bool(missing), "states": dict(sorted(states.items())),
            "missing_expected": [{"task_id": task, "trial_id": trial} for task, trial in missing],
            "unplanned_expected": [{"task_id": task, "trial_id": trial} for task, trial in sorted(expected_keys - planned_keys)],
            "metrics": metrics, "primary_metric": primary, "errors": errors,
            "api_call_errors": api_errors, "incorrect_answers": incorrect,
            "unique_groups": len(groups), "groups_with_multiple_tasks": sum(n > 1 for n in groups.values()),
            "elapsed_seconds_sum": math.fsum(_seconds(r["elapsed_seconds"], "elapsed_seconds") for r in records),
            **_usage(records, len(expected_keys))}


def summarize_run(root) -> dict:
    """Summarize one committed snapshot per shard without acquiring runner locks.

    Partial runs are valid snapshots. Unexpected, duplicate or tampered records
    are errors, rather than silently changing the denominator. All child shards
    must share one measurement protocol, condition specification and source hash.
    """
    root = Path(root).resolve(strict=True)
    run = _read_json(root / "run_manifest.json")
    if run.get("schema_version") != 1 or run.get("split") != "dev":
        raise ValueError("Run manifest requires schema_version=1 and split='dev'")
    if run.get("scoring_version", 2) != 2:
        raise ValueError("Run scoring_version must be 2")
    trials = _integer(run.get("trials"), "trials", minimum=1)
    if not isinstance(run.get("condition"), str) or not run["condition"]:
        raise ValueError("Run condition must be a nonempty string")
    if not isinstance(run.get("model"), str) or not run["model"]:
        raise ValueError("Run model must be a nonempty string")
    dataset_specs = run.get("datasets")
    if not isinstance(dataset_specs, list) or not dataset_specs:
        raise ValueError("Run datasets must be a nonempty list")
    now = datetime.now(timezone.utc)
    started = _instant(run.get("started_at"), "started_at")
    finished = _instant(run["finished_at"], "finished_at") if run.get("finished_at") else None
    wall_seconds = ((finished or now) - started).total_seconds()
    if wall_seconds < 0:
        raise ValueError("Run timestamps are reversed")
    summaries, snapshots, absent_shards = {}, [], []
    seen_paths, protocol_hash, condition_hash, source_hash = set(), None, None, None
    protocol = None
    for spec in dataset_specs:
        name = spec.get("dataset")
        if name not in _PRIMARY or name in summaries:
            raise ValueError(f"Unsupported or duplicate dataset: {name}")
        ids = spec.get("expected_ids")
        if (not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError(f"{name}: expected_ids must be nonempty and unique")
        expected = {(task, trial) for task in ids for trial in range(trials)}
        shards = spec.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"{name}: shards must be a nonempty list")
        planned, records, states, all_tasks = set(), [], Counter(), {}
        for relative in shards:
            directory = _child_path(root, relative)
            if directory in seen_paths:
                raise ValueError(f"Duplicate shard path: {relative}")
            seen_paths.add(directory)
            if not directory.exists():
                absent_shards.append(relative)
                continue
            manifest, tasks, rows = _snapshot(directory)
            if manifest["config"]["trials"] != trials:
                raise ValueError(f"Trial count differs from run manifest: {relative}")
            conditions = manifest["config"]["conditions"]
            if len(conditions) != 1 or conditions[0]["id"] != run["condition"]:
                raise ValueError(f"Condition differs from run manifest: {relative}")
            current_protocol = fingerprint(manifest["measurement_protocol"])
            current_condition = fingerprint(conditions)
            current_sources = fingerprint([(s["root"], s["sha256"]) for s in manifest["sources"]])
            if protocol_hash is not None and current_protocol != protocol_hash:
                raise ValueError(f"Measurement/scorer protocol differs across shards: {relative}")
            if condition_hash is not None and current_condition != condition_hash:
                raise ValueError(f"Condition configuration differs across shards: {relative}")
            if source_hash is not None and current_sources != source_hash:
                raise ValueError(f"Source code versions differ across shards: {relative}")
            protocol_hash, condition_hash, source_hash = current_protocol, current_condition, current_sources
            protocol = manifest["measurement_protocol"]
            task_map = {task["id"]: task for task in tasks}
            meta = {task["id"]: task for task in manifest["tasks"]}
            if len(task_map) != len(tasks) or len(meta) != len(manifest["tasks"]) or set(meta) != set(task_map):
                raise ValueError(f"Duplicate/mismatched task metadata: {relative}")
            for task_id, task in task_map.items():
                case = task.get("public_case", {})
                if case.get("dataset") != name or case.get("split") != "dev" or case.get("id") != task_id:
                    raise ValueError(f"Task is not a matching development case: {relative}/{task_id}")
                if fingerprint(task) != meta[task_id]["fingerprint"]:
                    raise ValueError(f"Task fingerprint mismatch: {relative}/{task_id}")
                if task_id in all_tasks:
                    raise ValueError(f"Duplicate dataset:task_id across shards: {name}:{task_id}")
                all_tasks[task_id] = task
            schedule = {(item["condition"], item["task_id"], item["trial_id"]) for item in manifest["schedule"]}
            anticipated = {(run["condition"], task, trial) for task in task_map for trial in range(trials)}
            database_keys = {(row["condition"], row["task_id"], row["trial_id"]) for row in rows}
            if (schedule != anticipated or len(schedule) != len(manifest["schedule"])
                    or database_keys != schedule or len(database_keys) != len(rows)):
                raise ValueError(f"Manifest/SQLite planned rows do not match: {relative}")
            for row in rows:
                key = (row["task_id"], row["trial_id"])
                if key not in expected or key in planned:
                    raise ValueError(f"Unexpected or duplicate trial: {name}:{key}")
                planned.add(key)
                if row["state"] not in _STATES:
                    raise ValueError(f"Unknown trial state: {row['state']}")
                states[row["state"]] += 1
                if row["state"] != "completed":
                    continue
                record = json.loads(row["record"])
                required = {"schema_version": 1, "scoring_version": manifest["scoring_version"],
                            "experiment_id": manifest["experiment_id"], "condition": row["condition"],
                            "task_id": row["task_id"], "trial_id": row["trial_id"], "status": "completed",
                            "task_fingerprint": meta[row["task_id"]]["fingerprint"],
                            "group_id": meta[row["task_id"]]["group_id"],
                            "dataset_fingerprint": manifest["dataset_fingerprint"]}
                if any(record.get(key) != value for key, value in required.items()):
                    raise ValueError(f"Completed record identity/version mismatch: {relative}/{row['task_id']}")
                model = record.get("outcome", {}).get("metadata", {}).get("model")
                if model is not None and model != run["model"]:
                    raise ValueError(f"Observed model differs from run manifest: {relative}/{row['task_id']}")
                records.append({**record, "shard": relative})
            snapshots.append({"dataset": name, "path": relative, "experiment_id": manifest["experiment_id"],
                              "manifest_fingerprint": manifest["manifest_fingerprint"],
                              "dataset_fingerprint": manifest["dataset_fingerprint"],
                              "planned": len(rows), "completed": sum(row["state"] == "completed" for row in rows)})
        states["unplanned"] += len(expected - planned)
        summaries[name] = _dataset_summary(name, expected, planned, records, states, all_tasks, trials)
    totals = {key: sum(dataset[key] for dataset in summaries.values()) for key in
              ("planned", "completed", "failed", "execution_errors", "agent_failed", "scorer_error_runs", "unavailable_scoring_runs",
               "observed_tokens", "tokens_observed_runs", "cost_unknown_runs", "elapsed_seconds_sum")}
    totals["completion_coverage"] = totals["completed"] / totals["planned"]
    totals["token_observation_coverage"] = totals["tokens_observed_runs"] / totals["planned"]
    partial = totals["completed"] != totals["planned"]
    return {"schema_version": 1, "scoring_version": 2, "model": run["model"], "condition": run["condition"],
            "split": "dev", "trials": trials, "summarized_at": now.isoformat(),
            "started_at": run["started_at"], "finished_at": run.get("finished_at"),
            "wall_seconds": wall_seconds, "wall_time_final": finished is not None,
            "partial": partial, "status": "partial" if partial else "completed",
            "run_manifest_fingerprint": fingerprint(run), "measurement_protocol": protocol,
            "summary_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "measurement_protocol_fingerprint": protocol_hash,
            "condition_fingerprint": condition_hash, "source_fingerprint": source_hash,
            "totals": totals, "datasets": summaries, "shards": snapshots, "absent_shards": absent_shards,
            "notes": ["Completed includes failed executions; answer errors remain zero scores, not discarded trials.",
                      "Token counts are observed lower bounds when a request/worker fails or usage is unavailable.",
                      "Each shard is a committed SQLite snapshot; concurrent shards may be sampled at slightly different times.",
                      "Question-level Wilson intervals assume independent tasks; family/document correlations are not corrected.",
                      "Development-set results support iteration, not an untouched-test generalization claim."]}


def render_markdown(summary) -> str:
    totals = summary["totals"]
    lines = [f"# {summary['model']} development evaluation", "",
             f"Status: **{summary['status']}**; {totals['completed']}/{totals['planned']} committed trials; "
             f"{totals['failed']} runtime/scoring failure runs.", "",
             "| Dataset | Completed/planned | Metric | Observed mean | Valid/planned | 95% Wilson CI |",
             "|---|---:|---|---:|---:|---|"]
    for name, dataset in summary["datasets"].items():
        keys = (["public_numeric_exact", "public_format_valid"] if name == "gsm8k" else
                ["public_answer_em", "public_answer_f1", "public_evidence_f1", "public_joint_em", "public_joint_f1", "public_format_valid"])
        for key in keys:
            metric = dataset["metrics"][key]
            mean = "unavailable" if metric["mean"] is None else f"{metric['mean']:.2%}"
            ci = metric["wilson_95"]
            interval = f"[{ci['low']:.2%}, {ci['high']:.2%}]" if ci else "—"
            lines.append(f"| {name} | {dataset['completed']}/{dataset['planned']} | {key} | {mean} | "
                         f"{metric['valid']}/{metric['planned']} | {interval} |")
    lines += ["", f"Observed tokens: **{totals['observed_tokens']:,}**; "
              f"{totals['cost_unknown_runs']} planned runs have unknown or incomplete token coverage.",
              f"Wall time: {summary['wall_seconds']:.1f} s "
              f"({'final' if summary['wall_time_final'] else 'snapshot'}); "
              f"sum of completed execution time: {totals['elapsed_seconds_sum']:.1f} s.", ""]
    for name, dataset in summary["datasets"].items():
        usage = dataset["api_usage"]
        lines.append(f"{name} API-reported prompt/completion/total tokens: "
                     f"{usage['prompt_tokens']} / {usage['completion_tokens']} / {usage['total_tokens']}.")
    lines += ["", "## Limits and accounting", "", *[f"- {note}" for note in summary["notes"]], "",
              "## Failure records", ""]
    for name, dataset in summary["datasets"].items():
        if dataset["errors"]:
            lines += [f"### {name}", "", "```json", json.dumps(dataset["errors"], ensure_ascii=False, indent=2), "```", ""]
    if not any(dataset["errors"] for dataset in summary["datasets"].values()):
        lines.append("No committed execution, agent-status or scoring failures.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    summary = summarize_run(args.root)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json_report(args.out, summary)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(summary), encoding="utf-8")
    if not args.out and not args.markdown:
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return summary


if __name__ == "__main__":
    main()
