"""Audit and compare saved intervention-ladder results without running agents.

Legacy dumps contain explicit task_id/trial identifiers but no task/dataset
fingerprints or independent group identities. They support descriptive paired
comparisons only: CI and p-values are disabled until verified lineage exists.
Errors remain observations; a failed/missing trial excludes the whole paired
unit in explicitly requested --allow-incomplete analyses.

Duplicate condition names require an explicit source selection, for example:
    python benchmarks/intervention_ladder/analyze.py results_flat_baseline.json \
        results_flat_prompts.json --condition-source instruction=results_flat_prompts.json \
        --condition-source fewshot=results_flat_prompts.json

No source result file is rewritten. Missing trial identifiers are never
reconstructed from row order, and trigger labels are never invented as groups.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent_eval.comparison import ComparisonError, compare_experiments
from agent_eval.score_reporting import write_json_report

BASELINE = "baseline"
FIELDS = ("exact_sequence_match", "step_recall", "answer_ok")


class AnalysisError(ComparisonError):
    """Legacy data cannot support the requested comparison."""


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise AnalysisError("Duplicate JSON key", [{"code": "duplicate_json_key", "key": key}])
        value[key] = item
    return value


def load(paths: list[str], *, condition_sources: dict[str, str] | None = None) -> dict[str, dict]:
    """Load every condition, refusing silent overwrites and preserving errors."""
    candidates = defaultdict(list)
    for path in paths:
        path = Path(path).resolve()
        blob = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object)
        if not isinstance(blob, dict) or not isinstance(blob.get("conditions"), dict):
            raise AnalysisError("Expected a conditions object", [{"code": "invalid_dump", "source": str(path)}])
        for name, res in blob["conditions"].items():
            if not isinstance(name, str) or not name or not isinstance(res, dict) or not isinstance(res.get("rows"), list):
                raise AnalysisError("Invalid condition result", [{"code": "invalid_condition", "source": str(path)}])
            if res.get("condition", name) != name:
                raise AnalysisError("Condition key differs from saved condition name", [{"code": "condition_name_mismatch", "condition": name}])
            candidates[name].append({**res, "_source": str(path), "_metadata": {key: value for key, value in blob.items() if key != "conditions"}})
    choices = {name: str(Path(path).resolve()) for name, path in (condition_sources or {}).items()}
    unknown = set(choices) - set(candidates)
    if unknown:
        raise AnalysisError("Unknown condition-source selection", [{"code": "unknown_condition_source", "condition": name} for name in sorted(unknown)])
    result = {}
    for name, versions in candidates.items():
        if name in choices:
            selected = [res for res in versions if res["_source"] == choices[name]]
            if len(selected) != 1:
                raise AnalysisError("Selected source must occur exactly once and contain that condition", [{"code": "invalid_condition_source", "condition": name, "source": choices[name]}])
            result[name] = selected[0]
            result[name]["_excluded_sources"] = [res["_source"] for res in versions if res is not selected[0]]
        elif len(versions) != 1:
            raise AnalysisError("Duplicate condition name; use --condition-source NAME=FILE", [{"code": "duplicate_condition", "condition": name, "sources": [res["_source"] for res in versions]}])
        else:
            result[name] = versions[0]
    if not result:
        raise AnalysisError("No conditions found", [{"code": "empty_conditions"}])
    return result


def _key(row):
    if not isinstance(row, dict):
        raise AnalysisError("Each result row must be an object", [{"code": "invalid_row"}])
    task, trial = row.get("task_id"), row.get("trial_id", row.get("trial"))
    if "trial_id" in row and "trial" in row and row["trial_id"] != row["trial"]:
        raise AnalysisError("Conflicting trial identifiers", [{"code": "conflicting_trial_identifiers", "task_id": task}])
    if not isinstance(task, str) or not task or isinstance(trial, bool) or not isinstance(trial, int) or trial < 0:
        raise AnalysisError("Rows need actual task_id and integer trial identifiers; row order cannot supply them", [{"code": "missing_or_invalid_identity", "task_id": task}])
    return task, trial


def _number(value):
    if not isinstance(value, (int, float)):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except OverflowError:
        return None


def per_task(res: dict, field: str, *, allow_incomplete: bool = False) -> dict[str, float]:
    """Legacy helper: exclude a whole task on error, never average fewer trials."""
    indexed = {}
    by_task = defaultdict(list)
    for row in res["rows"]:
        key = _key(row)
        if key in indexed:
            raise AnalysisError("Duplicate task/trial", [{"code": "duplicate_key", "task_id": key[0], "trial_id": key[1]}])
        indexed[key] = row
        by_task[key[0]].append(row)
    repeat = res.get("_metadata", {}).get("repeat")
    means = {}
    for task, rows in by_task.items():
        values = [_number(row.get(field)) for row in rows]
        invalid = any(row.get("error") for row in rows) or any(value is None or not 0 <= value <= 1 for value in values) or repeat is not None and len(rows) != repeat
        if invalid:
            if not allow_incomplete:
                raise AnalysisError("Task has an error, missing metric or incomplete trials", [{"code": "incomplete_task", "task_id": task, "metric": field}])
            continue
        means[task] = math.fsum(value / len(values) for value in values)
    return means


def analyze_conditions(conditions, *, baseline=BASELINE, fields=FIELDS, allow_incomplete=False, n_resamples=10000, seed=0):
    if baseline not in conditions:
        raise AnalysisError("Baseline condition is absent", [{"code": "missing_baseline", "condition": baseline}])
    if not fields or any(field not in FIELDS for field in fields) or len(fields) != len(set(fields)):
        raise ValueError("metrics must be unique members of " + ", ".join(FIELDS))
    indexed, diagnostics, repeats, counts = {}, [], set(), set()
    task_sequences = defaultdict(set)
    for name, res in conditions.items():
        rows = {}
        for row in res["rows"]:
            key = _key(row)
            if key in rows:
                raise AnalysisError("Duplicate task/trial", [{"code": "duplicate_key", "condition": name, "task_id": key[0], "trial_id": key[1]}])
            rows[key] = row
            sequence = row.get("expect_tool_sequence")
            if sequence is not None:
                if not isinstance(sequence, list) or any(not isinstance(tool, str) for tool in sequence):
                    raise AnalysisError("Invalid expected tool sequence", [{"code": "invalid_task_content", "task_id": key[0]}])
                task_sequences[key[0]].add(tuple(sequence))
        indexed[name] = rows
        meta = res.get("_metadata", {})
        for key, target in (("repeat", repeats), ("task_count", counts)):
            value = meta.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise AnalysisError("Saved task_count and repeat are required to audit completeness", [{"code": "missing_plan_metadata", "condition": name, "field": key}])
            target.add(value)
        if res.get("total") != len(rows):
            raise AnalysisError("Saved total disagrees with actual rows", [{"code": "row_count_mismatch", "condition": name, "declared": res.get("total"), "observed": len(rows)}])
        if res.get("_excluded_sources"):
            diagnostics.append({"code": "explicit_condition_source", "condition": name, "selected": res["_source"], "excluded": res["_excluded_sources"]})
    if len(repeats) != 1 or len(counts) != 1:
        raise AnalysisError("Conditions declare different task/trial plans", [{"code": "incompatible_declared_plans"}])
    conflicts = [task for task, values in task_sequences.items() if len(values) > 1]
    if conflicts:
        raise AnalysisError("Stored expected tool sequence changed for the same task id", [{"code": "task_content_mismatch", "task_id": task} for task in sorted(conflicts)])
    task_ids = sorted({task for rows in indexed.values() for task, _ in rows})
    trial_ids = sorted({trial for rows in indexed.values() for _, trial in rows})
    repeat, task_count = next(iter(repeats)), next(iter(counts))
    # Counts alone cannot tell us the identity of a task/trial missing everywhere.
    # Refuse rather than create fictional keys or call the observed subset complete.
    if len(task_ids) != task_count or len(trial_ids) != repeat:
        raise AnalysisError("Observed identities cannot reconstruct the declared plan; supply the original complete plan via the experiment API", [{"code": "unidentifiable_planned_keys", "declared_tasks": task_count, "observed_task_ids": len(task_ids), "declared_trials": repeat, "observed_trial_ids": trial_ids}])
    expected_keys = [(task, trial) for task in task_ids for trial in trial_ids]
    normalized, samples = {}, {}
    for name, rows in indexed.items():
        res = conditions[name]
        meta = res.get("_metadata", {})
        normalized[name] = []
        for (task, trial), row in rows.items():
            scores, statuses = {}, {}
            for field in fields:
                value = _number(row.get(field))
                scores[field] = value if value is not None and 0 <= value <= 1 else None
                statuses[field] = "execution_error" if row.get("error") else "valid" if scores[field] is not None else "failed"
            normalized[name].append({
                "condition": name, "task_id": task, "trial_id": trial,
                "group_id": row.get("group_id"), "task_fingerprint": row.get("task_fingerprint"),
                "dataset_fingerprint": row.get("dataset_fingerprint", meta.get("dataset_fingerprint")),
                "execution_error": {"message": str(row["error"])} if row.get("error") else None,
                "outcome": {"success": row.get("answer_ok") if isinstance(row.get("answer_ok"), bool) else None, "tokens": row.get("tokens")},
                "elapsed_seconds": row.get("elapsed_seconds"),
                "scores": scores, "metric_statuses": statuses,
            })
        samples[name] = {
            "source": res.get("_source"), "planned": task_count * repeat, "observed": len(rows),
            "missing_records": task_count * repeat - len(rows),
            "api_errors": sum(bool(row.get("error")) for row in rows.values()),
            "failure_kinds": dict(sorted(Counter("error" if row.get("error") else row.get("failure_kind", "unknown") for row in rows.values()).items())),
            "costs": {},
        }
        for field in ("steps", "delegate_calls", "tokens", "elapsed_seconds"):
            values = [number for row in rows.values() if (number := _number(row.get(field))) is not None and number >= 0]
            samples[name]["costs"][field] = {"observed": len(values), "missing": task_count * repeat - len(values), "mean_observed": math.fsum(value / len(values) for value in values) if values else None}
    comparisons = {}
    for field in fields:
        comparisons[field] = {}
        for name in conditions:
            comparisons[field][name] = compare_experiments(
                normalized[baseline], normalized[name], metric=field, expected_keys=expected_keys,
                allow_incomplete=allow_incomplete, n_resamples=n_resamples, seed=seed,
            )
    legacy = any(not row.get("task_fingerprint") or not row.get("dataset_fingerprint") for rows in normalized.values() for row in rows)
    if legacy:
        diagnostics.append({"code": "legacy_unverified_lineage", "message": "Historical task/trial IDs and saved expected sequences do not establish identical task content or independent group identity. No fingerprints are synthesized; CI and p-values are disabled."})
    return {"schema_version": 1, "baseline": baseline, "planned_tasks": task_count,
            "declared_trials_per_task": repeat, "observed_trial_ids": trial_ids,
            "allow_incomplete": allow_incomplete, "legacy_unverified_lineage": legacy,
            "conditions": samples, "comparisons": comparisons, "diagnostics": diagnostics}


def render(report):
    lines = [f"Baseline: {report['baseline']}; planned tasks={report['planned_tasks']}, trials/task={report['declared_trials_per_task']}"]
    if report["legacy_unverified_lineage"]:
        lines.append("LEGACY: fingerprints/group provenance unavailable; descriptive comparisons only, CI/p disabled.")
    lines.append("Condition | planned | observed | API errors | missing records")
    for name, samples in report["conditions"].items():
        lines.append(f"{name} | {samples['planned']} | {samples['observed']} | {samples['api_errors']} | {samples['missing_records']}")
    for field, comparisons in report["comparisons"].items():
        lines.append(f"\n{field}: condition | paired units | mean baseline | mean condition | delta | CI/p")
        for name, comparison in comparisons.items():
            paired = comparison["paired"]
            def number(value):
                return "unavailable" if value is None else f"{value:.4g}"
            inference = "disabled: " + paired["inference_unavailable_reason"] if paired["test"] is None else f"[{paired['ci']['low']:.4g}, {paired['ci']['high']:.4g}], p={paired['test']['p_value']:.3g}"
            lines.append(f"{name} | {paired['n_units']} | {number(paired['mean_a'])} | {number(paired['mean_b'])} | {number(paired['mean_delta_b_minus_a'])} | {inference}")
            if comparison["excluded_units"]:
                lines.append("  excluded units: " + ", ".join(unit["unit_id"] for unit in comparison["excluded_units"]))
            arm = comparison["arms"]["b"]
            lines.append(f"  condition observations: valid={arm['valid']}, metric failures={arm['metric_failures']}, execution errors={arm['execution_errors']}, missing={arm['missing_records'] + arm['missing_metrics']}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--condition-source", action="append", default=[], metavar="NAME=FILE")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--metric", action="append", choices=FIELDS)
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)
    try:
        choices = {}
        for selection in args.condition_source:
            name, sep, source = selection.partition("=")
            if not sep or not name or not source or name in choices:
                raise ValueError("Use each --condition-source NAME=FILE exactly once")
            choices[name] = source
        report = analyze_conditions(load(args.paths, condition_sources=choices), baseline=args.baseline,
                                    fields=args.metric or FIELDS, allow_incomplete=args.allow_incomplete)
        if args.json_out:
            write_json_report(args.json_out, report)
        print(render(report))
        return 0
    except (ValueError, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        if getattr(exc, "diagnostics", None):
            print(json.dumps(exc.diagnostics, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
