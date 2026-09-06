"""Auditable paired comparisons of persisted experiment records.

Repeated trials are averaged within task, then tasks within group. Inference
uses complete independent groups (or tasks without a group_id), never trials.
The reported permutation test requires within-pair label exchangeability;
matching records cannot establish that experimental-design assumption.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict
from numbers import Real
from typing import Any, Mapping, Sequence

from .stats import bootstrap_ci, paired_permutation_test

Key = tuple[str, int]


class ComparisonError(ValueError):
    """Invalid comparison, with machine-readable reasons for the CLI."""

    def __init__(self, message: str, diagnostics: list[dict[str, Any]]):
        super().__init__(message)
        self.diagnostics = diagnostics


def _finite(value: Any) -> float | None:
    if not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Sequence[float]) -> float:
    return math.fsum(value / len(values) for value in values)


def _key(task_id: Any, trial_id: Any) -> Key:
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task_id must be a nonempty string")
    if isinstance(trial_id, bool) or not isinstance(trial_id, int) or trial_id < 0:
        raise ValueError("trial_id must be a nonnegative integer")
    return task_id, trial_id


def _key_json(key: Key) -> dict[str, Any]:
    return {"task_id": key[0], "trial_id": key[1]}


def _index(records: Sequence[Mapping[str, Any]], arm: str) -> dict[Key, dict]:
    indexed: dict[Key, dict] = {}
    for row in records:
        if not isinstance(row, Mapping):
            raise ComparisonError("Each record must be an object", [{"code": "invalid_record", "arm": arm}])
        if "outcome" in row and not isinstance(row["outcome"], Mapping):
            raise ComparisonError("outcome must be an object", [{"code": "invalid_outcome", "arm": arm}])
        try:
            key = _key(row.get("task_id"), row.get("trial_id"))
        except ValueError as exc:
            raise ComparisonError(str(exc), [{"code": "invalid_key", "arm": arm}]) from exc
        if key in indexed:
            raise ComparisonError("Duplicate task/trial key", [{"code": "duplicate_key", "arm": arm, **_key_json(key)}])
        indexed[key] = dict(row)
    return indexed


def _outcome(row: Mapping[str, Any]) -> Mapping[str, Any]:
    # Flat schema-v2 scorecards remain readable when callers add trial metadata.
    outcome = row.get("outcome", row)
    return outcome if isinstance(outcome, Mapping) else {}


def _execution_failed(row: Mapping[str, Any]) -> bool:
    return (
        row.get("execution_error") is not None
        or row.get("status") in ("failed", "error", "execution_error", "blocked", "cancelled", "timeout", "running", "pending")
        or _outcome(row).get("stop_reason") == "execution_error"
    )


def _metric(row: Mapping[str, Any], metric: str) -> tuple[float | None, str]:
    if _execution_failed(row):
        return None, "execution_error"
    scores = row.get("scores", {})
    states = row.get("metric_statuses", row.get("statuses", {}))
    if not isinstance(scores, Mapping) or not isinstance(states, Mapping):
        return None, "failed"
    state = states.get(metric)
    value = _finite(scores.get(metric))
    if state not in (None, "valid"):
        return None, state if state in ("not_applicable", "failed", "execution_error", "blocked", "missing") else "failed"
    if value is None:
        return None, "failed" if metric in scores and scores[metric] is not None else "missing"
    return value, "valid"


def _cost(row: Mapping[str, Any], name: str) -> float | None:
    if name == "tokens" and row.get("tokens_observed") is False:
        return None
    value = row.get(name, _outcome(row).get(name))
    number = _finite(value)
    return number if not isinstance(value, bool) and number is not None and number >= 0 else None


def _paired_summary(
    pairs: list[dict], *, inference_allowed: bool, confidence: float,
    n_resamples: int, seed: int | None,
) -> dict:
    left = [pair["a"] for pair in pairs]
    right = [pair["b"] for pair in pairs]
    differences = [b - a for a, b in zip(left, right)]
    if not all(math.isfinite(value) for value in differences):
        raise ComparisonError("Paired differences overflow", [{"code": "nonfinite_difference"}])
    reason = None if inference_allowed and len(pairs) >= 2 else (
        "unverified_lineage" if not inference_allowed else "fewer_than_two_independent_units"
    )
    return {
        "n_units": len(pairs), "mean_a": _mean(left) if left else None,
        "mean_b": _mean(right) if right else None,
        "mean_delta_b_minus_a": _mean(differences) if differences else None,
        "ci": asdict(bootstrap_ci(differences, confidence=confidence, n_resamples=n_resamples, seed=seed)) if reason is None else None,
        "test": asdict(paired_permutation_test(left, right, n_resamples=n_resamples, seed=seed)) if reason is None else None,
        "inference_unavailable_reason": reason,
        "pairs": pairs,
    }


def compare_experiments(
    records_a: Sequence[Mapping[str, Any]],
    records_b: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    expected_keys: Sequence[tuple[str, int] | Mapping[str, Any]] | None = None,
    allow_incomplete: bool = False,
    higher_is_better: bool = True,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int | None = 0,
) -> dict[str, Any]:
    """Compare B minus A without silently discarding failed/missing records.

    expected_keys should come from the manifest's tasks x planned trials. It
    accepts (task_id, trial_id) pairs, or objects that can additionally provide
    group_id/task_fingerprint/dataset_fingerprint for wholly missing tasks.
    Without a manifest, the task x trial universe is inferred from observed
    records; entirely absent tasks/trials cannot be detected and are disclosed.

    Duplicate keys, changed fingerprints and changed group membership always
    raise ComparisonError. Missing/invalid metric observations also raise by
    default. allow_incomplete explicitly permits a complete-unit comparison:
    one missing trial excludes its whole task and group. Ordinary unsuccessful
    agent outcomes remain usable if their metric was validly scored.

    Missing lineage yields descriptive statistics only. A computed p-value is
    conditional on independent units and exchangeable paired A/B labels; this
    function does not verify randomization or correct multiple comparisons.
    """
    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("metric must be a nonempty string")
    if not isinstance(allow_incomplete, bool) or not isinstance(higher_is_better, bool):
        raise ValueError("allow_incomplete and higher_is_better must be booleans")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if isinstance(confidence, bool) or _finite(confidence) is None or not 0 < confidence < 1:
        raise ValueError("confidence must be finite and in (0, 1)")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("seed must be an integer or None")

    arms = {"a": _index(records_a, "a"), "b": _index(records_b, "b")}
    observed = set(arms["a"]) | set(arms["b"])
    diagnostics: list[dict] = []
    planned_metadata: dict[Key, dict] = {}
    if expected_keys is None:
        tasks = {key[0] for key in observed}
        trials = {key[1] for key in observed}
        planned = {(task, trial) for task in tasks for trial in trials}
        diagnostics.append({"code": "inferred_plan", "message": "Entirely absent tasks or trials cannot be detected without expected_keys."})
    else:
        planned = set()
        for item in expected_keys:
            try:
                if isinstance(item, Mapping):
                    key = _key(item.get("task_id"), item.get("trial_id"))
                    planned_metadata[key] = dict(item)
                else:
                    key = _key(*item)
            except (ValueError, TypeError) as exc:
                raise ComparisonError("Invalid expected key", [{"code": "invalid_expected_key"}]) from exc
            if key in planned:
                raise ComparisonError("Duplicate expected key", [{"code": "duplicate_expected_key", **_key_json(key)}])
            planned.add(key)
    if not planned:
        raise ComparisonError("No planned observations", [{"code": "empty_plan"}])
    unexpected = observed - planned
    if unexpected:
        raise ComparisonError("Observed records outside planned keys", [{"code": "unexpected_key", **_key_json(key)} for key in sorted(unexpected)])

    # Check lineage within an arm, across arms and against available manifests.
    sources = [(arm, key, row) for arm, rows in arms.items() for key, row in rows.items()]
    sources += [("plan", key, row) for key, row in planned_metadata.items()]
    groups: dict[str, set[Any]] = defaultdict(set)
    fingerprints: dict[str, set[str]] = defaultdict(set)
    datasets: set[str] = set()
    missing_lineage: list[dict] = []
    conditions: dict[str, set[str]] = {"a": set(), "b": set()}
    for arm, key, row in sources:
        group = row.get("group_id")
        if group is not None and (not isinstance(group, str) or not group.strip()):
            raise ComparisonError("Invalid group_id", [{"code": "invalid_group_id", "arm": arm, **_key_json(key)}])
        if arm != "plan" or "group_id" in row:
            groups[key[0]].add(group)
        for field in ("task_fingerprint", "dataset_fingerprint"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                if arm != "plan":
                    missing_lineage.append({"code": "missing_fingerprint", "field": field, "arm": arm, **_key_json(key)})
            elif field == "task_fingerprint":
                fingerprints[key[0]].add(value)
            else:
                datasets.add(value)
        condition = row.get("condition")
        if arm != "plan" and condition is not None:
            if not isinstance(condition, str) or not condition.strip():
                raise ComparisonError("Invalid condition", [{"code": "invalid_condition", "arm": arm}])
            conditions[arm].add(condition)
    conflicts = [{"code": "group_mismatch", "task_id": task} for task, values in groups.items() if len(values) > 1]
    conflicts += [{"code": "task_fingerprint_mismatch", "task_id": task} for task, values in fingerprints.items() if len(values) > 1]
    if len(datasets) > 1:
        conflicts.append({"code": "dataset_fingerprint_mismatch"})
    conflicts += [{"code": "mixed_conditions", "arm": arm} for arm, values in conditions.items() if len(values) > 1]
    if conflicts:
        raise ComparisonError("Records do not describe comparable experiments", conflicts)
    diagnostics.extend(missing_lineage)

    task_keys: dict[str, list[Key]] = defaultdict(list)
    units: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key in sorted(planned):
        task_keys[key[0]].append(key)
    unresolved_groups = []
    for task in task_keys:
        group = next(iter(groups.get(task, {None})))
        units[("group", group) if group is not None else ("task", task)].append(task)
        if task not in groups:
            unresolved_groups.append(task)
    if unresolved_groups and any(kind == "group" for kind, _ in units):
        diagnostics.append({"code": "unknown_group_membership", "tasks": unresolved_groups})

    values: dict[str, dict[Key, float]] = {"a": {}, "b": {}}
    arm_reports = {}
    unavailable = []
    for arm, indexed in arms.items():
        counts = Counter()
        for key in sorted(planned):
            row = indexed.get(key)
            value, state = (None, "missing_record") if row is None else _metric(row, metric)
            counts[state] += 1
            if value is not None:
                values[arm][key] = value
            else:
                unavailable.append({"code": "unavailable_observation", "arm": arm, "reason": state, **_key_json(key)})
        arm_reports[arm] = {
            "condition": next(iter(conditions[arm]), None),
            "planned": len(planned), "observed": len(indexed),
            "valid": counts["valid"], "missing_records": counts["missing_record"],
            "missing_metrics": counts["missing"], "not_applicable": counts["not_applicable"],
            "metric_failures": counts["failed"], "execution_errors": counts["execution_error"],
            "blocked": counts["blocked"],
            "agent_failed": sum(_outcome(row).get("success") is False and not _execution_failed(row) for row in indexed.values()),
            "metric_coverage": counts["valid"] / len(planned),
            "counts_by_status": dict(sorted(counts.items())),
        }
    diagnostics.extend(unavailable)
    if unavailable and not allow_incomplete:
        raise ComparisonError("Incomplete comparison: pass allow_incomplete=True to report complete units explicitly", diagnostics)

    inference_allowed = not missing_lineage and not (unresolved_groups and any(kind == "group" for kind, _ in units))
    excluded_units = []
    metric_pairs = []
    cost_pairs: dict[str, list[dict]] = {"tokens": [], "elapsed_seconds": []}
    for (kind, unit_id), tasks in sorted(units.items()):
        unit_keys = [key for task in tasks for key in task_keys[task]]
        base = {"unit_id": unit_id, "unit_kind": kind, "task_ids": tasks}
        if all(key in values[arm] for arm in arms for key in unit_keys):
            metric_pairs.append({**base, **{
                arm: _mean([_mean([values[arm][key] for key in task_keys[task]]) for task in tasks])
                for arm in arms
            }})
        else:
            excluded_units.append({**base, "reason": "incomplete_metric_observations", "keys": [_key_json(key) for key in unit_keys]})
        for cost, pairs in cost_pairs.items():
            costs = {arm: {key: _cost(arms[arm][key], cost) if key in arms[arm] else None for key in unit_keys} for arm in arms}
            if all(value is not None for arm_values in costs.values() for value in arm_values.values()):
                pairs.append({**base, **{
                    arm: _mean([_mean([costs[arm][key] for key in task_keys[task]]) for task in tasks])
                    for arm in arms
                }})
    kwargs = dict(inference_allowed=inference_allowed, confidence=confidence, n_resamples=n_resamples, seed=seed)
    paired = _paired_summary(metric_pairs, **kwargs)
    regressions, improvements = [], []
    for pair in metric_pairs:
        delta = pair["b"] - pair["a"]
        change = {**pair, "delta_b_minus_a": delta}
        directional_delta = delta if higher_is_better else -delta
        if directional_delta < 0:
            regressions.append(change)
        elif directional_delta > 0:
            improvements.append(change)
    new_failures, recovered = [], []
    for key in sorted(planned):
        a, b = arms["a"].get(key), arms["b"].get(key)
        if a is None or b is None:
            continue
        success_a = False if _execution_failed(a) else _outcome(a).get("success")
        success_b = False if _execution_failed(b) else _outcome(b).get("success")
        if success_a is True and success_b is False:
            new_failures.append(_key_json(key))
        elif success_a is False and success_b is True:
            recovered.append(_key_json(key))
    cost_reports = {}
    for name, pairs in cost_pairs.items():
        observed_costs = {}
        for arm, indexed in arms.items():
            recorded = [value for row in indexed.values() if (value := _cost(row, name)) is not None]
            try:
                total = math.fsum(recorded)
            except OverflowError:
                total = None
            observed_costs[arm] = {
                "observations_with_cost": len(recorded),
                "missing_or_invalid_cost": len(planned) - len(recorded),
                "observed_total": total,
            }
        cost_reports[name] = {
            **_paired_summary(pairs, **kwargs),
            "excluded_units": len(units) - len(pairs), "observed_costs": observed_costs,
        }
    return {
        "schema_version": 1, "metric": metric, "higher_is_better": higher_is_better,
        "difference_direction": "b_minus_a", "dataset_fingerprint": next(iter(datasets), None),
        "plan_source": "manifest" if expected_keys is not None else "observed_task_trial_product",
        "planned_observations_per_arm": len(planned), "planned_tasks": len(task_keys), "planned_units": len(units),
        "arms": arm_reports, "paired": paired,
        "excluded_units": excluded_units, "diagnostics": diagnostics,
        "regressions": regressions, "improvements": improvements,
        "new_failures": new_failures, "recovered": recovered,
        "costs": cost_reports,
        "allow_incomplete": allow_incomplete, "inference_scope": "complete_independent_units_only",
        "assumptions": [
            "Tasks sharing group_id are one independent unit; ungrouped tasks are independent units.",
            "Permutation inference requires exchangeable A/B labels within each pair; matching does not verify this.",
            "Complete-unit results may be biased when missingness depends on performance.",
            "Intervals/tests are unadjusted for multiple metric or condition comparisons.",
        ],
    }
