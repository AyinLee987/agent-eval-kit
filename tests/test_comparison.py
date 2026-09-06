import copy
import math

import pytest

from agent_eval.comparison import ComparisonError, compare_experiments


def record(task="t1", trial=0, score=1, *, condition="A", group=None, **changes):
    row = {
        "condition": condition, "task_id": task, "trial_id": trial,
        "group_id": group, "dataset_fingerprint": "dataset-v1",
        "task_fingerprint": f"fingerprint-{task}", "status": "completed",
        "outcome": {"answer": "ok", "success": True, "tokens": 10},
        "elapsed_seconds": 2, "scores": {"quality": score},
        "metric_statuses": {"quality": "valid"},
    }
    row.update(changes)
    return row


def compare(a, b, **kwargs):
    return compare_experiments(a, b, metric="quality", n_resamples=64, **kwargs)


def test_trials_and_correlated_tasks_are_not_independent_samples():
    a, b = [], []
    for task, group in (("t1", "g1"), ("t2", "g1"), ("t3", "g2")):
        for trial in range(10):
            a.append(record(task, trial, 0, group=group))
            b.append(record(task, trial, 1, group=group, condition="B"))
    report = compare(a, b, expected_keys=[(r["task_id"], r["trial_id"]) for r in a])
    assert report["planned_observations_per_arm"] == 30
    assert report["planned_tasks"] == 3
    assert report["planned_units"] == report["paired"]["n_units"] == 2
    assert report["paired"]["test"]["n"] == 2
    assert report["paired"]["test"]["p_value"] == .5
    assert report["paired"]["mean_delta_b_minus_a"] == 1
    assert len(report["improvements"]) == 2


def test_each_task_then_each_group_gets_equal_weight_not_each_trial():
    a = [record("t1", i, 0, group="g1") for i in range(3)]
    a += [record("t2", 0, 0, group="g1"), record("t3", 0, 0, group="g2")]
    b = [copy.deepcopy(r) for r in a]
    for row in b:
        row["condition"] = "B"
        row["scores"]["quality"] = 1 if row["task_id"] == "t1" else 0
    report = compare(a, b, expected_keys=[(r["task_id"], r["trial_id"]) for r in a])
    assert report["paired"]["mean_b"] == .25
    assert report["paired"]["pairs"][0]["b"] == .5


def test_unscored_trial_excludes_whole_group_but_is_counted():
    a = [record("t1", i, group="g1") for i in range(2)]
    a += [record("t2", 0, group="g1"), record("t3", 0, group="g2")]
    b = copy.deepcopy(a)
    b[0]["metric_statuses"]["quality"] = "failed"
    with pytest.raises(ComparisonError, match="Incomplete") as error:
        compare(a, b, expected_keys=[(r["task_id"], r["trial_id"]) for r in a])
    assert any(d.get("reason") == "failed" for d in error.value.diagnostics)
    report = compare(a, b, expected_keys=[(r["task_id"], r["trial_id"]) for r in a], allow_incomplete=True)
    assert report["arms"]["b"]["metric_failures"] == 1
    assert report["arms"]["b"]["metric_coverage"] == .75
    assert report["paired"]["n_units"] == 1
    assert report["excluded_units"][0]["task_ids"] == ["t1", "t2"]
    assert report["paired"]["test"] is None
    # Failed grading still incurred validly recorded costs.
    assert report["costs"]["tokens"]["n_units"] == 2


def test_true_agent_failure_is_not_dropped_and_new_failures_are_reported():
    a = [record("t1"), record("t2")]
    b = [record("t1", score=0, outcome={"success": False, "tokens": 30}), record("t2")]
    report = compare(a, b)
    assert report["arms"]["b"]["agent_failed"] == 1
    assert report["arms"]["b"]["valid"] == 2
    assert report["paired"]["mean_delta_b_minus_a"] == -.5
    assert report["new_failures"] == [{"task_id": "t1", "trial_id": 0}]
    assert report["costs"]["tokens"]["mean_delta_b_minus_a"] == 10


def test_failed_execution_never_uses_stale_valid_score():
    a = [record("t1"), record("t2")]
    b = copy.deepcopy(a)
    b[0]["execution_error"] = {"message": "crashed"}
    report = compare(a, b, allow_incomplete=True)
    assert report["arms"]["b"]["execution_errors"] == 1
    assert report["paired"]["n_units"] == 1
    assert report["new_failures"] == [{"task_id": "t1", "trial_id": 0}]


@pytest.mark.parametrize("field,value", [("task_fingerprint", "different"), ("dataset_fingerprint", "different"), ("group_id", "changed")])
def test_lineage_or_group_changes_are_never_allowed(field, value):
    a = [record()]
    b = [record(**{field: value})]
    with pytest.raises(ComparisonError, match="comparable"):
        compare(a, b, allow_incomplete=True)


def test_duplicate_keys_rejected_even_in_incomplete_mode():
    with pytest.raises(ComparisonError, match="Duplicate"):
        compare([record(), record()], [record()], allow_incomplete=True)


def test_mixed_conditions_cannot_be_merged():
    with pytest.raises(ComparisonError) as error:
        compare([record("t1"), record("t2", condition="B")], [record("t1"), record("t2")])
    assert error.value.diagnostics == [{"code": "mixed_conditions", "arm": "a"}]


def test_missing_fingerprint_is_descriptive_only_with_diagnostics():
    a, b = [record("t1"), record("t2")], [record("t1"), record("t2")]
    del b[1]["task_fingerprint"]
    report = compare(a, b)
    assert report["paired"]["n_units"] == 2
    assert report["paired"]["test"] is None
    assert report["paired"]["inference_unavailable_reason"] == "unverified_lineage"
    assert any(d["code"] == "missing_fingerprint" for d in report["diagnostics"])


def test_manifest_detects_trial_missing_from_both_arms():
    a = [record("t1", 0), record("t2", 0)]
    keys = [(task, trial) for task in ("t1", "t2") for trial in range(2)]
    with pytest.raises(ComparisonError, match="Incomplete"):
        compare(a, a, expected_keys=keys)
    report = compare(a, a, expected_keys=keys, allow_incomplete=True)
    assert report["arms"]["a"]["missing_records"] == 2
    assert report["paired"]["n_units"] == 0
    assert report["paired"]["mean_a"] is None


def test_missing_task_group_from_manifest_excludes_its_whole_group():
    a = [record("t1", group="g1"), record("t3", group="g2")]
    plan = [{"task_id": task, "trial_id": 0, "group_id": group} for task, group in (("t1", "g1"), ("t2", "g1"), ("t3", "g2"))]
    report = compare(a, a, expected_keys=plan, allow_incomplete=True)
    assert report["planned_units"] == 2
    assert report["paired"]["pairs"][0]["task_ids"] == ["t3"]
    assert report["excluded_units"][0]["task_ids"] == ["t1", "t2"]


def test_without_manifest_plan_limitation_is_explicit():
    report = compare([record()], [record()])
    assert report["plan_source"] == "observed_task_trial_product"
    assert any(d["code"] == "inferred_plan" for d in report["diagnostics"])


def test_lower_is_better_changes_regressions_but_not_delta_sign():
    report = compare([record(score=.4)], [record(score=.8)], higher_is_better=False)
    assert len(report["regressions"]) == 1
    assert report["paired"]["mean_delta_b_minus_a"] == .4


def test_missing_cost_is_explicit_and_does_not_remove_score_pair():
    a = [record("t1"), record("t2")]
    b = copy.deepcopy(a)
    b[0]["outcome"]["tokens"] = None
    report = compare(a, b)
    assert report["paired"]["n_units"] == 2
    assert report["costs"]["tokens"]["n_units"] == 1
    assert report["costs"]["tokens"]["excluded_units"] == 1


@pytest.mark.parametrize("value", [math.nan, math.inf, "1", [1]])
def test_invalid_metric_is_failure_never_silently_coerced(value):
    report = compare([record()], [record(score=value)], allow_incomplete=True)
    assert report["arms"]["b"]["metric_failures"] == 1
    assert report["paired"]["n_units"] == 0


@pytest.mark.parametrize("kwargs", [{"n_resamples": 0}, {"n_resamples": True}, {"confidence": 1}, {"confidence": math.nan}, {"seed": True}])
def test_inference_parameters_validated_even_for_one_unit(kwargs):
    with pytest.raises(ValueError):
        compare_experiments([record()], [record()], metric="quality", **kwargs)


def test_no_plan_or_out_of_plan_or_duplicate_plan_rejected():
    for a, b, keys in (([], [], []), ([record()], [record()], [("other", 0)]), ([record()], [record()], [("t1", 0), ("t1", 0)])):
        with pytest.raises(ComparisonError):
            compare(a, b, expected_keys=keys)


def test_seeded_report_reproducible_and_json_serializable():
    import json
    a = [record(f"t{i}", score=i / 10) for i in range(8)]
    b = [record(f"t{i}", score=(9 - i) / 10) for i in range(8)]
    first = compare(a, b)
    assert first == compare(a, b)
    json.dumps(first, allow_nan=False)



def test_cost_totals_preserve_recorded_failed_execution_costs():
    a = [record("t1"), record("t2")]
    b = [record("t1", execution_error={"message": "crash"}, outcome={"tokens": 50}), record("t2")]
    report = compare(a, b, allow_incomplete=True)
    assert report["costs"]["tokens"]["observed_costs"]["b"] == {"observations_with_cost": 2, "missing_or_invalid_cost": 0, "observed_total": 60}


def test_json_invalid_status_is_a_reported_metric_failure():
    a, b = [record()], [record(metric_statuses={"quality": []})]
    report = compare(a, b, allow_incomplete=True)
    assert report["arms"]["b"]["metric_failures"] == 1
