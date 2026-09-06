import copy
import json
import math
from pathlib import Path

import pytest

from agent_eval.calibration import CalibrationError, evaluate_calibration, export_review_template


@pytest.fixture
def program_check():
    return json.loads((Path(__file__).parent / "fixtures/calibration_program_check.json").read_text(encoding="utf-8"))


def test_deterministic_fixtures_check_math_without_claiming_human_calibration(program_check):
    report = evaluate_calibration(program_check["judge_records"], program_check["gold_document"])
    expected = program_check["expected"]
    assert report["reference_kind"] == "synthetic_program_check"
    assert report["provenance_verification"] == "synthetic_program_check_only"
    for name in ("quality", "pass"):
        assert report["dimensions"][name]["mae"] == expected[f"{name}_mae"]
        assert report["dimensions"][name]["agreement"] == expected[f"{name}_agreement"]
    for name in ("pass", "category"):
        assert report["dimensions"][name]["cohens_kappa"] == expected[f"{name}_kappa"]
        assert report["dimensions"][name]["confusion_matrix"]["counts"] == [[1, 1], [1, 1]]
    assert report["dimensions"]["category"]["mae"] is None
    json.dumps(report, allow_nan=False)


def test_export_is_unlabeled_blinded_and_does_not_mutate_records(program_check):
    records = program_check["judge_records"]
    before = copy.deepcopy(records)
    template = export_review_template(records, dimensions={"quality": {"kind": "numeric"}, "pass": {"kind": "binary"}})
    assert template["source_kind"] == "human_review"
    assert all(item["reference_scores"] == {"quality": None, "pass": None} for item in template["items"])
    assert all(item["human_reviewed"] is False and item["reviewer"] is None for item in template["items"])
    assert all("scores" not in item and "scores" not in item["review_material"] for item in template["items"])
    assert records == before
    report = evaluate_calibration(records, template)
    assert report["dimensions"]["quality"]["counts_by_status"] == {"unlabeled": 4}
    assert report["dimensions"]["quality"]["mae"] is None
    assert report["dimensions"]["quality"]["coverage"] == 0


def test_human_labels_require_explicit_attestation_and_reviewer(program_check):
    records = program_check["judge_records"]
    template = export_review_template(records, dimensions=["quality"])
    template["items"][0]["reference_scores"]["quality"] = 1
    with pytest.raises(CalibrationError, match="Human labels require"):
        evaluate_calibration(records, template)
    template["items"][0]["human_reviewed"] = True
    with pytest.raises(CalibrationError, match="Human labels require"):
        evaluate_calibration(records, template)
    template["items"][0]["reviewer"] = "reviewer-1"
    report = evaluate_calibration(records, template)
    assert report["provenance_verification"] == "user_attested"
    assert report["dimensions"]["quality"]["labeled"] == 1
    assert report["dimensions"]["quality"]["coverage"] == .25
    assert report["dimensions"]["quality"]["mae"] == 0


def test_synthetic_provenance_cannot_claim_human_review(program_check):
    program_check["gold_document"]["items"][0]["human_reviewed"] = True
    with pytest.raises(CalibrationError, match="Synthetic"):
        evaluate_calibration(program_check["judge_records"], program_check["gold_document"])


def test_missing_failed_and_invalid_judges_stay_in_denominator(program_check):
    records = program_check["judge_records"]
    records[0]["metric_statuses"]["quality"] = "failed"
    records[1]["scores"]["quality"] = math.nan
    records.pop()
    report = evaluate_calibration(records, program_check["gold_document"])["dimensions"]["quality"]
    assert report["planned"] == report["labeled"] == 4
    assert report["valid"] == 1
    assert report["coverage"] == report["judge_coverage_of_labeled"] == .25
    assert report["counts_by_status"] == {"judge_failed": 1, "invalid_judge_score": 1, "valid": 1, "missing_judge_record": 1}
    assert report["mae"] == .25


def test_execution_error_cannot_reuse_stale_judge_values(program_check):
    program_check["judge_records"][0]["execution_error"] = {"message": "run failed"}
    report = evaluate_calibration(program_check["judge_records"], program_check["gold_document"])
    assert report["dimensions"]["quality"]["counts_by_status"]["execution_error"] == 1


@pytest.mark.parametrize("field", ["dataset_fingerprint", "task_fingerprint"])
def test_fingerprint_conflict_rejected_missing_fingerprint_excluded(program_check, field):
    records, gold = program_check["judge_records"], program_check["gold_document"]
    records[0][field] = "wrong"
    with pytest.raises(CalibrationError, match="Mismatched"):
        evaluate_calibration(records, gold)
    del records[0][field]
    report = evaluate_calibration(records, gold)
    assert report["dimensions"]["quality"]["counts_by_status"]["unverified_lineage"] == 1


@pytest.mark.parametrize("side", ["judge_records", "gold_document"])
def test_duplicate_identities_rejected(program_check, side):
    items = program_check[side] if side == "judge_records" else program_check[side]["items"]
    items.append(copy.deepcopy(items[0]))
    with pytest.raises(CalibrationError, match="Duplicate"):
        evaluate_calibration(program_check["judge_records"], program_check["gold_document"])


def test_unexpected_predictions_are_reported_not_evaluated(program_check):
    extra = copy.deepcopy(program_check["judge_records"][0])
    extra["task_id"] = "new-task"
    program_check["judge_records"].append(extra)
    report = evaluate_calibration(program_check["judge_records"], program_check["gold_document"])
    assert report["unexpected_judge_records"] == 1
    assert report["dimensions"]["quality"]["planned"] == 4
    assert report["dimensions"]["quality"]["valid"] == 4


@pytest.mark.parametrize("value", [-.1, 1.1, math.nan, math.inf, "1", []])
def test_invalid_reference_labels_rejected(program_check, value):
    program_check["gold_document"]["items"][0]["reference_scores"]["quality"] = value
    with pytest.raises(CalibrationError, match="Invalid reference label"):
        evaluate_calibration(program_check["judge_records"], program_check["gold_document"])


def test_custom_numeric_scale_tolerance_and_undefined_kappa(program_check):
    gold, records = program_check["gold_document"], program_check["judge_records"]
    gold["dimensions"]["quality"] = {"kind": "numeric", "minimum": 0, "maximum": 5, "agreement_tolerance": .5}
    for ref, prediction in zip(gold["items"], records):
        ref["reference_scores"]["quality"] = 5
        prediction["scores"]["quality"] = 4.5
        ref["reference_scores"]["pass"] = prediction["scores"]["pass"] = 1
    report = evaluate_calibration(records, gold)
    assert report["dimensions"]["quality"]["mae"] == .5
    assert report["dimensions"]["quality"]["agreement"] == 1
    assert report["dimensions"]["pass"]["agreement"] == 1
    assert report["dimensions"]["pass"]["cohens_kappa"] is None
    assert report["dimensions"]["pass"]["kappa_unavailable_reason"] == "chance_agreement_is_one"


@pytest.mark.parametrize("dimensions", [[], ["x", "x"], {"x": {"kind": "categorical", "labels": ["same", "same"]}}, {"x": {"minimum": 1, "maximum": 0}}, {"x": {"agreement_tolerance": -1}}])
def test_invalid_dimension_schema_rejected_on_export(program_check, dimensions):
    with pytest.raises(CalibrationError):
        export_review_template(program_check["judge_records"], dimensions=dimensions)


def test_export_requires_verified_dataset_and_task_lineage(program_check):
    records = program_check["judge_records"]
    with pytest.raises(CalibrationError, match="match dataset"):
        export_review_template(records, dimensions=["quality"], dataset_fingerprint="wrong")
    del records[0]["task_fingerprint"]
    with pytest.raises(CalibrationError, match="task_fingerprint"):
        export_review_template(records, dimensions=["quality"])


@pytest.mark.parametrize("field,value", [("answer", "different rerun answer"), ("trajectory", [{"action": "new tool call"}])])
def test_labels_cannot_be_reused_for_changed_answer_or_trace(program_check, field, value):
    program_check["judge_records"][0]["outcome"][field] = value
    with pytest.raises(CalibrationError, match="Mismatched reviewed output"):
        evaluate_calibration(program_check["judge_records"], program_check["gold_document"])


def test_calibration_rejects_boolean_schema_version(program_check):
    program_check["gold_document"]["schema_version"] = True
    with pytest.raises(CalibrationError, match="schema"):
        evaluate_calibration(program_check["judge_records"], program_check["gold_document"])
