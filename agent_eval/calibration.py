"""Offline human-reference review templates and judge validation.

Exported templates are UNLABELED. Human provenance is an explicit attestation
by the person filling a template, not something software can independently
verify. Synthetic fixtures check the program only and are always reported as
such; a fixture result is never evidence of human calibration.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from numbers import Real
from typing import Any, Mapping, Sequence


class CalibrationError(ValueError):
    """Malformed labels, ambiguous identities, or incompatible provenance."""


def _number(value: Any) -> float | None:
    if not isinstance(value, Real):
        return None
    try:
        result = float(value)
    except (ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _identity(row: Mapping[str, Any]) -> tuple[str | None, str, int]:
    task, trial, condition = row.get("task_id"), row.get("trial_id"), row.get("condition")
    if not isinstance(task, str) or not task.strip():
        raise CalibrationError("task_id must be a nonempty string")
    if isinstance(trial, bool) or not isinstance(trial, int) or trial < 0:
        raise CalibrationError("trial_id must be a nonnegative integer")
    if condition is not None and (not isinstance(condition, str) or not condition.strip()):
        raise CalibrationError("condition must be a nonempty string or null")
    return condition, task, trial


def _index(records: Sequence[Mapping[str, Any]]) -> dict[tuple, Mapping[str, Any]]:
    result = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise CalibrationError("Each record/item must be an object")
        key = _identity(record)
        if key in result:
            raise CalibrationError(f"Duplicate condition/task/trial identity: {key!r}")
        result[key] = record
    return result


def _output_fingerprint(row: Mapping[str, Any]) -> str:
    """Bind labels to the exact answer and tool trace that the reviewer saw."""
    outcome = row.get("outcome", row)
    if not isinstance(outcome, Mapping):
        raise CalibrationError("outcome must be an object")
    material = {"answer": outcome.get("answer"), "trajectory": outcome.get("trajectory", [])}
    try:
        encoded = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, OverflowError) as exc:
        raise CalibrationError("Review output must contain valid finite JSON") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _dimensions(dimensions: Sequence[str] | Mapping[str, Mapping[str, Any]]) -> dict[str, dict]:
    if isinstance(dimensions, Mapping):
        source = list(dimensions.items())
    elif isinstance(dimensions, Sequence) and not isinstance(dimensions, (str, bytes)):
        source = [(name, {}) for name in dimensions]
    else:
        raise CalibrationError("dimensions must be a list of names or an object of specifications")
    result = {}
    for name, raw in source:
        if not isinstance(name, str) or not name.strip() or name in result:
            raise CalibrationError("Dimension names must be nonempty and unique")
        if not isinstance(raw, Mapping):
            raise CalibrationError(f"Specification for {name} must be an object")
        kind = raw.get("kind", "numeric")
        if not isinstance(kind, str) or kind not in {"numeric", "binary", "categorical"}:
            raise CalibrationError(f"Unknown dimension kind for {name}")
        if kind == "numeric":
            low, high, tolerance = raw.get("minimum", 0.0), raw.get("maximum", 1.0), raw.get("agreement_tolerance", 0.0)
            if any(isinstance(value, bool) or _number(value) is None for value in (low, high, tolerance)) or low >= high or tolerance < 0:
                raise CalibrationError(f"Invalid bounds/tolerance for {name}")
            result[name] = {"kind": kind, "minimum": float(low), "maximum": float(high), "agreement_tolerance": float(tolerance)}
        else:
            labels = [0, 1] if kind == "binary" else raw.get("labels")
            if not isinstance(labels, list) or len(labels) < 2:
                raise CalibrationError(f"{name} needs at least two categorical labels")
            if any(not isinstance(label, (str, int, bool)) or isinstance(label, str) and not label for label in labels):
                raise CalibrationError(f"{name} labels must be nonempty strings or integers")
            if len(set(labels)) != len(labels):
                raise CalibrationError(f"Duplicate categorical labels for {name}")
            result[name] = {"kind": kind, "labels": list(labels)}
    if not result:
        raise CalibrationError("At least one dimension is required")
    return result


def _valid_label(value: Any, spec: Mapping[str, Any]) -> bool:
    if spec["kind"] == "numeric":
        number = _number(value)
        return number is not None and spec["minimum"] <= number <= spec["maximum"]
    if spec["kind"] == "binary":
        return isinstance(value, Real) and _number(value) in (0.0, 1.0)
    return isinstance(value, (str, int, bool)) and value in spec["labels"]


def export_review_template(
    records: Sequence[Mapping[str, Any]],
    *,
    dimensions: Sequence[str] | Mapping[str, Mapping[str, Any]],
    dataset_fingerprint: str | None = None,
    source_kind: str = "human_review",
) -> dict[str, Any]:
    """Create a blinded, unlabeled review document; never generate human labels.

    A list of dimension names defaults to numeric [0,1] with exact agreement.
    For other scales use {name: {kind: numeric, minimum, maximum,
    agreement_tolerance}}, {name: {kind: binary}}, or
    {name: {kind: categorical, labels: [...]}}.

    A reviewer fills reference_scores, reviewer and human_reviewed=True for
    human_review items. Synthetic program checks use source_kind equal to
    synthetic_program_check, reference_scores, and human_reviewed=False.
    Judge scores are deliberately omitted from the review template.
    """
    if not isinstance(source_kind, str) or source_kind not in {"human_review", "synthetic_program_check"}:
        raise CalibrationError("source_kind must be human_review or synthetic_program_check")
    specs = _dimensions(dimensions)
    indexed = _index(records)
    if not indexed:
        raise CalibrationError("Cannot export an empty review set")
    observed = {row.get("dataset_fingerprint") for row in indexed.values() if isinstance(row.get("dataset_fingerprint"), str)}
    if dataset_fingerprint is None and len(observed) == 1:
        dataset_fingerprint = next(iter(observed))
    if not isinstance(dataset_fingerprint, str) or not dataset_fingerprint.strip():
        raise CalibrationError("A dataset_fingerprint is required")
    items = []
    for row in indexed.values():
        if row.get("dataset_fingerprint") != dataset_fingerprint:
            raise CalibrationError("All review records must match dataset_fingerprint")
        fingerprint = row.get("task_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise CalibrationError("Each review record needs task_fingerprint")
        outcome = row.get("outcome", row)
        if not isinstance(outcome, Mapping):
            raise CalibrationError("outcome must be an object")
        items.append({
            "condition": row.get("condition"), "task_id": row["task_id"], "trial_id": row["trial_id"],
            "task_fingerprint": fingerprint, "output_fingerprint": _output_fingerprint(row),
            "experiment_id": row.get("experiment_id"),
            "review_material": {
                "task": copy.deepcopy(row.get("task", row.get("prompt"))),
                "answer": copy.deepcopy(outcome.get("answer")),
                "trajectory": copy.deepcopy(outcome.get("trajectory", [])),
            },
            "reference_scores": {name: None for name in specs},
            "human_reviewed": False, "reviewer": None, "notes": "",
        })
    return {
        "schema_version": 1, "document_type": "judge_calibration_reference",
        "source_kind": source_kind, "dataset_fingerprint": dataset_fingerprint,
        "dimensions": specs, "items": items,
        "instructions": (
            "Unlabeled template. Fill reference_scores, reviewer and human_reviewed=true only after a person reviews each item. "
            "Leave unavailable labels null; do not convert model-generated labels into human gold."
            if source_kind == "human_review" else
            "Synthetic program-check template. Fill deterministic reference_scores; keep human_reviewed=false. These are not human gold labels."
        ),
    }


def evaluate_calibration(
    judge_records: Sequence[Mapping[str, Any]],
    gold_document: Mapping[str, Any],
    *,
    dimensions: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate saved judge predictions against labeled references, offline.

    All planned reference items remain in coverage denominators. Invalid
    reference labels/provenance and conflicting fingerprints raise; missing
    judges, failed judges, missing lineage, and unlabeled references are
    counted explicitly instead of becoming a disagreement or disappearing.
    Output fingerprints bind references to the exact reviewed answer/trace.
    Agreement is exact for categorical/binary dimensions and within the
    declared tolerance for numeric dimensions. Confusion matrices use gold
    rows and judge columns. Cohen's unweighted kappa is undefined when chance
    agreement is one (reported null). MAE applies only to numeric/binary scores.

    A human_review report reflects user-attested reference provenance. It does
    not certify label quality, sampling representativeness or human calibration.
    """
    if not isinstance(gold_document, Mapping):
        raise CalibrationError("gold_document must be an object")
    if type(gold_document.get("schema_version")) is not int or gold_document.get("schema_version") != 1 or gold_document.get("document_type") != "judge_calibration_reference":
        raise CalibrationError("Unsupported calibration reference schema")
    source_kind = gold_document.get("source_kind")
    if not isinstance(source_kind, str) or source_kind not in {"human_review", "synthetic_program_check"}:
        raise CalibrationError("Reference provenance must explicitly be human_review or synthetic_program_check")
    fingerprint = gold_document.get("dataset_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise CalibrationError("Reference dataset_fingerprint is required")
    specs = _dimensions(gold_document.get("dimensions"))
    if dimensions is not None and (not isinstance(dimensions, Sequence) or isinstance(dimensions, (str, bytes))):
        raise CalibrationError("Selected dimensions must be a sequence of names")
    selected = list(specs) if dimensions is None else list(dimensions)
    if not selected or any(not isinstance(name, str) for name in selected) or len(selected) != len(set(selected)) or any(name not in specs for name in selected):
        raise CalibrationError("Selected dimensions must be nonempty, unique and declared")
    raw_items = gold_document.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise CalibrationError("Reference items must be a nonempty list")
    references, predictions = _index(raw_items), _index(judge_records)
    diagnostics = []
    for key, row in predictions.items():
        if key not in references:
            diagnostics.append({"code": "unexpected_judge_record", "condition": key[0], "task_id": key[1], "trial_id": key[2]})
    # Validate all supplied gold labels, including dimensions not selected.
    for key, ref in references.items():
        task_fingerprint = ref.get("task_fingerprint")
        if not isinstance(task_fingerprint, str) or not task_fingerprint.strip():
            raise CalibrationError(f"Reference task_fingerprint required for {key!r}")
        output_fingerprint = ref.get("output_fingerprint")
        if not isinstance(output_fingerprint, str) or not output_fingerprint.strip():
            raise CalibrationError(f"Reference output_fingerprint required for {key!r}")
        scores = ref.get("reference_scores")
        if not isinstance(scores, Mapping) or any(name not in specs for name in scores):
            raise CalibrationError(f"Invalid/unknown reference dimensions for {key!r}")
        if not isinstance(ref.get("human_reviewed"), bool):
            raise CalibrationError(f"human_reviewed must be a boolean for {key!r}")
        has_labels = any(value is not None for value in scores.values())
        if source_kind == "human_review" and has_labels:
            if ref.get("human_reviewed") is not True or not isinstance(ref.get("reviewer"), str) or not ref["reviewer"].strip():
                raise CalibrationError(f"Human labels require human_reviewed=true and reviewer for {key!r}")
        if source_kind == "synthetic_program_check" and ref.get("human_reviewed"):
            raise CalibrationError("Synthetic fixtures cannot claim human review")
        for name, value in scores.items():
            if value is not None and not _valid_label(value, specs[name]):
                raise CalibrationError(f"Invalid reference label for {key!r}/{name}")
        prediction = predictions.get(key)
        if prediction is not None:
            if _output_fingerprint(prediction) != output_fingerprint:
                raise CalibrationError(f"Mismatched reviewed output for {key!r}")
            for field, expected in (("dataset_fingerprint", fingerprint), ("task_fingerprint", task_fingerprint)):
                value = prediction.get(field)
                if value is not None and value != expected:
                    raise CalibrationError(f"Mismatched {field} for {key!r}")

    reports = {}
    for name in selected:
        spec = specs[name]
        counts = Counter()
        pairs = []
        excluded = []
        for key, ref in references.items():
            reference = ref["reference_scores"].get(name)
            prediction = predictions.get(key)
            if reference is None:
                reason = "unlabeled"
            elif prediction is None:
                reason = "missing_judge_record"
            elif any(not isinstance(prediction.get(field), str) or not prediction[field].strip() for field in ("dataset_fingerprint", "task_fingerprint")):
                reason = "unverified_lineage"
            else:
                outcome = prediction.get("outcome", prediction)
                execution_failed = (
                    prediction.get("execution_error") is not None
                    or prediction.get("status") in ("failed", "error", "execution_error", "blocked", "cancelled", "timeout", "pending", "running")
                    or isinstance(outcome, Mapping) and outcome.get("stop_reason") == "execution_error"
                )
                scores = prediction.get("scores", {})
                states = prediction.get("metric_statuses", prediction.get("statuses", {}))
                if execution_failed:
                    reason = "execution_error"
                elif not isinstance(scores, Mapping) or not isinstance(states, Mapping):
                    reason = "invalid_judge_score"
                elif states.get(name) not in (None, "valid"):
                    reason = {"not_applicable": "not_applicable", "missing": "missing_judge_score", "blocked": "blocked", "execution_error": "execution_error"}.get(states.get(name), "judge_failed") if isinstance(states.get(name), str) else "judge_failed"
                elif name not in scores or scores[name] is None:
                    reason = "missing_judge_score"
                elif not _valid_label(scores[name], spec):
                    reason = "invalid_judge_score"
                else:
                    reason = "valid"
                    pairs.append((reference, scores[name]))
            counts[reason] += 1
            if reason != "valid":
                excluded.append({"condition": key[0], "task_id": key[1], "trial_id": key[2], "reason": reason})
        count = len(pairs)
        labeled = len(references) - counts["unlabeled"]
        errors = [abs(float(predicted) - float(reference)) for reference, predicted in pairs] if spec["kind"] != "categorical" else []
        if not all(math.isfinite(error) for error in errors):
            raise CalibrationError("Calibration score differences must remain finite")
        agreement = (
            sum(error <= spec["agreement_tolerance"] for error in errors)
            if spec["kind"] == "numeric" else sum(reference == predicted for reference, predicted in pairs)
        )
        confusion = None
        kappa = None
        kappa_reason = "numeric_dimension" if spec["kind"] == "numeric" else "no_valid_pairs"
        if spec["kind"] != "numeric":
            labels = spec["labels"]
            matrix = [[0 for _ in labels] for _ in labels]
            for reference, predicted in pairs:
                matrix[labels.index(reference)][labels.index(predicted)] += 1
            confusion = {"labels": labels, "rows": "reference", "columns": "judge", "counts": matrix}
            if count:
                expected_agreement = math.fsum(sum(matrix[i]) / count * sum(row[i] for row in matrix) / count for i in range(len(labels)))
                if expected_agreement < 1.0:
                    kappa = (agreement / count - expected_agreement) / (1.0 - expected_agreement)
                    kappa_reason = None
                else:
                    kappa_reason = "chance_agreement_is_one"
        reports[name] = {
            "kind": spec["kind"], "planned": len(references), "labeled": labeled,
            "valid": count, "coverage": count / len(references),
            "label_coverage": labeled / len(references),
            "judge_coverage_of_labeled": count / labeled if labeled else None,
            "counts_by_status": dict(sorted(counts.items())),
            "mae": math.fsum(error / count for error in errors) if errors else None,
            "agreement": agreement / count if count else None,
            "agreement_tolerance": spec.get("agreement_tolerance", 0),
            "confusion_matrix": confusion, "cohens_kappa": kappa,
            "kappa_unavailable_reason": kappa_reason, "excluded": excluded,
        }
    return {
        "schema_version": 1, "reference_kind": source_kind,
        "provenance_verification": "user_attested" if source_kind == "human_review" else "synthetic_program_check_only",
        "dataset_fingerprint": fingerprint, "planned_items": len(references),
        "observed_judge_records": len(predictions),
        "unexpected_judge_records": len(predictions.keys() - references.keys()),
        "dimensions": reports, "diagnostics": diagnostics,
        "limitations": [
            "Exported templates contain no human labels until a reviewer supplies them.",
            "Human-review provenance is user-attested; software cannot verify who produced labels.",
            "Synthetic fixtures verify calculations only and are not human gold or evidence of human calibration.",
            "This report does not establish representative sampling, label reliability, or external validity.",
        ],
    }
