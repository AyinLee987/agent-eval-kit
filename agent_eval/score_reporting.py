"""Shared score validation and explicit missing-score accounting.

The legacy numeric averages remain available, but reports always accompany them
with denominators and failures. A judge outage is not an incorrect agent answer.
"""

from __future__ import annotations

import math
import json
import os
import tempfile
from numbers import Number
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


def numeric(value: Any) -> float | None:
    if not isinstance(value, Number):
        return None
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def error_record(stage: str, exc: Exception) -> Dict[str, str]:
    return {"stage": stage, "type": type(exc).__name__, "message": str(exc)}


def score_outcome(scorers: Sequence[Any], task: dict, outcome: Any, *,
                  unavailable: str | None = None):
    """Return flat scores, per-metric states, and per-scorer errors.

    Custom scorers should declare ``metric_names`` so all-failed runs still
    have known metric denominators. Undeclared numeric keys remain supported.
    Text metadata (e.g. rationale) is retained and never averaged.
    """
    scores, states, errors = {}, {}, []
    owners = {}
    for scorer in scorers:
        name = getattr(scorer, "name", type(scorer).__name__)
        declared = tuple(getattr(scorer, "metric_names", ()))
        if unavailable:
            for key in declared:
                scores[key] = None
                states[key] = unavailable
            errors.append({"scorer": name, "stage": unavailable,
                           "type": "UnavailableOutcome", "message": unavailable})
            continue
        keys = set(declared)
        try:
            result = scorer.score(task, outcome)
            if not isinstance(result, dict) or any(not isinstance(k, str) for k in result):
                raise ValueError("Scorer must return a dict with string keys.")
            keys.update(k for k, v in result.items() if v is None or isinstance(v, Number))
            # Validate metadata as well: a NaN nested in a rationale payload
            # must not break serialization after all tasks have completed.
            json.dumps(result, allow_nan=False)
            missing = set(declared) - result.keys()
            if missing:
                raise ValueError(f"Missing declared metrics: {sorted(missing)}.")
            collisions = result.keys() & owners.keys()
            if collisions:
                keys.update(k for k in collisions if k in states)
                raise ValueError(f"Scorers produced duplicate keys: {sorted(collisions)}.")
            for key in keys:
                if result.get(key) is not None and numeric(result[key]) is None:
                    raise ValueError(f"Metric {key} is not a finite numeric score.")
            # Validate the entire scorer response before committing any values.
            scores.update(result)
            owners.update({key: name for key in result})
            for key in result:
                if key in keys:
                    states[key] = "not_applicable" if result[key] is None else "valid"
        except Exception as exc:
            errors.append({"scorer": name, **error_record("score", exc)})
            for key in dict.fromkeys([*declared, *sorted(keys - set(declared))]):
                scores[key] = None
                states[key] = "failed"
                owners[key] = name
    return scores, states, errors


def metric_summary(rows: Iterable[tuple[dict, dict]]) -> Dict[str, dict]:
    """Observed-score means and coverage; missing values never vanish silently.

    No range or value is imputed for a missing score. Generic scorers can emit
    unbounded values such as elapsed seconds as well as unit-scale quality.
    """
    rows = list(rows)
    keys = dict.fromkeys(key for scores, states in rows for key in (
        list(states) + [k for k, v in scores.items() if v is None or isinstance(v, Number)]
    ))
    summaries = {}
    for key in keys:
        counts = dict.fromkeys(("valid", "not_applicable", "failed",
                                "execution_error", "blocked", "missing"), 0)
        values = []
        for scores, states in rows:
            value = numeric(scores.get(key))
            state = states.get(key)
            if state is None:
                state = ("missing" if key not in scores else "not_applicable"
                         if scores[key] is None else "valid" if value is not None else "failed")
            if state == "valid" and value is None:
                state = "failed"
            if state not in counts:
                state = "failed"
            counts[state] += 1
            if state == "valid":
                values.append(value)
        denominator = len(rows) - counts["not_applicable"]
        summaries[key] = {
            "planned": len(rows), **counts,
            "mean": math.fsum(value / len(values) for value in values) if values else None,
            "coverage": len(values) / denominator if denominator else None,
        }
    return summaries


def average(rows: Iterable[tuple[dict, dict]]) -> Dict[str, float]:
    return {key: detail["mean"] for key, detail in metric_summary(rows).items()
            if detail["mean"] is not None}


def coverage_line(key: str, detail: dict) -> str:
    mean = "unavailable" if detail["mean"] is None else f"{detail['mean']:.3f}"
    return (f"avg {key}: {mean} (valid={detail['valid']}/{detail['planned']}, "
            f"not_applicable={detail['not_applicable']}, failed={detail['failed']}, "
            f"execution_error={detail['execution_error']}, blocked={detail['blocked']}, "
            f"missing={detail['missing']})")


def write_json_report(path: Any, payload: dict) -> None:
    """Validate before touching an existing report; replace it atomically."""
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    target = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def validate_tasks(tasks: Any, *, conversations: bool = False) -> list:
    if not isinstance(tasks, list):
        raise ValueError("Task set must be a list.")
    ids = set()
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("id"), str) or not task["id"]:
            raise ValueError("Every task needs a nonempty string id.")
        if task["id"] in ids:
            raise ValueError(f"Duplicate task id: {task['id']}.")
        ids.add(task["id"])
        turns = task.get("turns") if conversations else [task]
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"Task {task['id']} needs nonempty turns.")
        for turn in turns:
            if not isinstance(turn, dict) or not isinstance(turn.get("prompt"), str):
                raise ValueError(f"Task {task['id']} needs string prompts.")
    return tasks
