import copy
import json
from contextlib import nullcontext

import pytest

from agent_eval import experiments


@pytest.fixture
def saved_experiment(monkeypatch):
    tasks = [{"id": "t1", "prompt": "one"}, {"id": "t2", "prompt": "two"}]
    manifest = {
        "experiment_id": "experiment-1", "manifest_fingerprint": "manifest-1",
        "dataset_fingerprint": "dataset-1",
        "config": {"conditions": [{"id": "A"}, {"id": "B"}], "trials": 2, "source_roots": [], "timeout_seconds": 120},
        "tasks": [{"id": task["id"], "group_id": task["id"], "fingerprint": task["id"]} for task in tasks],
        "schedule": [{"condition": condition, "task_id": task["id"], "trial_id": trial} for condition in ("A", "B") for task in tasks for trial in range(2)],
    }
    records = []
    calls = []
    monkeypatch.setattr(experiments, "_load", lambda directory: (manifest, tasks))
    monkeypatch.setattr(experiments, "_lock", lambda directory: nullcontext())
    monkeypatch.setattr(experiments, "read_records", lambda directory: copy.deepcopy(records))
    monkeypatch.setattr(experiments, "normalize_config", lambda config, tasks: config)
    monkeypatch.setattr(experiments, "source_snapshot", lambda roots: [])
    def score(task, record, config):
        calls.append((record["condition"], record["task_id"], record["trial_id"]))
        record.update(scores={"quality": record.pop("fixture_score")}, metric_statuses={"quality": "valid"}, scorer_errors=[])
        return record
    monkeypatch.setattr(experiments, "_score", score)
    # Rescoring may call only the scoring wrapper; never an agent executor.
    monkeypatch.setattr(experiments, "execute", lambda *a, **k: pytest.fail("Agent execution is forbidden in this test"))
    return records, calls


def record(condition, task="t1", trial=0, score=1):
    return {"condition": condition, "task_id": task, "trial_id": trial,
            "outcome": {"success": True, "tokens": 10}, "execution_error": None,
            "elapsed_seconds": 1, "scorer_errors": [], "fixture_score": score}


def test_partial_rescore_preserves_planned_denominators_per_condition(saved_experiment, tmp_path):
    records, calls = saved_experiment
    records.extend([record("A", score=1), record("B", score=0)])
    before = copy.deepcopy(records)
    output = tmp_path / "rescored.json"
    report = experiments.rescore_experiment(tmp_path / "original", output)
    assert len(calls) == 2
    assert "metrics" not in report
    assert report["conditions"]["A"]["metrics"]["quality"] == {
        "mean": 1, "planned": 4, "valid": 1, "not_applicable": 0,
        "failed": 0, "execution_error": 0, "blocked": 0, "missing": 3, "coverage": .25,
    }
    assert report["conditions"]["B"]["metrics"]["quality"]["mean"] == 0
    assert report["conditions"]["B"]["metrics"]["quality"]["coverage"] == .25
    assert report["conditions"]["A"]["missing_records"] == 3
    assert len(report["planned_keys"]) == 4
    assert records == before
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_entirely_missing_condition_still_reports_known_metric_as_missing(saved_experiment, tmp_path):
    records, _ = saved_experiment
    records.append(record("A"))
    report = experiments.rescore_experiment(tmp_path / "original", tmp_path / "rescored.json")
    missing = report["conditions"]["B"]
    assert missing["planned"] == missing["missing_records"] == 4
    assert missing["completed"] == 0
    assert missing["metrics"]["quality"]["valid"] == 0
    assert missing["metrics"]["quality"]["missing"] == 4
    assert missing["metrics"]["quality"]["coverage"] == 0
    assert missing["metrics"]["quality"]["mean"] is None


def test_empty_experiment_rescore_does_not_claim_full_coverage(saved_experiment, tmp_path):
    _, calls = saved_experiment
    report = experiments.rescore_experiment(tmp_path / "original", tmp_path / "rescored.json")
    assert calls == []
    assert report["records"] == []
    assert all(c["completed"] == 0 and c["missing_records"] == 4 for c in report["conditions"].values())
    assert all(c["metrics"] for c in report["conditions"].values())
    assert all(metric["coverage"] == 0 and metric["missing"] == 4 and metric["mean"] is None for c in report["conditions"].values() for metric in c["metrics"].values())
