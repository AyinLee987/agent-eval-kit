import copy
import json
from pathlib import Path
import sqlite3

import pytest

from agent_eval import experiments as experiments
from agent_eval.dataset_scoring import PublicDatasetScorer, public_case_prompt
from agent_eval.execution import encode_outcome
from agent_eval.score_reporting import score_outcome
from agent_eval.types import AgentOutcome
from benchmarks.public_data.summarize_run import main, render_markdown, summarize_run


def _json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def _task(dataset, task_id):
    case = {"id": task_id, "dataset": dataset, "split": "dev", "family_id": task_id,
            "question": "What is two plus two?", "answer": "4", "context": [], "supporting_facts": []}
    return {"id": task_id, "prompt": public_case_prompt(case), "group_id": task_id, "public_case": case}


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "source_snapshot", lambda roots: [{"root": "fixture", "sha256": "fixture"}])
    config = {"trials": 1, "conditions": [{"id": "flash", "agent": "tests.experiment_fixtures:build_agent",
               "outcome_adapter": "tests.experiment_fixtures:adapt"}],
              "scorers": [{"factory": "agent_eval.dataset_scoring:PublicDatasetScorer"}]}
    manifest = {"schema_version": 1, "model": "deepseek-v4-flash", "condition": "flash", "split": "dev",
                "trials": 1, "started_at": "2026-01-01T00:00:00Z", "finished_at": "2026-01-01T00:01:00Z",
                "datasets": [{"dataset": "gsm8k", "expected_ids": ["g1", "g2"], "shards": ["gsm8k/one", "gsm8k/two"]},
                             {"dataset": "hotpotqa", "expected_ids": ["h1"], "shards": ["hotpotqa/one"]}]}
    _json(tmp_path / "run_manifest.json", manifest)
    for dataset, task_id, shard in [("gsm8k", "g1", "gsm8k/one"), ("gsm8k", "g2", "gsm8k/two"),
                                    ("hotpotqa", "h1", "hotpotqa/one")]:
        experiments.create_experiment(tmp_path / shard, [_task(dataset, task_id)], config)
    return tmp_path


def _complete(root, shard, *, answer="4", success=True, error=None, usage=None):
    directory = root / shard
    manifest = json.loads((directory / "manifest.json").read_text())
    task = json.loads((directory / "tasks.json").read_text())[0]
    outcome = AgentOutcome(answer, success, "completed" if success else "failed", 1, 12)
    scores, metric_statuses, scorer_errors = score_outcome([PublicDatasetScorer()], task, outcome,
                                                         unavailable="execution_error" if error else None)
    encoded = encode_outcome(outcome)
    encoded["metadata"] = {"provider": "deepseek", "model": "deepseek-v4-flash", "usage_complete": not bool(error),
                           "api_calls": [{"usage": usage if usage is not None else
                                          {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                                          "error": error, "elapsed_seconds": 0.5}]}
    record = {"schema_version": 1, "scoring_version": 2, "experiment_id": manifest["experiment_id"],
              "condition": "flash", "task_id": task["id"], "trial_id": 0,
              "group_id": task["group_id"], "task_fingerprint": manifest["tasks"][0]["fingerprint"],
              "dataset_fingerprint": manifest["dataset_fingerprint"], "status": "completed",
              "outcome": encoded, "tokens_observed": not bool(error), "execution_error": error,
              "scores": scores, "metric_statuses": metric_statuses, "scorer_errors": scorer_errors,
              "elapsed_seconds": 1.5}
    with sqlite3.connect(directory / "runs.sqlite3") as con:
        con.execute("UPDATE trials SET state='completed', record=?", (json.dumps(record),))
    return record


def _rewrite_manifest(root, shard, change):
    path = root / shard / "manifest.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    manifest.pop("manifest_fingerprint")
    manifest["manifest_fingerprint"] = experiments.fingerprint(manifest)
    _json(path, manifest)


def test_summary_preserves_partial_denominator_and_zero_answers(run):
    _complete(run, "gsm8k/one")
    _complete(run, "gsm8k/two", answer="5")
    summary = summarize_run(run)
    assert summary["partial"] is True
    assert summary["totals"]["planned"] == 3
    assert summary["totals"]["completed"] == 2
    assert summary["totals"]["failed"] == 0
    gsm = summary["datasets"]["gsm8k"]
    metric = gsm["metrics"]["public_numeric_exact"]
    assert metric["mean"] == 0.5
    assert metric["wilson_95"]["low"] < 0.5 < metric["wilson_95"]["high"]
    assert gsm["incorrect_answers"][0]["task_id"] == "g2"
    assert summary["datasets"]["hotpotqa"]["metrics"]["public_answer_em"]["missing"] == 1
    assert summary["totals"]["cost_unknown_runs"] == 1
    assert summary["wall_seconds"] == 60
    assert gsm["api_usage"]["prompt_tokens"] == 20
    assert gsm["api_usage"]["completion_tokens"] == 4
    assert gsm["api_usage"]["total_tokens"] == 24


def test_failures_keep_full_errors_and_unknown_usage(run):
    failure = {"stage": "request", "type": "TimeoutError", "message": "Request timed out"}
    _complete(run, "gsm8k/one", success=False, error=failure,
              usage={"prompt_tokens": None, "completion_tokens": None, "total_tokens": None})
    summary = summarize_run(run)
    dataset = summary["datasets"]["gsm8k"]
    assert dataset["completed"] == 1 and dataset["failed"] == 1
    assert dataset["execution_errors"] == 1
    assert dataset["scorer_error_runs"] == 0
    assert dataset["unavailable_scoring_runs"] == 1
    assert dataset["metrics"]["public_numeric_exact"]["mean"] is None
    assert dataset["metrics"]["public_numeric_exact"]["execution_error"] == 1
    assert dataset["errors"][0]["execution_error"] == failure
    assert dataset["api_call_errors"][0]["error"] == failure
    assert dataset["api_usage"]["total_tokens"] is None
    assert dataset["api_usage"]["unknown_calls_by_field"]["total_tokens"] == 1
    assert dataset["cost_unknown_runs"] == 2


def test_summaries_do_not_write_shards_or_read_stale_reports(run):
    _complete(run, "gsm8k/one")
    report = run / "gsm8k/one/report.json"
    report.write_text("This stale report is not JSON", encoding="utf-8")
    manifest_before = (run / "gsm8k/one/manifest.json").read_bytes()
    summary = summarize_run(run)
    assert summary["totals"]["completed"] == 1
    assert report.read_text() == "This stale report is not JSON"
    assert (run / "gsm8k/one/manifest.json").read_bytes() == manifest_before
    assert not (run / "gsm8k/one/.runner.lock").exists()


def test_missing_future_shard_is_explicit_partial(run):
    path = run / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["datasets"][0]["shards"].append("gsm8k/future")
    manifest["datasets"][0]["expected_ids"].append("g3")
    _json(path, manifest)
    summary = summarize_run(run)
    assert summary["absent_shards"] == ["gsm8k/future"]
    assert summary["datasets"]["gsm8k"]["unplanned_expected"] == [{"task_id": "g3", "trial_id": 0}]


@pytest.mark.parametrize("field", ["task_id", "scoring_version", "condition", "dataset_fingerprint"])
def test_completed_record_identity_is_checked(run, field):
    record = _complete(run, "gsm8k/one")
    record[field] = "changed"
    with sqlite3.connect(run / "gsm8k/one/runs.sqlite3") as con:
        con.execute("UPDATE trials SET record=?", (json.dumps(record),))
    with pytest.raises(ValueError, match="identity/version"):
        summarize_run(run)


def test_different_scorer_protocol_is_refused(run):
    _rewrite_manifest(run, "gsm8k/two", lambda doc: doc["measurement_protocol"].update(scoring_version=3))
    with pytest.raises(ValueError, match="protocol differs"):
        summarize_run(run)


def test_duplicate_task_across_shards_is_refused(run):
    path = run / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["datasets"][0]["shards"].append("gsm8k/duplicate")
    _json(path, manifest)
    original = run / "gsm8k/one"
    duplicate = run / "gsm8k/duplicate"
    duplicate.mkdir()
    for filename in ("manifest.json", "tasks.json", "runs.sqlite3"):
        (duplicate / filename).write_bytes((original / filename).read_bytes())
    with pytest.raises(ValueError, match="Duplicate dataset:task_id"):
        summarize_run(run)


@pytest.mark.parametrize("relative", ["../escape", "gsm8k/one"])
def test_escaping_and_duplicate_shard_paths_are_refused(run, relative):
    path = run / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["datasets"][0]["shards"].append(relative)
    _json(path, manifest)
    with pytest.raises(ValueError, match="escapes|Duplicate shard"):
        summarize_run(run)


def test_test_split_never_mixes_into_dev(run):
    directory = run / "gsm8k/one"
    tasks = json.loads((directory / "tasks.json").read_text())
    tasks[0]["public_case"]["split"] = "test"
    _json(directory / "tasks.json", tasks)
    def change(manifest):
        manifest["dataset_fingerprint"] = experiments.fingerprint(tasks)
        manifest["tasks"][0]["fingerprint"] = experiments.fingerprint(tasks[0])
    _rewrite_manifest(run, "gsm8k/one", change)
    with pytest.raises(ValueError, match="development case"):
        summarize_run(run)


def test_sqlite_schedule_must_match_manifest(run):
    with sqlite3.connect(run / "gsm8k/one/runs.sqlite3") as con:
        con.execute("DELETE FROM trials")
    with pytest.raises(ValueError, match="planned rows do not match"):
        summarize_run(run)


def test_model_identity_and_nonfinite_usage_are_rejected(run):
    record = _complete(run, "gsm8k/one")
    record["outcome"]["metadata"]["model"] = "other"
    with sqlite3.connect(run / "gsm8k/one/runs.sqlite3") as con:
        con.execute("UPDATE trials SET record=?", (json.dumps(record),))
    with pytest.raises(ValueError, match="Observed model differs"):
        summarize_run(run)
    record["outcome"]["metadata"]["model"] = "deepseek-v4-flash"
    record["outcome"]["metadata"]["api_calls"][0]["usage"]["total_tokens"] = -1
    with sqlite3.connect(run / "gsm8k/one/runs.sqlite3") as con:
        con.execute("UPDATE trials SET record=?", (json.dumps(record),))
    with pytest.raises(ValueError, match="integer"):
        summarize_run(run)


def test_cli_writes_complete_summary_and_markdown(run):
    _complete(run, "gsm8k/one")
    _complete(run, "gsm8k/two")
    _complete(run, "hotpotqa/one", answer='{"answer":"4","supporting_facts":[]}')
    output, markdown = run / "summary.json", run / "report.md"
    result = main([str(run), "--out", str(output), "--markdown", str(markdown)])
    assert result["partial"] is False
    assert result["status"] == "completed"
    assert json.loads(output.read_text())["totals"]["completed"] == 3
    assert "public_answer_f1" in markdown.read_text(encoding="utf-8")
    assert "No committed" in render_markdown(result)


def test_multiple_trials_do_not_get_independent_bernoulli_intervals(run):
    # Direct helper exercises the statistical boundary without executing agents.
    from benchmarks.public_data.summarize_run import _dataset_summary
    record = _complete(run, "gsm8k/one")
    record["shard"] = "gsm8k/one"
    repeated = copy.deepcopy(record)
    repeated["trial_id"] = 1
    summary = _dataset_summary("gsm8k", {("g1", 0), ("g1", 1)}, {("g1", 0), ("g1", 1)},
                               [record, repeated], {"completed": 2}, {"g1": _task("gsm8k", "g1")}, 2)
    assert summary["metrics"]["public_numeric_exact"]["wilson_95"] is None
