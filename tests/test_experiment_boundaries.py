import json

import pytest

from agent_eval import experiments as e
from agent_eval.cli import main
from agent_eval.comparison import _cost
from tests.test_experiments import config


@pytest.fixture(autouse=True)
def stable_source(monkeypatch):
    monkeypatch.setattr(e, "source_snapshot", lambda roots: [{"root": "fixture", "sha256": "fixture"}])


def test_corrupt_worker_result_is_terminal_failure_not_automatic_retry(tmp_path):
    cfg = config(tmp_path, trials=1)
    cfg["conditions"] = cfg["conditions"][:1]
    cfg["conditions"][0]["config"]["corrupt_result"] = True
    directory = tmp_path / "exp"
    e.create_experiment(directory, [{"id": "x", "prompt": "x"}], cfg)
    report = e.run_experiment(directory)
    assert report["states"] == {"completed": 1}
    record = report["records"][0]
    assert record["execution_error"]["stage"] == "result"
    assert record["tokens_observed"] is False
    assert _cost(record, "tokens") is None
    e.run_experiment(directory)
    assert (tmp_path / "a.log").read_text().splitlines() == ["x"]


def test_entire_scoring_worker_timeout_has_failed_metric_denominator(tmp_path):
    cfg = config(tmp_path, trials=1, score_timeout_seconds=.5,
                 scorers=[{"factory": "tests.experiment_fixtures:SlowScorer", "metric_names": ["slow"]}])
    cfg["conditions"] = cfg["conditions"][:1]
    directory = tmp_path / "exp"
    e.create_experiment(directory, [{"id": "x", "prompt": "x"}], cfg)
    report = e.run_experiment(directory)
    metric = report["conditions"]["a"]["metrics"]["slow"]
    assert metric["failed"] == 1 and metric["coverage"] == 0
    assert report["records"][0]["scoring_error"]["type"] == "TimeoutError"
    assert report["records"][0]["execution_error"] is None


def test_compare_cli_rejects_mismatched_time_budget_before_inference(tmp_path, capsys):
    task = [{"id": "x", "prompt": "x"}]
    for name, timeout in (("a", 10), ("b", 20)):
        cfg = config(tmp_path, timeout_seconds=timeout)
        e.create_experiment(tmp_path / name, task, cfg)
    assert main(["compare", "--a", str(tmp_path / "a"), "--a-condition", "a", "--b", str(tmp_path / "b"),
                 "--b-condition", "b", "--metric", "answer_correct", "--allow-incomplete"]) == 2
    assert "time budgets differ" in capsys.readouterr().err


def test_unstarted_report_records_missing_metric_coverage(tmp_path):
    directory = tmp_path / "exp"
    e.create_experiment(directory, [{"id": "x", "prompt": "x"}], config(tmp_path, trials=2))
    report = e.export_report(directory)
    for condition in report["conditions"].values():
        metric = condition["metrics"]["answer_correct"]
        assert metric["mean"] is None and metric["missing"] == 2 and metric["coverage"] == 0


def test_rescore_cannot_replace_original_or_previous_output(tmp_path):
    directory = tmp_path / "exp"
    e.create_experiment(directory, [{"id": "x", "prompt": "x"}], config(tmp_path))
    with pytest.raises(ValueError):
        e.rescore_experiment(directory, directory / "new.json")
    output = tmp_path / "rescore.json"
    output.write_text("untouched")
    with pytest.raises(ValueError):
        e.rescore_experiment(directory, output)
    assert output.read_text() == "untouched"
