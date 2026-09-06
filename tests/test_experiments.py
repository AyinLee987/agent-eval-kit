import copy
import json
import os
from pathlib import Path
import sqlite3

import pytest

from agent_eval import experiments as e
from agent_eval.execution import execute
from agent_eval.cli import main


def config(tmp_path, **changes):
    base = {"trials": 2, "conditions": [{"id": name,
             "agent": "tests.experiment_fixtures:build_agent",
             "outcome_adapter": "tests.experiment_fixtures:adapt",
             "config": {"log_path": str(tmp_path / (name + ".log"))}}
            for name in ("a", "b")]}
    base.update(changes)
    return base


@pytest.fixture
def tasks():
    return [{"id": "one", "prompt": "compute", "expect_number": 42},
            {"id": "two", "prompt": "again", "expect_number": 42, "group_id": "same-family"}]


@pytest.fixture(autouse=True)
def stable_source(monkeypatch):
    # Independent edits elsewhere in the working tree cannot alter this fixture.
    monkeypatch.setattr(e, "source_snapshot", lambda roots: [{"root": "fixture", "sha256": "fixture"}])


def test_resume_runs_each_planned_trial_once_and_preserves_failures(tmp_path, tasks):
    cfg = config(tmp_path)
    cfg["conditions"][1]["config"]["fail"] = True
    directory = tmp_path / "experiment"
    manifest = e.create_experiment(directory, tasks, cfg)
    assert len(manifest["schedule"]) == 8
    report = e.run_experiment(directory, max_runs=3)
    assert report["states"] == {"completed": 3, "pending": 5}
    report = e.run_experiment(directory)
    assert report["states"] == {"completed": 8}
    assert report["conditions"]["b"]["execution_errors"] == 4
    for name in ("a", "b"):
        assert len((tmp_path / (name + ".log")).read_text().splitlines()) == 4
    saved = (directory / "report.json").read_bytes()
    assert e.run_experiment(directory) == report
    assert (directory / "report.json").read_bytes() == saved
    assert all(r["outcome"]["tokens"] == 7 for r in report["records"] if r["condition"] == "a")


def test_resume_scoring_does_not_execute_agent_again(tmp_path, tasks, monkeypatch):
    directory = tmp_path / "experiment"
    e.create_experiment(directory, tasks[:1], config(tmp_path, trials=1))
    original = e._score
    monkeypatch.setattr(e, "_score", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        e.run_experiment(directory)
    with e._connect(directory) as con:
        assert con.execute("SELECT state FROM trials WHERE position=0").fetchone()[0] == "executed"
    monkeypatch.setattr(e, "_score", original)
    e.run_experiment(directory)
    assert sum(len(p.read_text().splitlines()) for p in tmp_path.glob("*.log")) == 2


def test_progress_exception_does_not_uncomplete_trial(tmp_path, tasks):
    directory = tmp_path / "experiment"
    e.create_experiment(directory, tasks[:1], config(tmp_path, trials=1))
    with pytest.raises(BrokenPipeError):
        e.run_experiment(directory, progress=lambda row: (_ for _ in ()).throw(BrokenPipeError()))
    assert len(e.read_records(directory)) == 1


def test_changes_to_sources_or_saved_tasks_refuse_resume(tmp_path, tasks, monkeypatch):
    directory = tmp_path / "experiment"
    e.create_experiment(directory, tasks, config(tmp_path))
    monkeypatch.setattr(e, "source_snapshot", lambda roots: [{"root": "fixture", "sha256": "changed"}])
    with pytest.raises(ValueError, match="Source code changed"):
        e.run_experiment(directory)
    taskfile = directory / "tasks.json"
    changed = json.loads(taskfile.read_text())
    changed[0]["prompt"] = "changed"
    taskfile.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="task content changed"):
        e.read_records(directory)


def test_manifest_integrity_and_existing_directory(tmp_path, tasks):
    directory = tmp_path / "experiment"
    e.create_experiment(directory, tasks, config(tmp_path))
    with pytest.raises(FileExistsError):
        e.create_experiment(directory, tasks, config(tmp_path))
    path = directory / "manifest.json"
    document = json.loads(path.read_text())
    document["config"]["seed"] = 19
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="manifest was modified"):
        e.read_records(directory)


def test_abrupt_interruption_requires_reconciliation_and_live_worker_blocks(tmp_path, tasks):
    directory = tmp_path / "experiment"
    e.create_experiment(directory, tasks, config(tmp_path))
    with e._connect(directory) as con:
        e._save(con, 0, "running", None, os.getpid())
    with pytest.raises(RuntimeError, match="uncertain trial"):
        e.run_experiment(directory)
    with pytest.raises(RuntimeError, match="still alive"):
        e.run_experiment(directory, recover_interrupted=True)
    with e._connect(directory) as con:
        e._save(con, 0, "running", None, None)
    e.run_experiment(directory, recover_interrupted=True, max_runs=0)
    with e._connect(directory) as con:
        assert con.execute("SELECT state FROM trials WHERE position=0").fetchone()[0] == "interrupted"


def test_experiment_lock_prevents_second_owner(tmp_path):
    with e._lock(tmp_path):
        with pytest.raises(RuntimeError, match="another runner"):
            with e._lock(tmp_path):
                pass


@pytest.mark.parametrize("change", [{"trials": 0}, {"trials": True}, {"timeout_seconds": -1}, {"seed": False}, {"conditions": []}])
def test_invalid_config_rejected_before_creation(tmp_path, tasks, change):
    with pytest.raises(ValueError):
        e.create_experiment(tmp_path / "experiment", tasks, config(tmp_path, **change))
    assert not (tmp_path / "experiment").exists()


def test_credentials_refused_and_schedule_randomized_reproducibly(tmp_path, tasks):
    cfg = config(tmp_path)
    cfg["conditions"][0]["config"]["api_key"] = "fixture"
    with pytest.raises(ValueError, match="credentials"):
        e.create_experiment(tmp_path / "bad", tasks, cfg)
    one = e.create_experiment(tmp_path / "one", tasks, config(tmp_path))
    two = e.create_experiment(tmp_path / "two", tasks, config(tmp_path))
    assert one["schedule"] == two["schedule"]
    for position in range(0, len(one["schedule"]), 2):
        a, b = one["schedule"][position:position+2]
        assert (a["task_id"], a["trial_id"]) == (b["task_id"], b["trial_id"])
        assert a["condition"] != b["condition"]


def test_process_timeout_reaps_descendant_and_scorer_timeout_is_observable(tmp_path):
    pidfile = tmp_path / "child.pid"
    condition = config(tmp_path)["conditions"][0]
    condition["config"] = {"child_pid_path": str(pidfile)}
    result = execute({"kind": "agent", "condition": condition, "prompt": "start"}, timeout_seconds=2)
    assert result["worker_error"]["type"] == "TimeoutError"
    assert pidfile.exists(), "The fixture must reach the descendant creation before the deadline"
    pid = int(pidfile.read_text())
    if os.name == "nt":
        assert not e._alive(pid)
    else:
        # A killed orphan can remain a zombie briefly until init reaps it.
        status = Path(f"/proc/{pid}/stat")
        assert not status.exists() or status.read_text().split()[2] == "Z"
    result = execute({"kind": "score", "scorers": [{"factory": "tests.experiment_fixtures:SlowScorer"}],
                      "task": {"id": "x", "prompt": "x"},
                      "outcome": {"answer": "x", "success": True, "stop_reason": "finished", "tokens": 0, "steps": 0}},
                     timeout_seconds=1)
    assert result["worker_error"]["type"] == "TimeoutError"


def test_cli_plan_resume_compare_review_and_rescore(tmp_path, tasks, capsys):
    taskfile, cfgfile, directory = tmp_path / "tasks.json", tmp_path / "cfg.json", tmp_path / "experiment"
    taskfile.write_text(json.dumps(tasks[:1]))
    cfgfile.write_text(json.dumps(config(tmp_path, trials=1)))
    assert main(["experiment", "--tasks", str(taskfile), "--config", str(cfgfile), "--out", str(directory), "--plan-only"]) == 0
    assert not list(tmp_path.glob("*.log"))
    assert main(["resume", str(directory)]) == 0
    assert main(["compare", "--a", str(directory), "--b", str(directory), "--a-condition", "a", "--b-condition", "b", "--metric", "answer_correct"]) == 0
    review = tmp_path / "review.json"
    assert main(["review-template", str(directory), "--dimensions", "answer_correct", "--out", str(review)]) == 0
    assert json.loads(review.read_text())["items"][0]["human_reviewed"] is False
    rescored = tmp_path / "rescored.json"
    assert main(["rescore", str(directory), "--out", str(rescored)]) == 0
    assert len(json.loads(rescored.read_text())["records"]) == 2
    assert sum(len(p.read_text().splitlines()) for p in tmp_path.glob("*.log")) == 2



def test_cleanup_uncertain_preserves_claim(tmp_path, tasks, monkeypatch):
    from agent_eval.execution import ExecutionCleanupError

    directory = tmp_path / "experiment"
    e.create_experiment(directory, tasks[:1], config(tmp_path, trials=1))
    calls = []

    def uncertain_cleanup(request, *, on_start, **kwargs):
        calls.append(request["kind"])
        on_start(os.getpid())
        raise ExecutionCleanupError("controlled cleanup uncertainty")

    monkeypatch.setattr(e, "execute", uncertain_cleanup)
    with pytest.raises(ExecutionCleanupError, match="cleanup uncertainty"):
        e.run_experiment(directory)
    with e._connect(directory) as con:
        row = con.execute("SELECT state, worker_pid, record FROM trials WHERE position=0").fetchone()
    assert row["state"] == "running"
    assert row["worker_pid"] == os.getpid()
    assert json.loads(row["record"])["status"] == "running"
    with pytest.raises(RuntimeError, match="uncertain trial"):
        e.run_experiment(directory)
    with pytest.raises(RuntimeError, match="still alive"):
        e.run_experiment(directory, recover_interrupted=True)
    assert calls == ["agent"]
