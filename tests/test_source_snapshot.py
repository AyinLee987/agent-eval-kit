"""Source fingerprints remain scoped after installing the distribution."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_eval import experiments as e


def _write(path, content="value = 1\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def installed(tmp_path, monkeypatch):
    base = tmp_path / "site-packages"
    module = _write(base / "agent_eval" / "experiments.py")
    monkeypatch.setattr(e, "__file__", str(module))
    return base


@pytest.fixture(autouse=True)
def git_metadata(monkeypatch):
    def fake_run(command, **kwargs):
        assert command[:2] == ["git", "-C"]
        output = "fixture-head\n" if command[3:] == ["rev-parse", "HEAD"] else ""
        return SimpleNamespace(returncode=0, stdout=output)

    monkeypatch.setattr(e.subprocess, "run", fake_run)


def test_installed_snapshot_ignores_unrelated_distribution_changes(installed):
    for name in ("adapters", "benchmarks"):
        _write(installed / name / "module.py")
    foreign = _write(installed / "unrelated_distribution" / "foreign.py")
    before = e.source_snapshot([])
    _write(foreign, "value = 999\n")
    _write(installed / "another_package" / "new.py")
    after = e.source_snapshot([])
    assert before == after
    assert {Path(snapshot["root"]) for snapshot in after} == {
        installed / name for name in ("agent_eval", "adapters", "benchmarks")
    }
    assert all(snapshot["git_head"] == "fixture-head" and snapshot["git_dirty"] is False for snapshot in after)
    assert all(Path(snapshot["root"]) != installed for snapshot in after)


def test_installed_snapshot_detects_own_module_changes(installed):
    owned = _write(installed / "adapters" / "own.py")
    before = {item["root"]: item["sha256"] for item in e.source_snapshot([])}
    _write(owned, "value = 2\n")
    after = {item["root"]: item["sha256"] for item in e.source_snapshot([])}
    assert before[str(installed / "adapters")] != after[str(installed / "adapters")]
    assert before[str(installed / "agent_eval")] == after[str(installed / "agent_eval")]


def test_installed_snapshot_includes_explicit_external_source_roots(installed, tmp_path):
    explicit = tmp_path / "custom-adapter"
    file = _write(explicit / "adapter.py")
    before = {item["root"]: item["sha256"] for item in e.source_snapshot([explicit])}
    _write(file, "value = 2\n")
    after = {item["root"]: item["sha256"] for item in e.source_snapshot([explicit])}
    assert before[str(explicit)] != after[str(explicit)]
    assert str(installed / "agent_eval") in after
    assert len(after) == 2


def test_checkout_snapshot_keeps_repository_scope_and_exclusions(installed):
    _write(installed / "pyproject.toml", "[project]\nname = 'fixture'\n")
    test = _write(installed / "tests" / "test_example.py")
    _write(installed / ".env", "sensitive fixture")
    _write(installed / ".venv" / "ignored.py")
    _write(installed / "results" / "ignored.py")
    before = e.source_snapshot([])
    assert len(before) == 1
    assert before[0]["root"] == str(installed)
    assert set(before[0]["files"]) == {"agent_eval/experiments.py", "pyproject.toml", "tests/test_example.py"}
    _write(test, "value = 2\n")
    assert e.source_snapshot([])[0]["sha256"] != before[0]["sha256"]


def test_partial_install_does_not_require_optional_package_directories(installed):
    # A pyproject marker alone is not enough to treat site-packages as a checkout.
    _write(installed / "pyproject.toml", "[project]\nname = 'unrelated'\n")
    snapshots = e.source_snapshot([installed / "agent_eval"])
    assert len(snapshots) == 1
    assert snapshots[0]["root"] == str(installed / "agent_eval")


def test_explicit_missing_source_root_is_still_rejected(installed, tmp_path):
    with pytest.raises(ValueError, match="Source root must be a directory"):
        e.source_snapshot([tmp_path / "missing-source"])
