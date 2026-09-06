"""Packaging checks run with the optional test build dependencies installed."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_package_metadata_declares_cli_and_offline_fixture_data():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["version"] == "0.2.0"
    assert metadata["project"]["scripts"]["agent-eval"] == "agent_eval.cli:main"
    assert metadata["project"]["dependencies"] == []
    assert "cases.json" in metadata["tool"]["setuptools"]["package-data"]["benchmarks.runtime_regression"]
    assert metadata["tool"]["setuptools"]["include-package-data"] is False


def test_wheel_contains_executable_modules_and_fixture_but_no_raw_caches(tmp_path):
    if importlib.util.find_spec("setuptools") is None:
        pytest.skip("Install .[test] to exercise the local wheel backend")
    stage = tmp_path / "source"
    stage.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(ROOT / name, stage / name)
    # Keep the test small and never copy downloaded public data or API caches.
    for package in ("agent_eval", "adapters", "benchmarks"):
        for source in (ROOT / package).rglob("*.py"):
            relative = source.relative_to(ROOT)
            if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
                continue
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for package, names in metadata["tool"]["setuptools"]["package-data"].items():
        directory = Path(*package.split("."))
        for name in names:
            destination = stage / directory / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / directory / name, destination)
    excluded = ["benchmarks/runtime_regression/results.json", "benchmarks/runtime_regression/.embedding_cache.json",
                "benchmarks/public_data/raw/download.json", "benchmarks/public_data/prepared/tasks.jsonl"]
    for relative in excluded:
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("private cached payload", encoding="utf-8")
    built = subprocess.run([sys.executable, "-B", "-c",
                            "from setuptools.build_meta import build_wheel; build_wheel('dist')"],
                           cwd=stage, capture_output=True, text=True, timeout=60)
    assert built.returncode == 0, built.stdout + built.stderr
    wheel = next((stage / "dist").glob("*.whl"))
    target = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert "agent_eval/cli.py" in names
        assert "adapters/react_agent_adapter.py" in names
        assert "benchmarks/runtime_regression/cases.py" in names
        assert "benchmarks/runtime_regression/cases.json" in names
        assert not names.intersection(excluded)
        entry = next(name for name in names if name.endswith("entry_points.txt"))
        assert "agent-eval = agent_eval.cli:main" in archive.read(entry).decode()
        archive.extractall(target)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(target)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    code = """
import json
from pathlib import Path
import agent_eval
from benchmarks.runtime_regression.cases import make_healthy_cases
assert str(Path(agent_eval.__file__).resolve()).startswith(str(Path.cwd().resolve()))
from agent_eval.concurrency_bench import benchmark
report = benchmark(case_factory=make_healthy_cases, repeats=2, max_workers=3)
assert report.speedup_ci is not None
print(json.dumps(report.quality_coverage))
"""
    checked = subprocess.run([sys.executable, "-B", "-c", code], cwd=target, env=env,
                             capture_output=True, text=True, timeout=15)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert json.loads(checked.stdout)["parallel"]["passed"] == 6
    cli = subprocess.run([sys.executable, "-B", "-m", "agent_eval", "--help"], cwd=target, env=env,
                         capture_output=True, text=True, timeout=10)
    assert cli.returncode == 0, cli.stderr
    assert "concurrency" in cli.stdout
