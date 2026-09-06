"""The live benchmark runner is exercised only with a controlled fake runtime."""
from __future__ import annotations

import importlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import pytest


@pytest.fixture
def live_runner(monkeypatch):
    module = importlib.import_module("benchmarks.multi_agent_latency.run_benchmark")

    class FakeRecoverableError(Exception):
        pass

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def run(self, instruction, **kwargs):
            if instruction == "slow":
                time.sleep(0.025)
            if instruction == "raises":
                raise RuntimeError("controlled worker error")
            return SimpleNamespace(success=True, stop_reason="finished", answer="5", steps=1, tokens=2,
                                   trajectory=[{"tool_calls": [{"name": "calculator", "observation": "5", "ok": True}]}])

    class FakeRegistry:
        def __init__(self):
            self.factories = {}

        def register(self, spec, factory):
            self.factories[spec.name] = factory

        def create(self, name):
            return self.factories[name]()

    class FakeOrchestrator:
        force_wait_timeout = False
        closed = False

        def __init__(self, registry, budget):
            self.registry = registry
            self.executor = ThreadPoolExecutor(max_workers=budget.max_parallel_tasks)
            self.futures = {}
            self.results = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.executor.shutdown(wait=True)
            type(self).closed = True

        @contextmanager
        def leader_scope(self):
            yield "root"

        def spawn_subagent(self, role, instruction):
            task_id = str(len(self.futures))

            def execute():
                try:
                    outcome = self.registry.create(role).run(instruction)
                    self.results[task_id] = SimpleNamespace(task_id=task_id, **vars(outcome))
                except Exception:
                    self.results[task_id] = SimpleNamespace(task_id=task_id, success=False,
                                                           stop_reason="fatal", error_type="RuntimeError")

            self.futures[task_id] = self.executor.submit(execute)
            return {"task_id": task_id}

        def wait_subagents(self, ids, timeout_seconds):
            if self.force_wait_timeout:
                raise FakeRecoverableError("Timed out waiting for subagents")
            for task_id in ids:
                self.futures[task_id].result()

        def results_for_run(self, root):
            return list(self.results.values())

    monkeypatch.setattr(module, "_load_runtime", lambda: None)
    replacements = {
        "ReActAgent": FakeAgent, "AgentRegistry": FakeRegistry,
        "AgentSpec": lambda name, description: SimpleNamespace(name=name),
        "BailianLLM": lambda: None, "ToolRegistry": lambda tools: tools,
        "MultiAgentOrchestrator": FakeOrchestrator,
        "RunBudget": lambda **kwargs: SimpleNamespace(**kwargs),
        "RecoverableToolError": FakeRecoverableError,
        "calculator": object(), "lookup_fact": object(), "current_datetime": object(), "broken_tool": object(),
    }
    for name, value in replacements.items():
        monkeypatch.setattr(module, name, value, raising=False)
    return module


def _task(name="case", prompt="normal", checked=True):
    return {"id": name, "prompt": prompt, "expect_substrings": ["5"] if checked else [], "expect_tool": "calculator"}


def test_live_runner_import_does_not_touch_dotenv_or_sibling_runtime():
    code = """
import sys
class DenyRuntime:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'dotenv' or fullname == 'agent' or fullname.startswith('agent.'):
            raise AssertionError('unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, DenyRuntime())
import benchmarks.multi_agent_latency.run_benchmark
"""
    completed = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr


def test_live_arms_both_record_answer_quality_and_task_errors(live_runner):
    tasks = [_task("ok"), _task("failed", "raises"), _task("unchecked", checked=False)]
    for payload in (live_runner.run_sequential_baseline(tasks), live_runner.run_parallel_level(tasks, 2)):
        records = {record["name"]: record for record in payload["tasks"]}
        assert records["ok"]["quality_pass"] is True
        assert records["ok"]["scores"]["answer_correct"] is True
        assert records["failed"]["success"] is False
        assert "controlled worker error" in records["failed"]["error"]
        assert records["unchecked"]["quality_pass"] is None
        assert records["unchecked"]["scores"]["answer_correct"] is None
        assert payload["success_count"] == 2
        assert payload["failure_count"] == 1
        for record in records.values():
            assert record["total_seconds"] == pytest.approx(record["queue_seconds"] + record["exec_seconds"])


def test_live_wait_timeout_includes_actual_worker_termination(live_runner):
    live_runner.MultiAgentOrchestrator.force_wait_timeout = True
    payload = live_runner.run_parallel_level([_task(prompt="slow")], 1)
    assert live_runner.MultiAgentOrchestrator.closed
    assert payload["parallel_seconds"] >= 0.025
    assert payload["tasks"][0]["exec_seconds"] >= 0.025
    assert payload["tasks"][0]["timed_out"] is True


def test_live_paired_report_repeats_every_arm_and_keeps_quality(live_runner):
    payload = live_runner.run_paired_benchmark([_task("a"), _task("b")], parallelism_levels=(1, 2), repeats=3, seed=4)
    assert len(payload["block_schedule"]) == 3
    for comparison in payload["comparisons"]:
        assert comparison["repeats"] == 3
        assert comparison["speedup_ci"]["n"] == 3
        assert comparison["quality_coverage"]["serial"]["passed"] == 6
        assert comparison["quality_coverage"]["parallel"]["passed"] == 6
        assert len(comparison["detailed_pairs"]) == 3
        for trial in comparison["trials"]:
            assert [record["name"] for record in trial["serial"]["cases"]] == list(trial["case_order"])
            assert [record["name"] for record in trial["parallel"]["cases"]] == list(trial["case_order"])


def test_live_missing_answer_contract_blocks_qualified_speedup(live_runner):
    payload = live_runner.run_paired_benchmark([_task(checked=False)], parallelism_levels=(1,), repeats=2)
    comparison = payload["comparisons"][0]
    assert comparison["speedup_ci"] is None
    assert comparison["raw_speedup_ci"] is not None
    assert comparison["quality_coverage"]["serial"]["assessed"] == 0
