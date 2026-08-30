"""Multi-agent latency + failure-isolation benchmark.

Measures the real wall-clock benefit of MultiAgentOrchestrator's concurrent
Worker dispatch against a plain sequential loop of ReActAgent.run() calls —
using real Bailian API calls, not synthetic time.sleep cases, since the
whole point is that parallelism benefit comes from overlapping network wait
time, which a synthetic delay can't demonstrate honestly.

MultiAgentOrchestrator owns its own ThreadPoolExecutor, sized by
RunBudget.max_parallel_tasks at construction time — so "how parallel" is
controlled by that budget, not by anything this script wraps around it.
spawn_subagent/wait_subagents are called directly (bypassing a real Leader
LLM's own delegate-or-not decision) so this measures pure dispatch/wait
infrastructure latency, not model decision variance.

Also verifies, with real data, a specific claim the sibling repo's README
makes: "a child fatal error terminates only that child." Two distinct
failure shapes, because they exercise different code paths:
  1. a pre-flight validation failure (an unknown Worker role) — rejected
     before a task is even queued;
  2. a mid-run fatal tool error inside an already-running Worker — the
     concurrent-execution case the README's claim is actually about.

Usage (from the evaluation/ repo root):

    python benchmarks/multi_agent_latency/run_benchmark.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

EVAL_ROOT = Path(__file__).resolve().parents[2]
HARNESS_REPO = Path(os.environ.get("AGENT_HARNESS_PATH", EVAL_ROOT.parent / "agent" / "agent-harness-from-scratch"))
BENCHMARK_DIR = Path(__file__).resolve().parent

for path in (str(EVAL_ROOT), str(HARNESS_REPO), str(BENCHMARK_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dotenv import load_dotenv  # noqa: E402

if not load_dotenv(dotenv_path=HARNESS_REPO / ".env"):
    raise SystemExit(f"Could not find a .env file at {HARNESS_REPO / '.env'}.")

from agent import (  # noqa: E402
    AgentRegistry,
    AgentSpec,
    BailianLLM,
    MultiAgentOrchestrator,
    ReActAgent,
    RecoverableToolError,
    RunBudget,
    ToolRegistry,
)

from tools import broken_tool, calculator, current_datetime, lookup_fact  # noqa: E402

TASKS_PATH = BENCHMARK_DIR / "tasks.json"
RESULTS_PATH = BENCHMARK_DIR / "results.json"
PARALLELISM_LEVELS = (1, 2, 3, 6)
WAIT_TIMEOUT_SECONDS = 120.0

WORKER_SYSTEM_PROMPT = (
    "You are a Worker. Solve only the delegated task, using a tool only "
    "when the task actually needs it, and return a concise answer."
)
BROKEN_WORKER_SYSTEM_PROMPT = (
    "You are a Worker whose only tool is broken_tool. Always call it with "
    "the task text as the query, then report whatever it returns."
)


def require_env(name: str) -> None:
    if not os.environ.get(name):
        raise SystemExit(
            f"Refusing to run: {name} is not set (checked {HARNESS_REPO / '.env'})."
        )


def load_tasks() -> List[Dict[str, Any]]:
    with open(TASKS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_worker_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(
        AgentSpec("worker", "General-purpose worker for latency benchmark tasks."),
        lambda: ReActAgent(
            llm=BailianLLM(),
            tools=ToolRegistry([calculator, lookup_fact, current_datetime]),
            system_prompt=WORKER_SYSTEM_PROMPT,
            max_steps=6,
            agent_name="latency-worker",
        ),
    )
    registry.register(
        AgentSpec("broken-worker", "Deliberately-broken worker; only for the isolation test."),
        lambda: ReActAgent(
            llm=BailianLLM(),
            tools=ToolRegistry([broken_tool]),
            system_prompt=BROKEN_WORKER_SYSTEM_PROMPT,
            max_steps=3,
            agent_name="broken-worker",
        ),
    )
    return registry


def run_sequential_baseline(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    started = time.perf_counter()
    outcomes = []
    for task in tasks:
        agent = ReActAgent(
            llm=BailianLLM(),
            tools=ToolRegistry([calculator, lookup_fact, current_datetime]),
            system_prompt=WORKER_SYSTEM_PROMPT,
            max_steps=6,
            agent_name="latency-worker",
        )
        result = agent.run(task["prompt"])
        outcomes.append(result.success)
    elapsed = time.perf_counter() - started
    return {
        "task_count": len(tasks),
        "serial_seconds": elapsed,
        "success_count": sum(outcomes),
    }


def run_parallel_level(tasks: List[Dict[str, Any]], max_parallel_tasks: int) -> Dict[str, Any]:
    registry = build_worker_registry()
    budget = RunBudget(max_parallel_tasks=max_parallel_tasks, max_subagents=len(tasks) + 2)
    with MultiAgentOrchestrator(registry, budget) as orchestrator:
        with orchestrator.leader_scope() as root_id:
            started = time.perf_counter()
            task_ids = [orchestrator.spawn_subagent("worker", t["prompt"]) for t in tasks]
            orchestrator.wait_subagents([r["task_id"] for r in task_ids], timeout_seconds=WAIT_TIMEOUT_SECONDS)
            elapsed = time.perf_counter() - started

            results = {r.task_id: r for r in orchestrator.results_for_run(root_id)}
            snapshots = {s.task_id: s for s in orchestrator.tasks_for_run(root_id)}

            per_task = []
            for r in task_ids:
                tid = r["task_id"]
                snap = snapshots[tid]
                result = results[tid]
                per_task.append({
                    "task_id": tid,
                    "success": result.success,
                    "queue_seconds": (snap.started_at - snap.created_at) if snap.started_at else None,
                    "exec_seconds": (snap.finished_at - snap.started_at) if snap.finished_at and snap.started_at else None,
                })

    return {
        "max_parallel_tasks": max_parallel_tasks,
        "task_count": len(tasks),
        "parallel_seconds": elapsed,
        "success_count": sum(1 for r in results.values() if r.success),
        "failure_count": sum(1 for r in results.values() if not r.success),
        "tasks": per_task,
    }


def run_preflight_failure_isolation(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    registry = build_worker_registry()
    budget = RunBudget(max_parallel_tasks=3, max_subagents=len(tasks) + 2)
    preflight_error = None
    with MultiAgentOrchestrator(registry, budget) as orchestrator:
        with orchestrator.leader_scope() as root_id:
            try:
                orchestrator.spawn_subagent("not-a-real-role", "irrelevant")
            except RecoverableToolError as exc:
                preflight_error = str(exc)

            task_ids = [orchestrator.spawn_subagent("worker", t["prompt"])["task_id"] for t in tasks]
            orchestrator.wait_subagents(task_ids, timeout_seconds=WAIT_TIMEOUT_SECONDS)
            results = orchestrator.results_for_run(root_id)

    return {
        "preflight_rejected": preflight_error is not None,
        "preflight_error": preflight_error,
        "other_tasks_total": len(tasks),
        "other_tasks_succeeded": sum(1 for r in results if r.success),
    }


def run_midrun_fatal_isolation(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    registry = build_worker_registry()
    normal_tasks = tasks[:-1]
    budget = RunBudget(max_parallel_tasks=3, max_subagents=len(tasks) + 2)
    with MultiAgentOrchestrator(registry, budget) as orchestrator:
        with orchestrator.leader_scope() as root_id:
            normal_ids = [orchestrator.spawn_subagent("worker", t["prompt"])["task_id"] for t in normal_tasks]
            broken_id = orchestrator.spawn_subagent(
                "broken-worker", "Call broken_tool to run a diagnostic check."
            )["task_id"]
            all_ids = normal_ids + [broken_id]
            orchestrator.wait_subagents(all_ids, timeout_seconds=WAIT_TIMEOUT_SECONDS)
            results = {r.task_id: r for r in orchestrator.results_for_run(root_id)}

    broken_result = results[broken_id]
    normal_results = [results[tid] for tid in normal_ids]
    return {
        "broken_task_success": broken_result.success,
        "broken_task_stop_reason": broken_result.stop_reason,
        "broken_task_error_type": broken_result.error_type,
        "normal_tasks_total": len(normal_tasks),
        "normal_tasks_succeeded": sum(1 for r in normal_results if r.success),
    }


def main() -> None:
    require_env("BAILIAN_API_KEY")
    tasks = load_tasks()

    print(f"Agent model: Bailian (qwen-plus). {len(tasks)} independent tasks.\n")

    print("Running sequential baseline (no orchestrator)...")
    baseline = run_sequential_baseline(tasks)
    print(f"  serial_seconds={baseline['serial_seconds']:.2f}  "
          f"success={baseline['success_count']}/{baseline['task_count']}\n")

    parallel_runs = []
    for level in PARALLELISM_LEVELS:
        print(f"Running orchestrated batch at max_parallel_tasks={level}...")
        report = run_parallel_level(tasks, level)
        speedup = baseline["serial_seconds"] / report["parallel_seconds"] if report["parallel_seconds"] else 0.0
        report["speedup_vs_serial"] = speedup
        parallel_runs.append(report)
        print(f"  parallel_seconds={report['parallel_seconds']:.2f}  speedup={speedup:.2f}x  "
              f"success={report['success_count']}/{report['task_count']}\n")

    print("Running pre-flight failure-isolation test (unknown role)...")
    preflight = run_preflight_failure_isolation(tasks)
    print(f"  preflight_rejected={preflight['preflight_rejected']}  "
          f"other_tasks_succeeded={preflight['other_tasks_succeeded']}/{preflight['other_tasks_total']}\n")

    print("Running mid-run failure-isolation test (fatal tool error)...")
    midrun = run_midrun_fatal_isolation(tasks)
    print(f"  broken_task_success={midrun['broken_task_success']} "
          f"error_type={midrun['broken_task_error_type']}  "
          f"normal_tasks_succeeded={midrun['normal_tasks_succeeded']}/{midrun['normal_tasks_total']}\n")

    print("=" * 58)
    print("MULTI-AGENT LATENCY: orchestrated vs. sequential")
    print("=" * 58)
    print(f"{'mode':<28}{'seconds':<12}{'speedup':<10}{'ok'}")
    print(f"{'sequential (baseline)':<28}{baseline['serial_seconds']:<12.2f}{'1.00x':<10}"
          f"{baseline['success_count']}/{baseline['task_count']}")
    for report in parallel_runs:
        label = f"orchestrated (k={report['max_parallel_tasks']})"
        print(f"{label:<28}{report['parallel_seconds']:<12.2f}"
              f"{report['speedup_vs_serial']:.2f}x{'':<4}{report['success_count']}/{report['task_count']}")
    print("=" * 58)

    payload = {
        "baseline": baseline,
        "parallel_runs": parallel_runs,
        "failure_isolation": {
            "preflight_invalid_role": preflight,
            "midrun_fatal_tool_error": midrun,
        },
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
