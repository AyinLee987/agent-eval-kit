"""Optional live paired latency benchmark for MultiAgentOrchestrator.

This explicitly uses real Bailian calls when main() is invoked; importing the
module does not load .env, import the sibling runtime, or contact an API.
Independent paired blocks randomize both mode and case order. Both modes
include fresh worker construction and actual worker shutdown in batch time;
quality grading is outside the timing boundary. Per-case times use wrapper
observations, not the orchestrator's potentially earlier logical cancellation
stamps. A wait deadline can request cooperative cancellation, but close() is
still joined. This script cannot forcibly terminate an uncooperative thread.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, List

EVAL_ROOT = Path(__file__).resolve().parents[2]
HARNESS_REPO = Path(os.environ.get("AGENT_HARNESS_PATH", EVAL_ROOT.parent / "agent" / "agent-harness-from-scratch"))
BENCHMARK_DIR = Path(__file__).resolve().parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from adapters.react_agent_adapter import adapt
from agent_eval.concurrency_bench import BatchMeasurement, CaseMeasurement, PairedTrial, summarize_trials
from agent_eval.scoring import RuleScorer, ToolUsageScorer

TASKS_PATH = BENCHMARK_DIR / "tasks.json"
RESULTS_PATH = BENCHMARK_DIR / "results.json"
PARALLELISM_LEVELS = (1, 2, 3, 6)
WAIT_TIMEOUT_SECONDS = 120.0
_RUNTIME_LOADED = False

WORKER_SYSTEM_PROMPT = (
    "You are a Worker. Solve only the delegated task, using a tool only "
    "when the task actually needs it, and return a concise answer."
)
BROKEN_WORKER_SYSTEM_PROMPT = (
    "You are a Worker whose only tool is broken_tool. Always call it with "
    "the task text as the query, then report whatever it returns."
)


def _load_runtime() -> None:
    global _RUNTIME_LOADED, AgentRegistry, AgentSpec, BailianLLM, MultiAgentOrchestrator
    global ReActAgent, RecoverableToolError, RunBudget, ToolRegistry
    global broken_tool, calculator, current_datetime, lookup_fact
    if _RUNTIME_LOADED:
        return
    if str(HARNESS_REPO) not in sys.path:
        sys.path.insert(0, str(HARNESS_REPO))
    from agent import (AgentRegistry, AgentSpec, BailianLLM, MultiAgentOrchestrator,
                       ReActAgent, RecoverableToolError, RunBudget, ToolRegistry)
    from benchmarks.multi_agent_latency.tools import broken_tool, calculator, current_datetime, lookup_fact
    _RUNTIME_LOADED = True


def require_env(name: str) -> None:
    if not os.environ.get(name):
        raise SystemExit(f"Refusing to run: {name} is not set.")


def load_tasks() -> List[Dict[str, Any]]:
    return json.loads(TASKS_PATH.read_text(encoding="utf-8"))


def _new_worker(measurement=None, *, broken=False):
    if measurement is not None:
        measurement["started"] = time.perf_counter()
    try:
        worker = ReActAgent(
            llm=BailianLLM(),
            tools=ToolRegistry([broken_tool] if broken else [calculator, lookup_fact, current_datetime]),
            system_prompt=BROKEN_WORKER_SYSTEM_PROMPT if broken else WORKER_SYSTEM_PROMPT,
            max_steps=3 if broken else 6,
            agent_name="broken-worker" if broken else "latency-worker",
        )
    except Exception as exc:
        if measurement is not None:
            measurement.update(finished=time.perf_counter(), error=f"{type(exc).__name__}: {exc}")
        raise
    if measurement is not None:
        original_run = worker.run

        def tracked_run(*args, **kwargs):
            try:
                result = original_run(*args, **kwargs)
                measurement["result"] = result
                return result
            except Exception as exc:
                measurement["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                measurement["finished"] = time.perf_counter()

        worker.run = tracked_run
    return worker


def build_worker_registry(measurements=None, tasks=()):
    registry = AgentRegistry()
    registry.register(AgentSpec("worker", "General-purpose worker."), lambda: _new_worker())
    registry.register(AgentSpec("broken-worker", "Controlled fatal worker."), lambda: _new_worker(broken=True))
    for index, task in enumerate(tasks):
        record = measurements[task["id"]]
        registry.register(AgentSpec(f"case-{index}", "Independently measured worker."),
                          lambda record=record: _new_worker(record))
    return registry


def _quality(task, result):
    outcome = adapt(result)
    scores = {**RuleScorer().score(task, outcome), **ToolUsageScorer().score(task, outcome)}
    answer = scores["answer_correct"]
    if answer is None:
        return None, scores
    tools = [scores.get("used_expected_tool"), scores.get("tool_contract_pass")]
    return bool(scores["run_completed"] and answer and all(value is not False for value in tools)), scores


def _finish_records(tasks, observations, started, finished, logical_results=None):
    records = []
    for task in tasks:
        observed = observations[task["id"]]
        began = observed.get("started", finished)
        ended = observed.get("finished", finished)
        result = observed.get("result")
        logical = logical_results.get(task["id"]) if logical_results is not None else result
        success = bool(result is not None and logical is not None and result.success and logical.success
                       and result.stop_reason == "finished")
        error = observed.get("error")
        if not success and error is None:
            error = getattr(logical, "error_type", None) or getattr(logical, "stop_reason", None) or "no completed worker result"
        timed_out = bool(observed.get("wait_timed_out") or ended - started > WAIT_TIMEOUT_SECONDS
                         or getattr(logical, "stop_reason", None) == "timed_out")
        quality = None
        scores = {}
        quality_error = None
        if success:
            try:
                quality, scores = _quality(task, result)
            except Exception as exc:
                quality_error = f"{type(exc).__name__}: {exc}"
        record = CaseMeasurement(
            name=task["id"], queue_seconds=max(0.0, began - started),
            exec_seconds=max(0.0, ended - began), total_seconds=max(0.0, ended - started),
            success=success, quality_pass=quality, error=error, timed_out=timed_out,
            quality_error=quality_error, quality_requested=True,
        )
        records.append({**asdict(record), "scores": scores,
                        "worker_started": "started" in observed,
                        "task_id": observed.get("task_id")})
    return records


def _batch_payload(records, seconds, *, mode, max_workers=1):
    return {
        "task_count": len(records), f"{mode}_seconds": seconds,
        "success_count": sum(record["success"] for record in records),
        "failure_count": sum(not record["success"] for record in records),
        "max_parallel_tasks": max_workers,
        "tasks": records,
        "timing_boundary": "before registry/worker construction through actual worker and executor shutdown; quality excluded",
        "timeout_mode": "observed_sla_and_cooperative_orchestrator_wait; no forced thread termination",
    }


def run_sequential_baseline(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    _load_runtime()
    observations = {task["id"]: {} for task in tasks}
    started = time.perf_counter()
    # Both arms include registry construction in their timed region.
    registry = build_worker_registry(observations, tasks)
    for index, task in enumerate(tasks):
        try:
            registry.create(f"case-{index}").run(task["prompt"])
        except Exception as exc:
            observations[task["id"]].setdefault("error", f"{type(exc).__name__}: {exc}")
    finished = time.perf_counter()
    records = _finish_records(tasks, observations, started, finished)
    return _batch_payload(records, finished - started, mode="serial")


def run_parallel_level(tasks: List[Dict[str, Any]], max_parallel_tasks: int) -> Dict[str, Any]:
    _load_runtime()
    observations = {task["id"]: {} for task in tasks}
    logical_results = {}
    started = time.perf_counter()
    registry = build_worker_registry(observations, tasks)
    budget = RunBudget(max_parallel_tasks=max_parallel_tasks, max_subagents=len(tasks) + 2,
                       subagent_timeout_seconds=WAIT_TIMEOUT_SECONDS)
    with MultiAgentOrchestrator(registry, budget) as orchestrator:
        with orchestrator.leader_scope() as root_id:
            ids = {}
            for index, task in enumerate(tasks):
                try:
                    task_id = orchestrator.spawn_subagent(f"case-{index}", task["prompt"])["task_id"]
                    ids[task["id"]] = task_id
                    observations[task["id"]]["task_id"] = task_id
                except Exception as exc:
                    observations[task["id"]]["error"] = f"{type(exc).__name__}: {exc}"
            if ids:
                try:
                    orchestrator.wait_subagents(list(ids.values()), timeout_seconds=WAIT_TIMEOUT_SECONDS)
                except RecoverableToolError:
                    # Scope close requests cooperative cancellation; the outer
                    # context still joins every actual worker before timing ends.
                    for name in ids:
                        if "finished" not in observations[name]:
                            observations[name]["wait_timed_out"] = True
    finished = time.perf_counter()
    by_id = {result.task_id: result for result in orchestrator.results_for_run(root_id)}
    logical_results = {name: by_id.get(task_id) for name, task_id in ids.items()}
    records = _finish_records(tasks, observations, started, finished, logical_results)
    return _batch_payload(records, finished - started, mode="parallel", max_workers=max_parallel_tasks)


def _as_batch(payload, mode):
    keys = {item.name for item in fields(CaseMeasurement)}
    records = tuple(CaseMeasurement(**{key: value for key, value in record.items() if key in keys})
                    for record in payload["tasks"])
    return BatchMeasurement(mode, payload[f"{mode}_seconds"], records)


def run_paired_benchmark(tasks, *, parallelism_levels=PARALLELISM_LEVELS, repeats=3, seed=0):
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    levels = list(parallelism_levels)
    if not levels or len(set(levels)) != len(levels) or any(type(level) is not int or level < 1 for level in levels):
        raise ValueError("parallelism_levels must be unique positive integers")
    if not tasks or len({task["id"] for task in tasks}) != len(tasks):
        raise ValueError("tasks must have unique nonempty case identities")
    rng = random.Random(seed)
    pairs = {level: [] for level in levels}
    detailed_pairs = {level: [] for level in levels}
    schedule = []
    for trial in range(repeats):
        block_levels = levels[:]
        rng.shuffle(block_levels)
        schedule.append({"trial": trial, "levels": block_levels})
        for level in block_levels:
            ordered_tasks = list(tasks)
            rng.shuffle(ordered_tasks)
            order = ["serial", "parallel"]
            rng.shuffle(order)
            arms = {}
            for mode in order:
                arms[mode] = (run_sequential_baseline(ordered_tasks) if mode == "serial"
                              else run_parallel_level(ordered_tasks, level))
            pairs[level].append(PairedTrial(trial, tuple(order), tuple(task["id"] for task in ordered_tasks),
                                           _as_batch(arms["serial"], "serial"), _as_batch(arms["parallel"], "parallel")))
            detailed_pairs[level].append({"trial": trial, "order": order, **arms})
    return {
        "repeats": repeats, "seed": seed, "block_schedule": schedule,
        "comparisons": [{**summarize_trials(pairs[level], max_workers=level, seed=seed).to_dict(),
                         "detailed_pairs": detailed_pairs[level]} for level in levels],
        "tasks": tasks,
        "limitation": "Direct worker dispatch with live provider variance; no Leader decision or A2A/Skill evaluation. Missing quality contracts remain unassessed.",
    }


def run_preflight_failure_isolation(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    _load_runtime()
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
    _load_runtime()
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    parser.add_argument("--skip-isolation", action="store_true")
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=HARNESS_REPO / ".env")
    require_env("BAILIAN_API_KEY")
    _load_runtime()
    tasks = load_tasks()
    payload = run_paired_benchmark(tasks, repeats=args.repeats, seed=args.seed)
    if not args.skip_isolation:
        payload["failure_isolation"] = {
            "preflight_invalid_role": run_preflight_failure_isolation(tasks),
            "midrun_fatal_tool_error": run_midrun_fatal_isolation(tasks),
        }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for comparison in payload["comparisons"]:
        print(f"workers={comparison['max_workers']}: raw speedup={comparison['speedup']:.2f}x; "
              f"quality coverage={comparison['quality_coverage']}; "
              f"quality speedup CI={comparison['speedup_ci'] or comparison['qualification_error']}")
    print(f"Full paired results written to {args.out}")


if __name__ == "__main__":
    main()
