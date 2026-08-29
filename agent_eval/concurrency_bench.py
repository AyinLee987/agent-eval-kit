"""Measures the speedup and failure isolation of parallel task dispatch.

Framework-agnostic: a "case" is any zero-arg callable. Point it at
independent LLM/tool calls, at ``MultiAgentOrchestrator.spawn_subagent`` +
``wait_subagents`` versus a sequential loop of ``agent.run()`` calls, or at
anything else shaped like "N independent units of work, one of them might
fail."
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Sequence


@dataclass(frozen=True)
class ConcurrencyCase:
    """One unit of work. ``task`` may raise — a raised exception counts as

    a failure for that case without stopping the others.
    """

    name: str
    task: Callable[[], object]


@dataclass(frozen=True)
class ConcurrencyReport:
    """Serial vs. parallel timing and failure-isolation result for one

    batch of cases.
    """

    case_count: int
    serial_seconds: float
    parallel_seconds: float
    success_count: int
    failure_count: int
    max_workers: int

    @property
    def speedup(self) -> float:
        return self.serial_seconds / self.parallel_seconds if self.parallel_seconds else 0.0

    def render(self) -> str:
        return (
            f"{self.case_count} cases, max_workers={self.max_workers}: "
            f"serial={self.serial_seconds:.3f}s parallel={self.parallel_seconds:.3f}s "
            f"speedup={self.speedup:.2f}x  ok={self.success_count} failed={self.failure_count}"
        )


def _run_one(task: Callable[[], object]) -> bool:
    """Run one case, swallowing its exception. Returns whether it succeeded."""

    try:
        task()
        return True
    except Exception:
        return False


def run_serial(cases: Sequence[ConcurrencyCase]) -> tuple[float, List[bool]]:
    start = time.perf_counter()
    outcomes = [_run_one(case.task) for case in cases]
    return time.perf_counter() - start, outcomes


def run_parallel(cases: Sequence[ConcurrencyCase], max_workers: int) -> tuple[float, List[bool]]:
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        outcomes = list(executor.map(_run_one, [case.task for case in cases]))
    return time.perf_counter() - start, outcomes


def benchmark(cases: Sequence[ConcurrencyCase], max_workers: int) -> ConcurrencyReport:
    """Run ``cases`` both serially and in parallel and report the speedup.

    Runs the same cases twice (once per mode), so ``task`` should be
    idempotent and side-effect-free beyond what it returns/raises — a
    one-shot mock LLM call or a pure function is fine; a stateful counter
    isn't.
    """

    serial_seconds, _ = run_serial(cases)
    parallel_seconds, outcomes = run_parallel(cases, max_workers)
    return ConcurrencyReport(
        case_count=len(cases),
        serial_seconds=serial_seconds,
        parallel_seconds=parallel_seconds,
        success_count=sum(outcomes),
        failure_count=len(outcomes) - sum(outcomes),
        max_workers=max_workers,
    )
