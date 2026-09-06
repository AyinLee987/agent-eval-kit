"""Paired serial/parallel batches with explicit quality and observed latency.

This thread runner has no hard execution deadline. ``timeout_seconds`` is an
observed end-to-end SLA (including queue time), evaluated after the actual task
has terminated. All workers are joined before returning, even on SLA misses.
Use an external process executor to stop uncooperative or hanging workloads.
"""

from __future__ import annotations

import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Sequence

from .stats import ConfidenceInterval, bootstrap_ci


@dataclass(frozen=True)
class ConcurrencyCase:
    """One isolated unit of work and an optional correctness oracle.

    Supply exactly one of task or task_factory. A factory is invoked inside
    each arm's timed execution and returns a fresh zero-argument task. Legacy
    task callables are reused and must therefore be safe to run repeatedly.
    quality_check runs outside the timed batch and must return a strict bool.
    No oracle means quality is unassessed, never an automatic pass.
    """

    name: str
    task: Callable[[], object] | None = None
    task_factory: Callable[[], Callable[[], object]] | None = field(default=None, kw_only=True)
    quality_check: Callable[[object], bool] | None = field(default=None, kw_only=True)
    timeout_seconds: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("case name must be a nonempty string")
        if (self.task is None) == (self.task_factory is None):
            raise ValueError("supply exactly one of task or task_factory")
        for name in ("task", "task_factory", "quality_check"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise ValueError(f"{name} must be callable")
        if self.timeout_seconds is not None and (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")


@dataclass(frozen=True)
class CaseMeasurement:
    name: str
    queue_seconds: float
    exec_seconds: float
    total_seconds: float
    success: bool
    quality_pass: bool | None = None
    error: str | None = None
    timed_out: bool = False
    quality_error: str | None = None
    quality_requested: bool = False

    @property
    def qualified(self) -> bool:
        return self.success and self.quality_pass is True and not self.timed_out


@dataclass(frozen=True)
class BatchMeasurement:
    mode: str
    seconds: float
    cases: tuple[CaseMeasurement, ...]

    @property
    def quality_passed_throughput(self) -> float:
        return sum(case.qualified for case in self.cases) / self.seconds if self.seconds else 0.0


@dataclass(frozen=True)
class PairedTrial:
    trial: int
    order: tuple[str, str]
    case_order: tuple[str, ...]
    serial: BatchMeasurement
    parallel: BatchMeasurement


@dataclass(frozen=True)
class ConcurrencyReport:
    """Times are sums over paired batches; counts include every repetition.

    speedup is a raw diagnostic ratio of total serial to parallel time.
    speedup_ci estimates the geometric mean of paired batch ratios, and is
    withheld unless EVERY case in EVERY arm passes execution, quality and SLA.
    raw_speedup_ci retains unqualified timing for debugging, never evidence of
    faster correct work. The sampling unit is a paired batch, not its tasks.
    Three to five repeats are a smoke measurement, not a precise performance
    claim; dependence and provider drift can require more experimental blocks.
    """

    case_count: int
    serial_seconds: float
    parallel_seconds: float
    success_count: int
    failure_count: int
    max_workers: int
    repeats: int = 1
    seed: int = 0
    trials: tuple[PairedTrial, ...] = ()
    raw_speedup_ci: ConfidenceInterval | None = None
    speedup_ci: ConfidenceInterval | None = None
    qualification_error: str | None = None

    @property
    def speedup(self) -> float:
        return self.serial_seconds / self.parallel_seconds if self.parallel_seconds else 0.0

    @property
    def quality_passed_throughput(self) -> dict[str, float]:
        return {
            mode: sum(case.qualified for trial in self.trials for case in getattr(trial, mode).cases)
            / seconds if seconds else 0.0
            for mode, seconds in (("serial", self.serial_seconds), ("parallel", self.parallel_seconds))
        }

    @property
    def quality_coverage(self) -> dict[str, dict[str, int]]:
        summary = {}
        for mode in ("serial", "parallel"):
            cases = [case for trial in self.trials for case in getattr(trial, mode).cases]
            summary[mode] = {
                "total": len(cases),
                "assessed": sum(case.quality_pass is not None for case in cases),
                "passed": sum(case.qualified for case in cases),
                "execution_failed": sum(not case.success for case in cases),
                "timed_out": sum(case.timed_out for case in cases),
                "quality_errors": sum(case.quality_error is not None for case in cases),
            }
        return summary

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "speedup": self.speedup,
            "quality_passed_throughput": self.quality_passed_throughput,
            "quality_coverage": self.quality_coverage,
            "timing_boundary": "batch dispatch through worker termination and executor shutdown; grading excluded",
            "timeout_mode": "observed_total_latency_sla_no_forced_termination",
            "ci_sampling_unit": "paired_batch",
        }

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def render(self) -> str:
        rates = self.quality_passed_throughput
        ci = self.speedup_ci.render() if self.speedup_ci else f"unavailable ({self.qualification_error})"
        coverage = self.quality_coverage
        return (
            f"{self.case_count} cases x {self.repeats} paired batches, max_workers={self.max_workers}: "
            f"serial={self.serial_seconds:.3f}s parallel={self.parallel_seconds:.3f}s "
            f"raw_speedup={self.speedup:.2f}x  parallel_ok={self.success_count} failed={self.failure_count}\n"
            f"quality/SLA qualified speedup CI: {ci}\n"
            f"quality-passed throughput: serial={rates['serial']:.3f}/s parallel={rates['parallel']:.3f}/s; "
            f"coverage={coverage}\n"
            "Timeouts are observed SLA misses; every worker was allowed to terminate and joined."
        )


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_cases(cases: Sequence[ConcurrencyCase]) -> list[ConcurrencyCase]:
    values = list(cases)
    if not all(isinstance(case, ConcurrencyCase) for case in values):
        raise ValueError("cases must contain ConcurrencyCase objects")
    if len({case.name for case in values}) != len(values):
        raise ValueError("case names must be unique within each batch")
    return values


def _execute(case: ConcurrencyCase, submitted: float) -> tuple[CaseMeasurement, object]:
    started = time.perf_counter()
    result = None
    error = None
    timeout_raised = False
    try:
        task = case.task_factory() if case.task_factory is not None else case.task
        if not callable(task):
            raise TypeError("task_factory must return a callable")
        result = task()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        timeout_raised = isinstance(exc, TimeoutError)
    finished = time.perf_counter()
    total = finished - submitted
    return CaseMeasurement(
        name=case.name,
        queue_seconds=started - submitted,
        exec_seconds=finished - started,
        total_seconds=total,
        success=error is None,
        error=error,
        timed_out=timeout_raised or (case.timeout_seconds is not None and total > case.timeout_seconds),
        quality_requested=case.quality_check is not None,
    ), result


def measure_batch(cases: Sequence[ConcurrencyCase], *, mode: str, max_workers: int = 1) -> BatchMeasurement:
    """Measure one batch; factory setup is timed, quality grading is excluded.

    Every case is considered queued at dispatch, including serial cases.
    ``mode`` is serial or parallel. This waits for actual thread completion;
    timeout_seconds never cancels a Future or releases an occupied worker.
    """
    cases = _validate_cases(cases)
    _positive_int(max_workers, "max_workers")
    if mode not in ("serial", "parallel"):
        raise ValueError("mode must be serial or parallel")
    started = time.perf_counter()
    if mode == "serial":
        raw = [_execute(case, started) for case in cases]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_execute, case, started) for case in cases]
            raw = [future.result() for future in futures]
    seconds = time.perf_counter() - started
    measured = []
    for case, (record, result) in zip(cases, raw):
        if case.quality_check is not None and record.success:
            try:
                passed = case.quality_check(result)
                if type(passed) is not bool:
                    raise TypeError("quality_check must return bool")
                record = replace(record, quality_pass=passed)
            except Exception as exc:
                record = replace(record, quality_error=f"{type(exc).__name__}: {exc}")
        measured.append(record)
    return BatchMeasurement(mode, seconds, tuple(measured))


def run_serial(cases: Sequence[ConcurrencyCase]) -> tuple[float, list[bool]]:
    """Compatibility wrapper; benchmark() retains full observations."""
    batch = measure_batch(cases, mode="serial")
    return batch.seconds, [case.success for case in batch.cases]


def run_parallel(cases: Sequence[ConcurrencyCase], max_workers: int) -> tuple[float, list[bool]]:
    """Compatibility wrapper; workers always terminate before return."""
    batch = measure_batch(cases, mode="parallel", max_workers=max_workers)
    return batch.seconds, [case.success for case in batch.cases]


def summarize_trials(trials: Sequence[PairedTrial], *, max_workers: int, seed: int = 0) -> ConcurrencyReport:
    """Summarize complete paired batches, also usable by custom dispatchers."""
    _positive_int(max_workers, "max_workers")
    if not trials:
        raise ValueError("at least one paired trial is required")
    names = {case.name for case in trials[0].serial.cases}
    for trial in trials:
        for mode in ("serial", "parallel"):
            batch = getattr(trial, mode)
            if batch.mode != mode or {case.name for case in batch.cases} != names or len(batch.cases) != len(names):
                raise ValueError("every paired arm must contain the same unique case names")
            if not math.isfinite(batch.seconds) or batch.seconds < 0:
                raise ValueError("batch seconds must be finite and nonnegative")
    serial = sum(trial.serial.seconds for trial in trials)
    parallel = sum(trial.parallel.seconds for trial in trials)
    parallel_cases = [case for trial in trials for case in trial.parallel.cases]
    all_cases = [case for trial in trials for batch in (trial.serial, trial.parallel) for case in batch.cases]
    raw_ci = None
    if len(trials) >= 2 and all(trial.serial.seconds > 0 and trial.parallel.seconds > 0 for trial in trials):
        logs = [math.log(trial.serial.seconds) - math.log(trial.parallel.seconds) for trial in trials]
        interval = bootstrap_ci(logs, seed=seed)
        raw_ci = replace(interval, mean=math.exp(interval.mean), low=math.exp(interval.low),
                         high=math.exp(interval.high), method="paired_batch_log_ratio_bootstrap")
    qualified = bool(all_cases) and all(case.qualified for case in all_cases)
    reason = None
    if not qualified:
        reason = "every case in both arms must complete, pass an explicit quality check, and meet its SLA"
    elif raw_ci is None:
        reason = "at least two paired batches with positive durations are required"
    return ConcurrencyReport(
        case_count=len(names), serial_seconds=serial, parallel_seconds=parallel,
        success_count=sum(case.success for case in parallel_cases),
        failure_count=sum(not case.success for case in parallel_cases), max_workers=max_workers,
        repeats=len(trials), seed=seed, trials=tuple(trials), raw_speedup_ci=raw_ci,
        speedup_ci=raw_ci if qualified else None, qualification_error=reason,
    )


def benchmark(
    cases: Sequence[ConcurrencyCase] = (), max_workers: int = 1, *,
    repeats: int = 1, seed: int = 0,
    case_factory: Callable[[], Sequence[ConcurrencyCase]] | None = None,
) -> ConcurrencyReport:
    """Repeat randomized paired blocks with fresh state from case_factory.

    Each block runs both arms in seeded random order, with the same shuffled
    case order. case_factory is called independently before EACH arm outside
    the timed region; task_factory runs inside it. Neither factory alone can
    isolate external/global state: callers must provide independent fixtures.
    Default repeats=1 preserves the old benchmark(cases, max_workers) contract.
    """
    _positive_int(max_workers, "max_workers")
    _positive_int(repeats, "repeats")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    cases = _validate_cases(cases)
    if case_factory is not None and (cases or not callable(case_factory)):
        raise ValueError("supply either cases or a callable case_factory")
    rng = random.Random(seed)
    expected_names = None
    trials = []
    for index in range(repeats):
        order = ["serial", "parallel"]
        rng.shuffle(order)
        case_order = None
        batches = {}
        for mode in order:
            fresh = _validate_cases(case_factory() if case_factory is not None else cases)
            by_name = {case.name: case for case in fresh}
            if expected_names is None:
                expected_names = set(by_name)
            elif set(by_name) != expected_names:
                raise ValueError("case_factory must return the same case names for every arm and trial")
            if case_order is None:
                case_order = sorted(by_name)
                rng.shuffle(case_order)
            batches[mode] = measure_batch([by_name[name] for name in case_order], mode=mode, max_workers=max_workers)
        trials.append(PairedTrial(index, tuple(order), tuple(case_order), batches["serial"], batches["parallel"]))
    return summarize_trials(trials, max_workers=max_workers, seed=seed)
