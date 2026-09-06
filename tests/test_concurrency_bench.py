import time

from agent_eval.concurrency_bench import ConcurrencyCase, benchmark


def _sleepy(seconds: float, should_fail: bool = False):
    def task():
        time.sleep(seconds)
        if should_fail:
            raise RuntimeError("boom")
        return "ok"

    return task


def test_parallel_dispatch_is_faster_than_serial_for_independent_tasks():
    cases = [ConcurrencyCase(name=f"t{i}", task=_sleepy(0.05)) for i in range(4)]

    report = benchmark(cases, max_workers=4)

    # Serial must take at least ~4x one task; parallel should be close to 1x.
    assert report.serial_seconds > report.parallel_seconds
    assert report.speedup > 1.5
    assert report.success_count == 4
    assert report.failure_count == 0


def test_one_failing_case_does_not_prevent_others_from_completing():
    cases = [
        ConcurrencyCase(name="ok-1", task=_sleepy(0.01)),
        ConcurrencyCase(name="fails", task=_sleepy(0.01, should_fail=True)),
        ConcurrencyCase(name="ok-2", task=_sleepy(0.01)),
    ]

    report = benchmark(cases, max_workers=3)

    assert report.success_count == 2
    assert report.failure_count == 1


import json
import threading

import pytest

from agent_eval.concurrency_bench import (
    BatchMeasurement, CaseMeasurement, PairedTrial, measure_batch, summarize_trials,
)


def test_factories_create_fresh_state_for_every_arm_and_trial():
    states = []
    case_batches = []

    def case_factory():
        case_batches.append(object())

        def task_factory():
            state = []
            states.append(state)

            def task():
                state.append("one use")
                return len(state)

            return task

        return [ConcurrencyCase("fresh", task_factory=task_factory, quality_check=lambda value: value == 1)]

    report = benchmark(max_workers=2, repeats=3, seed=11, case_factory=case_factory)
    assert len(case_batches) == len(states) == 6
    assert len({id(state) for state in states}) == 6
    assert report.success_count == 3
    assert report.speedup_ci.n == 3
    assert report.quality_coverage["serial"]["passed"] == 3
    assert report.quality_coverage["parallel"]["passed"] == 3


def test_seed_reproduces_arm_and_case_order():
    cases = [ConcurrencyCase(str(i), lambda: 1, quality_check=lambda value: value == 1) for i in range(5)]
    first = benchmark(cases, 2, repeats=5, seed=21)
    second = benchmark(cases, 2, repeats=5, seed=21)
    schedule = lambda report: [(trial.order, trial.case_order) for trial in report.trials]
    assert schedule(first) == schedule(second)
    assert {trial.order for trial in first.trials} == {("serial", "parallel"), ("parallel", "serial")}
    for trial in first.trials:
        assert tuple(case.name for case in trial.serial.cases) == trial.case_order
        assert tuple(case.name for case in trial.parallel.cases) == trial.case_order


def test_missing_quality_never_becomes_implicit_pass():
    report = benchmark([ConcurrencyCase("unchecked", lambda: "wrong")], 1, repeats=2)
    assert report.success_count == 2
    assert report.speedup_ci is None
    assert report.raw_speedup_ci is not None
    assert report.quality_coverage["parallel"]["assessed"] == 0
    assert report.quality_passed_throughput == {"serial": 0.0, "parallel": 0.0}


def test_quality_failures_and_grader_errors_stay_separate_from_execution():
    def broken_grader(value):
        raise ValueError("invalid oracle")

    cases = [
        ConcurrencyCase("wrong", lambda: 2, quality_check=lambda value: value == 1),
        ConcurrencyCase("invalid", lambda: 1, quality_check=lambda value: 1),
        ConcurrencyCase("grader-error", lambda: 1, quality_check=broken_grader),
    ]
    report = benchmark(cases, 3, repeats=2)
    for trial in report.trials:
        for batch in (trial.serial, trial.parallel):
            records = {case.name: case for case in batch.cases}
            assert records["wrong"].quality_pass is False
            assert records["invalid"].quality_pass is None
            assert "must return bool" in records["invalid"].quality_error
            assert "invalid oracle" in records["grader-error"].quality_error
    assert report.success_count == 6
    assert report.failure_count == 0
    assert report.speedup_ci is None
    assert report.quality_coverage["serial"]["quality_errors"] == 4


def test_timed_out_threads_really_finish_before_batch_returns():
    finished = threading.Event()

    def slow_task():
        time.sleep(0.03)
        finished.set()
        return "ok"

    batch = measure_batch([
        ConcurrencyCase("slow", slow_task, quality_check=lambda value: value == "ok", timeout_seconds=0.005),
        ConcurrencyCase("queued", lambda: "ok", quality_check=lambda value: value == "ok", timeout_seconds=0.005),
    ], mode="parallel", max_workers=1)
    assert finished.is_set()
    assert batch.seconds >= 0.03
    slow, queued = batch.cases
    assert slow.success and slow.quality_pass and slow.timed_out
    assert queued.queue_seconds >= 0.02
    assert queued.timed_out
    assert batch.quality_passed_throughput == 0.0
    for record in batch.cases:
        assert record.total_seconds == pytest.approx(record.queue_seconds + record.exec_seconds)


def test_task_timeout_error_is_recorded_and_other_tasks_run():
    def timeout():
        raise TimeoutError("cooperative cancellation")

    batch = measure_batch([ConcurrencyCase("timeout", timeout), ConcurrencyCase("ok", lambda: 1)], mode="parallel", max_workers=1)
    assert batch.cases[0].timed_out
    assert batch.cases[0].error == "TimeoutError: cooperative cancellation"
    assert batch.cases[1].success


def test_setup_is_inside_timing_and_grading_outside(monkeypatch):
    import agent_eval.concurrency_bench as module
    clock = [0.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])

    def setup():
        clock[0] += 2.0

        def task():
            clock[0] += 3.0
            return 7

        return task

    def grade(value):
        clock[0] += 100.0
        return value == 7

    batch = measure_batch([ConcurrencyCase("timing", task_factory=setup, quality_check=grade)], mode="serial")
    assert batch.seconds == batch.cases[0].exec_seconds == 5.0
    assert clock[0] == 105.0


def _paired_trial(index, serial_seconds, parallel_seconds, *, serial_pass=True, parallel_pass=True):
    serial_case = CaseMeasurement("case", 0, serial_seconds, serial_seconds, True, serial_pass)
    parallel_case = CaseMeasurement("case", 0, parallel_seconds, parallel_seconds, True, parallel_pass)
    return PairedTrial(index, ("serial", "parallel"), ("case",),
                       BatchMeasurement("serial", serial_seconds, (serial_case,)),
                       BatchMeasurement("parallel", parallel_seconds, (parallel_case,)))


def test_speedup_interval_resamples_paired_batches_not_individual_cases():
    report = summarize_trials([_paired_trial(0, 8, 4), _paired_trial(1, 6, 3), _paired_trial(2, 4, 2)], max_workers=2)
    assert report.speedup == pytest.approx(2.0)
    assert report.speedup_ci.mean == pytest.approx(2.0)
    assert report.speedup_ci.n == 3
    assert report.speedup_ci.method == "paired_batch_log_ratio_bootstrap"
    assert report.quality_passed_throughput["serial"] == pytest.approx(3 / 18)
    assert report.quality_passed_throughput["parallel"] == pytest.approx(3 / 9)


@pytest.mark.parametrize("bad_arm", ["serial", "parallel"])
def test_one_bad_batch_prevents_quality_speedup_claim_without_filtering(bad_arm):
    trials = [_paired_trial(0, 10, 1), _paired_trial(1, 10, 1, **{bad_arm + "_pass": False})]
    report = summarize_trials(trials, max_workers=2)
    assert report.speedup == 10
    assert report.raw_speedup_ci.n == 2
    assert report.speedup_ci is None
    assert report.quality_coverage[bad_arm]["passed"] == 1


def test_empty_and_one_repeat_do_not_publish_interval():
    empty = benchmark([], 2, repeats=2)
    assert empty.speedup_ci is None
    single = benchmark([ConcurrencyCase("ok", lambda: 1, quality_check=lambda value: True)], 1)
    assert single.speedup_ci is None
    assert "at least two" in single.qualification_error


def test_json_report_preserves_every_arm_and_case(tmp_path):
    report = benchmark([ConcurrencyCase("safe", lambda: object(), quality_check=lambda value: True)], 1, repeats=2)
    path = tmp_path / "concurrency.json"
    report.dump(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["trials"]) == 2
    assert data["trials"][0]["serial"]["cases"][0]["name"] == "safe"
    assert data["ci_sampling_unit"] == "paired_batch"
    assert data["timeout_mode"] == "observed_total_latency_sla_no_forced_termination"
    assert "raw_speedup=" in report.render()
    assert "quality-passed throughput" in report.render()


@pytest.mark.parametrize("kwargs", [{"max_workers": 0}, {"repeats": 0}, {"repeats": True}, {"seed": True}])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        benchmark([], **kwargs)


def test_case_factory_identity_drift_rejected():
    calls = [0]

    def factory():
        calls[0] += 1
        return [ConcurrencyCase(str(calls[0]), lambda: 1)]

    with pytest.raises(ValueError, match="same case names"):
        benchmark(case_factory=factory)


def test_case_validation_and_failed_setup():
    with pytest.raises(ValueError, match="exactly one"):
        ConcurrencyCase("missing")
    with pytest.raises(ValueError, match="exactly one"):
        ConcurrencyCase("both", lambda: 1, task_factory=lambda: lambda: 1)
    with pytest.raises(ValueError, match="finite and positive"):
        ConcurrencyCase("deadline", lambda: 1, timeout_seconds=float("nan"))
    with pytest.raises(ValueError, match="unique"):
        benchmark([ConcurrencyCase("dup", lambda: 1), ConcurrencyCase("dup", lambda: 2)], 1)
    batch = measure_batch([ConcurrencyCase("bad-factory", task_factory=lambda: None)], mode="serial")
    assert batch.cases[0].success is False
    assert "must return a callable" in batch.cases[0].error



def test_offline_fixture_exposes_faults_without_masking_fast_failures():
    from benchmarks.runtime_regression.cases import make_cases, make_healthy_cases
    report = benchmark(max_workers=3, repeats=2, case_factory=make_cases)
    assert report.case_count == 6
    assert report.quality_coverage["parallel"]["execution_failed"] == 2
    assert report.quality_coverage["serial"]["passed"] < 12
    assert report.speedup_ci is None
    for trial in report.trials:
        records = {case.name: case for case in trial.parallel.cases}
        assert records["recoverable-error"].quality_pass is True
        assert records["incorrect-result"].quality_pass is False
        assert records["slow-sla"].timed_out
    healthy = benchmark(max_workers=3, repeats=2, case_factory=make_healthy_cases)
    assert healthy.speedup_ci is not None
    assert healthy.quality_coverage["serial"]["passed"] == 6
