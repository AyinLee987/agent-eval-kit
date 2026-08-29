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
