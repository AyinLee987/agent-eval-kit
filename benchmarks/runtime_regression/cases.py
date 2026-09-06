"""Controlled offline workloads for dispatch/measurement regression tests.

These are synthetic fixtures, not public datasets or real model capability
measurements. Recoverable faults are local task state, reset on every arm.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from agent_eval.concurrency_bench import ConcurrencyCase


def _case(spec):
    def setup():
        pending = list(spec["steps"])
        should_recover = spec.get("recover_once", False)

        def task():
            nonlocal should_recover
            time.sleep(spec["delay_seconds"])
            if spec.get("fatal"):
                raise RuntimeError("controlled terminal tool failure")
            retries = 0
            while should_recover:
                try:
                    should_recover = False
                    raise ValueError("controlled recoverable tool failure")
                except ValueError:
                    retries += 1
            calls = len(pending)
            total = sum(pending)
            pending.clear()
            return {"value": total, "calls": calls, "retries": retries}

        return task

    def quality(result):
        return (result["value"] == spec["expected"]
                and result["calls"] == len(spec["steps"])
                and result["retries"] == int(spec.get("recover_once", False)))

    return ConcurrencyCase(spec["name"], task_factory=setup, quality_check=quality,
                           timeout_seconds=spec.get("timeout_seconds"))


def make_cases():
    """Normal, slow, failed and wrong-answer cases; no API or environment access."""
    data = json.loads(Path(__file__).with_name("cases.json").read_text(encoding="utf-8"))
    return [_case(spec) for spec in data["cases"]]


def make_healthy_cases():
    """Only the normal and internally recovered cases, for quality-qualified timing."""
    return [case for case in make_cases() if case.name in {"normal", "multi-call", "recoverable-error"}]
