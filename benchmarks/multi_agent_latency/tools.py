"""Tool set for the multi-agent latency benchmark.

Same calculator/lookup_fact/current_datetime shape as the other benchmarks
(duplicated, not imported — see agent_trajectory/tools.py for the
convention), plus one tool that exists purely to test failure isolation:
``broken_tool`` always raises ``FatalToolError``. It is registered only for
the deliberately-broken "broken-worker" role — see run_benchmark.py's
mid-run fatal-error isolation test.
"""

from __future__ import annotations

from datetime import datetime, timezone

from agent import FatalToolError, RecoverableToolError, tool

from adapters.bare_baseline import safe_calculate

_FACTS = {
    "sev1_ack_minutes": "15",
    "oncall_stipend_usd": "200",
    "pto_days_per_year": "15",
    "standard_rate_limit_rpm": "60",
}


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the numeric result."""

    try:
        return safe_calculate(expression)
    except ZeroDivisionError:
        raise RecoverableToolError("Division by zero is undefined.")
    except Exception as exc:  # noqa: BLE001 - deliberately recoverable, not fatal
        raise RecoverableToolError(f"Could not evaluate expression {expression!r}: {exc}")


@tool
def lookup_fact(key: str) -> str:
    """Look up a known company fact by key.

    Available keys: sev1_ack_minutes, oncall_stipend_usd, pto_days_per_year,
    standard_rate_limit_rpm.
    """

    if key not in _FACTS:
        raise RecoverableToolError(f"Unknown fact key {key!r}. Available keys: {', '.join(_FACTS)}.")
    return _FACTS[key]


@tool
def current_datetime() -> str:
    """Return the current UTC date and time in ISO 8601 format."""

    return datetime.now(timezone.utc).isoformat()


@tool
def broken_tool(query: str) -> str:
    """Always fails. Exists only to test multi-agent failure isolation —
    never call this unless a benchmark task explicitly asks you to."""

    raise FatalToolError(
        f"Simulated fatal failure for the multi-agent latency benchmark's "
        f"failure-isolation test (query={query!r})."
    )
