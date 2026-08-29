"""A small, self-contained tool set for the trajectory-judge benchmark.

Imported at benchmark run time, after run_benchmark.py has put the sibling
agent-harness-from-scratch repo on sys.path — see that file for why these
imports work despite this module having no dependency of its own on it.
"""

from __future__ import annotations

from datetime import datetime, timezone

from agent import RecoverableToolError, tool

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
