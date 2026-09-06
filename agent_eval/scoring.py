"""Independent answer, execution and tool-contract checks over normalized outcomes.

Task declarations describe observable requirements, not a single mandatory
agent trace. Scorers distinguish an absent assertion from a failed assertion.
"""

from __future__ import annotations

import json
import math
import re
from decimal import Decimal, DecimalException
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from .types import AgentOutcome, ToolCall

Task = Dict[str, Any]
_NUMBER = re.compile(r"(?<![0-9A-Za-z_.,+-])[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?(?![0-9A-Za-z_]|[.,]\d)")
_MISSING = object()


class Scorer(Protocol):
    name: str
    metric_names: Sequence[str]

    def score(self, task: Task, outcome: AgentOutcome) -> Dict[str, Any]: ...


def _decimal(value: Any) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError("Expected a finite numeric value, not a boolean or string.")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("Expected a finite numeric value.")
    return result


def _numeric_text(text: str) -> str:
    return text.replace("−", "-").replace("－", "-").replace("＋", "+")


def _numbers(text: str) -> List[Decimal]:
    values = []
    for match in _NUMBER.finditer(_numeric_text(text)):
        try:
            values.append(Decimal(match.group().replace(",", "")))
        except DecimalException:
            values.append(Decimal("NaN"))
    return values


def _contains(answer: str, expected: str) -> bool:
    numeric = _numeric_text(expected.strip())
    if _NUMBER.fullmatch(numeric):
        try:
            return Decimal(numeric.replace(",", "")) in _numbers(answer)
        except DecimalException:
            return False
    return expected.casefold() in answer.casefold()


def _json_answer(answer: str) -> Any:
    text = answer.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1)
    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate answer field: {key}.")
            result[key] = value
        return result

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Nonfinite number in structured answer.")
        return number

    return json.loads(text, object_pairs_hook=unique_fields, parse_float=finite_float,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _pointer(value: Any, path: str) -> Any:
    if not isinstance(path, str) or (path and not path.startswith("/")):
        raise ValueError("A field path must be a JSON Pointer, for example /answer/value.")
    if not path:
        return value
    for component in path[1:].split("/"):
        component = component.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict):
            if component not in value:
                return _MISSING
            value = value[component]
        elif isinstance(value, list) and component.isdigit():
            index = int(component)
            if index >= len(value):
                return _MISSING
            value = value[index]
        else:
            return _MISSING
    return value


def _equal(actual: Any, expected: Any) -> bool:
    if actual is _MISSING:
        return False
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float, Decimal)) and isinstance(expected, (int, float, Decimal)):
        try:
            return _decimal(actual) == _decimal(expected)
        except ValueError:
            return False
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_equal(actual[key], value) for key, value in expected.items())
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(_equal(a, e) for a, e in zip(actual, expected))
    return type(actual) is type(expected) and actual == expected


def _subset(actual: Any, expected: Dict[str, Any]) -> bool:
    if not isinstance(actual, dict):
        return False
    return all(
        key in actual and (_subset(actual[key], value) if isinstance(value, dict) else _equal(actual[key], value))
        for key, value in expected.items()
    )


def _expect_number(answer: str, declaration: Any) -> bool:
    spec = declaration if isinstance(declaration, dict) else {"value": declaration}
    if not set(spec) <= {"value", "abs_tol", "rel_tol", "path"} or "value" not in spec:
        raise ValueError("expect_number requires value and optional abs_tol, rel_tol, path.")
    expected = _decimal(spec["value"])
    absolute = _decimal(spec.get("abs_tol", 0))
    relative = _decimal(spec.get("rel_tol", 0))
    if absolute < 0 or relative < 0:
        raise ValueError("Numeric tolerances cannot be negative.")
    if "path" in spec:
        # Validate paths even when the agent returned invalid JSON.
        _pointer({}, spec["path"])
        try:
            actual = _decimal(_pointer(_json_answer(answer), spec["path"]))
        except (ValueError, TypeError):
            return False
    else:
        values = _numbers(answer)
        if len(values) != 1:
            return False
        actual = values[0]
    try:
        return actual.is_finite() and abs(actual - expected) <= max(absolute, relative * abs(expected))
    except DecimalException:
        return False


class RuleScorer:
    """Separate execution completion from the task's answer assertions.

    ``expect_number`` accepts a scalar or value/abs_tol/rel_tol/path mapping.
    Without a JSON Pointer path the answer must contain exactly one number.
    ``expect_fields`` maps JSON Pointers to exact values; ``expect_set``
    supplies values and an optional path to a JSON array (order ignored).
    Legacy substrings remain useful for text; numeric ones match full values.
    """

    name = "rule"
    metric_names = ("run_completed", "answer_correct", "rule_pass")

    def __init__(self, finished_value: str = "finished") -> None:
        self.finished_value = finished_value

    def score(self, task: Task, outcome: AgentOutcome) -> Dict[str, Any]:
        completed = bool(outcome.success and outcome.stop_reason == self.finished_value)
        checks: List[bool] = []
        substrings = task.get("expect_substrings", [])
        if not isinstance(substrings, list) or any(not isinstance(item, str) or not item for item in substrings):
            raise ValueError("expect_substrings must be a list of nonempty strings.")
        if substrings:
            checks.append(all(_contains(outcome.answer, value) for value in substrings))
        if "expect_number" in task:
            checks.append(_expect_number(outcome.answer, task["expect_number"]))
        fields = task.get("expect_fields", {})
        if not isinstance(fields, dict):
            raise ValueError("expect_fields must map JSON Pointers to expected values.")
        for path, expected in fields.items():
            _pointer({}, path)
            try:
                actual = _pointer(_json_answer(outcome.answer), path)
            except (ValueError, TypeError):
                actual = _MISSING
            checks.append(_equal(actual, expected))
        if "expect_set" in task:
            spec = task["expect_set"]
            if not isinstance(spec, dict) or not set(spec) <= {"values", "path"} or not isinstance(spec.get("values"), list):
                raise ValueError("expect_set requires a values list and an optional JSON Pointer path.")
            _pointer({}, spec.get("path", ""))
            try:
                actual = _pointer(_json_answer(outcome.answer), spec.get("path", ""))
            except (ValueError, TypeError):
                actual = _MISSING
            expected = spec["values"]
            checks.append(isinstance(actual, list)
                          and all(any(_equal(value, item) for item in expected) for value in actual)
                          and all(any(_equal(value, item) for item in actual) for value in expected))
        correct = all(checks) if checks else None
        return {"run_completed": completed, "answer_correct": correct,
                "rule_pass": completed and correct if correct is not None else None}


def _contract(task: Task) -> Dict[str, Any]:
    contract = task.get("tool_contract", {})
    if not isinstance(contract, dict) or not set(contract) <= {"required", "forbidden", "order", "dependencies", "allow_errors"}:
        raise ValueError("Unknown or invalid tool_contract fields.")
    for key in ("required", "forbidden", "order", "dependencies", "allow_errors"):
        if not isinstance(contract.get(key, []), list):
            raise ValueError(f"tool_contract.{key} must be a list.")
    for key in ("forbidden", "allow_errors"):
        if any(not isinstance(name, str) or not name for name in contract.get(key, [])):
            raise ValueError(f"tool_contract.{key} must contain nonempty tool names.")
    return contract


def _tool_contract_pass(task: Task, outcome: AgentOutcome) -> Optional[bool]:
    contract = _contract(task)
    calls = outcome.tool_calls
    step_positions = [index for index, step in enumerate(outcome.trajectory) for _ in step.calls]
    checks: List[bool] = []
    for spec in contract.get("required", []):
        if not isinstance(spec, dict) or not set(spec) <= {"name", "arguments", "min_calls", "max_calls", "successful"}:
            raise ValueError("Invalid required tool declaration.")
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("A required tool needs a nonempty name.")
        arguments = spec.get("arguments", {})
        minimum, maximum = spec.get("min_calls", 1), spec.get("max_calls")
        if not isinstance(arguments, dict) or type(spec.get("successful", True)) is not bool:
            raise ValueError("Required arguments must be an object and successful must be boolean.")
        if type(minimum) is not int or minimum < 0 or (maximum is not None and (type(maximum) is not int or maximum < minimum)):
            raise ValueError("Required tool call counts must be nonnegative integers with max_calls >= min_calls.")
        named = [call for call in calls if call.name == name]
        matching = [call for call in named if _subset(call.arguments, arguments)]
        accepted = [call for call in matching if not spec.get("successful", True) or call.succeeded()]
        checks.append(len(accepted) >= minimum and (maximum is None or len(named) <= maximum))
    for name in contract.get("forbidden", []):
        checks.append(not outcome.used_tool(name))
    for pair in contract.get("order", []):
        if not isinstance(pair, list) or len(pair) != 2 or any(not isinstance(name, str) or not name for name in pair) or pair[0] == pair[1]:
            raise ValueError("Each order constraint needs two distinct tool names.")
        before, after = pair
        targets = [index for index, call in enumerate(calls) if call.name == after]
        checks.append(bool(targets) and all(any(call.name == before and call.succeeded() for call in calls[:index]) for index in targets))
    for spec in contract.get("dependencies", []):
        if not isinstance(spec, dict) or not set(spec) <= {"source", "target", "argument", "result_path", "match"}:
            raise ValueError("Invalid tool result dependency.")
        source, target = spec.get("source"), spec.get("target")
        if not isinstance(source, str) or not source or not isinstance(target, str) or not target or source == target:
            raise ValueError("A result dependency needs distinct source and target tool names.")
        _pointer({}, spec.get("argument"))
        _pointer({}, spec.get("result_path", ""))
        if spec.get("match", "equals") not in {"equals", "contains"}:
            raise ValueError("A result dependency match must be equals or contains.")
        targets = [index for index, call in enumerate(calls) if call.name == target]
        dependent = bool(targets)
        for index in targets:
            earlier = [(position, call) for position, call in enumerate(calls[:index]) if call.name == source and call.succeeded()]
            if not earlier or earlier[-1][1].observation is None or step_positions[earlier[-1][0]] >= step_positions[index]:
                # Arguments issued in the same model batch cannot depend on
                # observations that only arrive while that batch executes.
                dependent = False
                continue
            observation = earlier[-1][1].observation
            try:
                result = _json_answer(observation)
            except (ValueError, TypeError):
                result = observation
            result = _pointer(result, spec.get("result_path", ""))
            argument = _pointer(calls[index].arguments, spec["argument"])
            if spec.get("match", "equals") == "contains":
                dependent = dependent and isinstance(argument, str) and result is not _MISSING and isinstance(result, (str, int, float)) and not isinstance(result, bool) and _contains(argument, str(result))
            else:
                dependent = dependent and _equal(argument, result) and result is not _MISSING
        checks.append(dependent)
    return all(checks) if checks else None


class ToolUsageScorer:
    """Check legacy expected-tool presence and explicit observable contracts."""

    name = "tool_usage"
    metric_names = ("used_expected_tool", "tool_contract_pass")

    def score(self, task: Task, outcome: AgentOutcome) -> Dict[str, Any]:
        expected = task.get("expect_tool")
        return {"used_expected_tool": outcome.used_tool(expected) if expected else None,
                "tool_contract_pass": _tool_contract_pass(task, outcome)}


class TrajectoryScorer:
    """Report process health separately from answer correctness.

    A later successful call to the same tool can recover a failed attempt;
    explicitly allowed failures support tasks such as graceful division-by-
    zero handling. Error occurrence remains visible even after recovery.
    """

    name = "trajectory"
    metric_names = ("trajectory_score", "tool_error_free", "tool_recovery_pass")

    def __init__(self, finished_value: str = "finished", error_marker: str = "ERROR") -> None:
        self.finished_value = finished_value
        self.error_marker = error_marker

    def score(self, task: Task, outcome: AgentOutcome) -> Dict[str, Any]:
        completed = bool(outcome.success and outcome.stop_reason == self.finished_value)
        components = [float(completed)]
        if task.get("expect_tool"):
            components.append(float(outcome.used_tool(task["expect_tool"])))
        contract_result = _tool_contract_pass(task, outcome)
        if contract_result is not None:
            components.append(float(contract_result))
        error_free = not outcome.had_error(self.error_marker)
        recovery: Optional[bool] = None
        if not error_free:
            calls = outcome.tool_calls
            failures = [(index, call) for index, call in enumerate(calls) if call.failed(self.error_marker)]
            allowed = _contract(task).get("allow_errors", [])
            unattributed = any(step.error and not any(call.failed(self.error_marker) for call in step.calls) for step in outcome.trajectory)
            recovery = completed and bool(failures) and not unattributed and all(
                call.status not in {"fatal", "fatal_tool_error"} and (
                    call.name in allowed or any(later.name == call.name and later.succeeded(self.error_marker) for later in calls[index + 1:])
                ) for index, call in failures
            )
        components.append(float(error_free or recovery is True))
        return {"trajectory_score": sum(components) / len(components),
                "tool_error_free": error_free, "tool_recovery_pass": recovery}


class AnswerRelevancyScorer:
    name = "answer_relevancy"
    metric_names = ("answer_relevancy",)

    def __init__(self, judge_fn: Callable[[Task, AgentOutcome], Dict[str, Any]]) -> None:
        self.judge_fn = judge_fn

    def score(self, task: Task, outcome: AgentOutcome) -> Dict[str, Any]:
        from .judge import validate_judge_result

        result = validate_judge_result(self.judge_fn(task, outcome), self.metric_names)
        return {"answer_relevancy": result["answer_relevancy"],
                "answer_relevancy_rationale": result.get("rationale")}


class TrajectoryJudgeScorer:
    """Validate named dimensions, including those from an injected judge.

    Built judge functions declare their dimensions before execution. Legacy
    callables can declare them explicitly here, or infer them on their first
    valid response; later responses must keep the same dimensions.
    """

    name = "trajectory_judge"

    def __init__(
        self, judge_fn: Callable[[Task, AgentOutcome], Dict[str, Any]],
        dimensions: Optional[Sequence[str]] = None,
    ) -> None:
        self.judge_fn = judge_fn
        declared = dimensions if dimensions is not None else getattr(judge_fn, "dimensions", None)
        if declared is None:
            declared = getattr(judge_fn, "metric_names", None)
        self.dimensions = tuple(declared) if declared is not None else None

    @property
    def metric_names(self) -> Tuple[str, ...]:
        return tuple(f"judge_{name}" for name in (self.dimensions or ()))

    def score(self, task: Task, outcome: AgentOutcome) -> Dict[str, Any]:
        from .judge import validate_judge_result

        result = validate_judge_result(self.judge_fn(task, outcome), self.dimensions)
        if self.dimensions is None:
            self.dimensions = tuple(key for key in result if key != "rationale")
        scores = {f"judge_{key}": result[key] for key in self.dimensions}
        scores["judge_rationale"] = result.get("rationale")
        return scores


class LLMJudgeScorer:
    """A binary judge must return a real bool, not a truthy string or number."""

    name = "judge"
    metric_names = ("judge_pass",)

    def __init__(self, judge_fn: Callable[[Task, AgentOutcome], bool]) -> None:
        self.judge_fn = judge_fn

    def score(self, task: Task, outcome: AgentOutcome) -> Dict[str, Any]:
        result = self.judge_fn(task, outcome)
        if type(result) is not bool:
            raise ValueError("A binary judge must return True or False.")
        return {"judge_pass": result}
