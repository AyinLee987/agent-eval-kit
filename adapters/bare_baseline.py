"""A deliberately weak, self-contained baseline agent.

No reasoning trace beyond one tool call, no retries, no error recovery —
this exists to (a) prove ``agent_eval`` works against a result shape other
than a full ReAct loop, and (b) give the harness's other adapters something
weaker to be compared against. Zero dependencies beyond the standard
library, so it runs with no setup.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from agent_eval.types import AgentOutcome, TrajectoryStep

_BIN_OPS: Dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS: Dict[type, Callable[[float], float]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression node: {ast.dump(node)}")


def safe_calculate(expression: str) -> str:
    """Evaluate a numeric expression without falling back to ``eval``."""

    tree = ast.parse(expression, mode="eval").body
    return str(_eval_node(tree))


def echo(text: str) -> str:
    return text


@dataclass
class BareAgentResult:
    answer: str
    success: bool
    stop_reason: str
    steps: int
    tokens: int
    trajectory: List[Dict[str, Any]] = field(default_factory=list)


class BareAgent:
    """Single-pass router: 'calculate: <expr>' goes to the calculator,

    everything else is echoed back unchanged. No planning, no iteration.
    """

    def run(self, prompt: str) -> BareAgentResult:
        if prompt.lower().startswith("calculate:"):
            expression = prompt.split(":", 1)[1].strip()
            try:
                observation = safe_calculate(expression)
            except Exception as exc:
                observation = f"ERROR: {exc}"
                return BareAgentResult(
                    answer=f"I couldn't evaluate that expression: {exc}",
                    success=False,
                    stop_reason="tool_error",
                    steps=1,
                    tokens=len(prompt.split()),
                    trajectory=[{
                        "thought": "Detected an arithmetic request.",
                        "action": {"name": "calculator", "arguments": {"expression": expression}},
                        "observation": observation,
                    }],
                )
            return BareAgentResult(
                answer=f"Based on the tool result, the answer is: {observation}",
                success=True,
                stop_reason="finished",
                steps=1,
                tokens=len(prompt.split()) + len(observation.split()),
                trajectory=[{
                    "thought": "Detected an arithmetic request.",
                    "action": {"name": "calculator", "arguments": {"expression": expression}},
                    "observation": observation,
                }],
            )

        observation = echo(prompt)
        return BareAgentResult(
            answer=observation,
            success=True,
            stop_reason="finished",
            steps=1,
            tokens=len(prompt.split()),
            trajectory=[{
                "thought": "No matching tool pattern; echoing the prompt back.",
                "action": {"name": "echo", "arguments": {"text": prompt}},
                "observation": observation,
            }],
        )


def build_agent() -> BareAgent:
    return BareAgent()


def adapt(result: BareAgentResult) -> AgentOutcome:
    return AgentOutcome(
        answer=result.answer,
        success=result.success,
        stop_reason=result.stop_reason,
        steps=result.steps,
        tokens=result.tokens,
        trajectory=[TrajectoryStep.from_dict(step) for step in result.trajectory],
        raw=result,
    )
