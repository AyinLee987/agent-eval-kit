"""LLM-as-judge over a full agent trajectory, not just the final answer.

Framework-agnostic: ``ChatFn`` is the smallest possible interface — a
callable taking OpenAI-style chat messages and returning the model's text
reply — so any provider works (OpenAI/DeepSeek/Bailian/a local model) as
long as the caller wraps its client into that shape. Nothing here imports a
specific LLM SDK.

Cross-model grading matters: a model judging its own agent's trajectories
tends to score them more favorably — a well-documented LLM-as-judge failure
mode ("self-preference bias"). This module has no way to enforce that the
judge differs from the agent being judged; it's a caller responsibility,
documented here and at every call site that wires one up.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Sequence

from .types import AgentOutcome

ChatFn = Callable[[List[Dict[str, str]]], str]

DEFAULT_DIMENSIONS: Sequence[str] = (
    "tool_selection",
    "tool_execution",
    "process_control",
    "output_quality",
)

_DIMENSION_GUIDANCE: Dict[str, str] = {
    "tool_selection": (
        "Did the agent choose tools that were actually necessary and "
        "appropriate for the task, without skipping a tool it clearly "
        "needed or calling one that wasn't relevant? If the task needed no "
        "tool at all and the agent correctly answered without calling one, "
        "that is the ideal outcome — score it 1.0, not 0.0 for 'no tool used'."
    ),
    "tool_execution": (
        "Were tool calls well-formed, with arguments correct given the "
        "task and prior observations? If no tool call was necessary for "
        "this task, there is nothing to execute incorrectly — score 1.0."
    ),
    "process_control": (
        "Did the agent's reasoning stay coherent step to step — no "
        "needless repetition, no contradicting its own prior observations, "
        "no runaway loop?"
    ),
    "output_quality": (
        "Is the final answer correct, complete, and grounded in what the "
        "trajectory actually observed, rather than invented?"
    ),
}


def render_trajectory(outcome: AgentOutcome) -> str:
    """Render a trajectory as readable text for a judge prompt."""

    lines = [f"Final answer: {outcome.answer}", f"Stop reason: {outcome.stop_reason}", ""]
    for index, step in enumerate(outcome.trajectory, start=1):
        lines.append(f"Step {index}:")
        if step.thought:
            lines.append(f"  thought: {step.thought}")
        if step.action:
            lines.append(f"  tool call: {step.action.name}({step.action.arguments})")
        if step.observation is not None:
            lines.append(f"  observation: {step.observation}")
    return "\n".join(lines)


def build_judge_prompt(
    task_prompt: str, outcome: AgentOutcome, dimensions: Sequence[str]
) -> List[Dict[str, str]]:
    guidance = "\n".join(f"- {dim}: {_DIMENSION_GUIDANCE.get(dim, '')}" for dim in dimensions)
    schema = ", ".join(f'"{dim}": <0.0-1.0>' for dim in dimensions)
    system = (
        "You are a strict, impartial evaluator of an AI agent's completed run. "
        "Judge only what the trajectory actually shows — do not reward a "
        "lucky right answer reached through a bad process, and do not "
        "penalize a correct process that hit a tool error outside the "
        "agent's control. Respond with a single JSON object and nothing else."
    )
    user = (
        f"Task given to the agent:\n{task_prompt}\n\n"
        f"Agent trajectory:\n{render_trajectory(outcome)}\n\n"
        f"Score each dimension from 0.0 (fails badly) to 1.0 (flawless):\n{guidance}\n\n"
        f'Respond with exactly this JSON shape: {{{schema}, "rationale": "<one or two sentences>"}}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _extract_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Judge response did not contain a JSON object: {text!r}")
    return json.loads(match.group(0))


def build_llm_judge_fn(
    chat_fn: ChatFn, dimensions: Sequence[str] = DEFAULT_DIMENSIONS
) -> Callable[[Dict[str, Any], AgentOutcome], Dict[str, Any]]:
    """Build a ``judge_fn`` for :class:`agent_eval.scoring.TrajectoryJudgeScorer`.

    ``chat_fn`` must call a model different from the one that produced the
    trajectories being judged — see the module docstring.
    """

    def judge_fn(task: Dict[str, Any], outcome: AgentOutcome) -> Dict[str, Any]:
        messages = build_judge_prompt(task.get("prompt", ""), outcome, dimensions)
        reply = chat_fn(messages)
        parsed = _extract_json(reply)
        result: Dict[str, Any] = {dim: float(parsed[dim]) for dim in dimensions if dim in parsed}
        result["rationale"] = parsed.get("rationale", "")
        return result

    return judge_fn
