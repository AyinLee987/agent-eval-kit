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

from .types import AgentOutcome, ConversationOutcome

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


# ---------------------------------------------------------------------------
# Answer relevancy (single-turn: does the answer address what was asked)
# ---------------------------------------------------------------------------


def build_answer_relevancy_prompt(task_prompt: str, outcome: AgentOutcome) -> List[Dict[str, str]]:
    system = (
        "You are a strict, impartial evaluator of whether an AI assistant's "
        "response actually engages with what the user said — not whether "
        "it is correct, just whether it is relevant. Respond with a single "
        "JSON object and nothing else."
    )
    user = (
        f"User's message: {task_prompt}\n\n"
        f"Assistant's response: {outcome.answer}\n\n"
        "Score how relevant the response is, from 0.0 (completely "
        "off-topic or non-responsive) to 1.0 (fully and appropriately "
        "engages with it). Not every user turn is a question — a "
        "statement that only shares information (e.g. introducing "
        "themselves, stating a fact, correcting something they said "
        "before) should be scored on whether the assistant appropriately "
        "acknowledged or used it, not on whether it answered a question "
        "that was never asked. Give partial credit for a response that "
        "addresses only part of a multi-part message.\n\n"
        'Respond with exactly this JSON shape: {"answer_relevancy": <0.0-1.0>, "rationale": "<one sentence>"}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_answer_relevancy_judge_fn(
    chat_fn: ChatFn,
) -> Callable[[Dict[str, Any], AgentOutcome], Dict[str, Any]]:
    """Build a ``judge_fn`` for :class:`agent_eval.scoring.AnswerRelevancyScorer`.

    ``chat_fn`` must call a model different from the one that produced the
    answer being judged — see the module docstring.
    """

    def judge_fn(task: Dict[str, Any], outcome: AgentOutcome) -> Dict[str, Any]:
        messages = build_answer_relevancy_prompt(task.get("prompt", ""), outcome)
        reply = chat_fn(messages)
        parsed = _extract_json(reply)
        return {
            "answer_relevancy": float(parsed["answer_relevancy"]),
            "rationale": parsed.get("rationale", ""),
        }

    return judge_fn


# ---------------------------------------------------------------------------
# Conversation-level (multi-turn) judging
# ---------------------------------------------------------------------------

CONVERSATION_DEFAULT_DIMENSIONS: Sequence[str] = (
    "knowledge_retention",
    "conversation_completeness",
)

_CONVERSATION_DIMENSION_GUIDANCE: Dict[str, str] = {
    "knowledge_retention": (
        "Across the whole conversation, did the assistant remember and "
        "correctly use information the user already provided, without "
        "contradicting it or asking for it again in a later turn? If the "
        "user updated a fact mid-conversation, the assistant should use "
        "the latest value, not an earlier one."
    ),
    "conversation_completeness": (
        "By the end of the conversation, was every one of the user's "
        "distinct requests addressed at some point — not necessarily in "
        "the same turn it was raised, but at least once before the "
        "conversation ends?"
    ),
}


def render_conversation(outcome: ConversationOutcome) -> str:
    """Render a full conversation as numbered turns for a judge prompt."""

    lines = []
    for index, turn in enumerate(outcome.turns, start=1):
        lines.append(f"Turn {index} — User: {turn.user_message}")
        lines.append(f"Turn {index} — Assistant: {turn.outcome.answer}")
    return "\n".join(lines)


def build_conversation_judge_prompt(
    task: Dict[str, Any],
    outcome: ConversationOutcome,
    dimensions: Sequence[str] = CONVERSATION_DEFAULT_DIMENSIONS,
) -> List[Dict[str, str]]:
    intentions = task.get("expected_intentions") or []
    intentions_block = (
        "\n".join(f"- {item}" for item in intentions)
        if intentions
        else "(none listed — judge based on what the user evidently asked for across the conversation)"
    )
    guidance = "\n".join(
        f"- {dim}: {_CONVERSATION_DIMENSION_GUIDANCE.get(dim, '')}" for dim in dimensions
    )
    schema = ", ".join(f'"{dim}": <0.0-1.0>' for dim in dimensions)
    system = (
        "You are a strict, impartial evaluator of a multi-turn conversation "
        "between a user and an AI assistant. Judge the conversation as a "
        "whole — a later turn can only be understood in light of earlier "
        "ones. Respond with a single JSON object and nothing else."
    )
    user = (
        f"Full conversation:\n{render_conversation(outcome)}\n\n"
        f"The user's expected intentions for this conversation:\n{intentions_block}\n\n"
        f"Score each dimension from 0.0 (fails badly) to 1.0 (flawless):\n{guidance}\n\n"
        f'Respond with exactly this JSON shape: {{{schema}, "rationale": "<one or two sentences>"}}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_conversation_judge_fn(
    chat_fn: ChatFn, dimensions: Sequence[str] = CONVERSATION_DEFAULT_DIMENSIONS
) -> Callable[[Dict[str, Any], ConversationOutcome], Dict[str, Any]]:
    """Build a ``judge_fn`` for a conversation-level judge scorer.

    ``chat_fn`` must call a model different from the one that produced the
    conversation being judged — see the module docstring.
    """

    def judge_fn(task: Dict[str, Any], outcome: ConversationOutcome) -> Dict[str, Any]:
        messages = build_conversation_judge_prompt(task, outcome, dimensions)
        reply = chat_fn(messages)
        parsed = _extract_json(reply)
        result: Dict[str, Any] = {dim: float(parsed[dim]) for dim in dimensions if dim in parsed}
        result["rationale"] = parsed.get("rationale", "")
        return result

    return judge_fn
