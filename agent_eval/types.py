"""The framework-agnostic contract every adapter targets.

Any agent's run result can be scored by this toolkit once it is translated
into an :class:`AgentOutcome`. The toolkit never imports a specific agent
framework — translation is the adapter's job (see ``adapters/``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ToolCall:
    """One invocation and its own result, without flattening a tool batch."""

    name: str
    arguments: Any = field(default_factory=dict)
    id: Optional[str] = None
    observation: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None
    ok: Optional[bool] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        return cls(
            name=data.get("name", ""),
            arguments=data["arguments"] if "arguments" in data else data.get("args", {}),
            id=data.get("id", data.get("tool_call_id")),
            observation=data.get("observation"),
            status=data.get("status"),
            error=data.get("error"),
            ok=data.get("ok"),
        )

    def failed(self, marker: str = "ERROR") -> bool:
        if self.error or self.status in {"failed", "error", "fatal", "fatal_tool_error", "cancelled", "timed_out"}:
            return True
        if self.ok is not None:
            return not self.ok
        return _error_observation(self.observation, marker)

    def succeeded(self, marker: str = "ERROR") -> bool:
        if self.failed(marker) or self.status in {"pending", "running", "submitted", "requested", "suspended"}:
            return False
        return self.ok is True or self.status in {"succeeded", "success", "completed", "finished"} or self.observation is not None


def _error_observation(observation: Optional[str], marker: str) -> bool:
    # Legacy flattened observations may put a later tool's error after a
    # successful result. Structured per-call status takes precedence above.
    return any(line.lstrip().startswith(marker) for line in (observation or "").splitlines())


@dataclass(frozen=True)
class TrajectoryStep:
    """A model step may contain several calls, or no call at all.

    ``tool_calls=None`` means an older producer only supplied ``action``;
    an explicit empty list means no invocation was recorded. In particular,
    a suspended step's action placeholder must not become a completed call.
    """

    thought: Optional[str] = None
    action: Optional[ToolCall] = None
    observation: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    error: Optional[str] = None

    @property
    def calls(self) -> List[ToolCall]:
        if self.tool_calls is not None:
            return list(self.tool_calls)
        if self.action is None:
            return []
        return [replace(
            self.action,
            observation=self.action.observation if self.action.observation is not None else self.observation,
            error=self.action.error if self.action.error is not None else self.error,
        )]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryStep":
        action = data.get("action")
        return cls(
            thought=data.get("thought"),
            action=ToolCall.from_dict(action) if action else None,
            observation=data.get("observation"),
            tool_calls=[ToolCall.from_dict(call) for call in (data.get("tool_calls") or [])]
            if data.get("tool_calls") is not None else None,
            error=data.get("error"),
        )


@dataclass(frozen=True)
class AgentOutcome:
    """Normalized result of one agent run — the toolkit's only input type.

    ``raw`` is an escape hatch: adapters may stash the original,
    framework-specific result object there for scorers that need more than
    the normalized fields (most scorers don't).
    """

    answer: str
    success: bool
    stop_reason: str
    steps: int
    tokens: int
    trajectory: List[TrajectoryStep] = field(default_factory=list)
    raw: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def tool_calls(self) -> List[ToolCall]:
        """All invocations in trajectory order, including later calls in a step."""
        return [call for step in self.trajectory for call in step.calls]

    def used_tool(self, name: str) -> bool:
        return any(call.name == name for call in self.tool_calls)

    def had_error(self, marker: str = "ERROR") -> bool:
        return any(call.failed(marker) for call in self.tool_calls) or any(
            bool(step.error) or (step.tool_calls is None and _error_observation(step.observation, marker))
            for step in self.trajectory
        )


@dataclass(frozen=True)
class Turn:
    """One exchange in a multi-turn conversation: what the user said, and

    the agent's normalized outcome for that turn.
    """

    user_message: str
    outcome: AgentOutcome


@dataclass(frozen=True)
class ConversationOutcome:
    """A full multi-turn conversation: ordered turns from one continuous

    session with a single agent instance. Unlike a single-turn
    :class:`AgentOutcome`, later turns may depend on earlier ones — that
    dependency is the entire point of evaluating a conversation rather than
    scoring each turn in isolation (see :mod:`agent_eval.conversation_scoring`).
    """

    turns: List[Turn]

    @property
    def total_tokens(self) -> int:
        return sum(turn.outcome.tokens for turn in self.turns)

    @property
    def total_steps(self) -> int:
        return sum(turn.outcome.steps for turn in self.turns)

    def transcript(self) -> str:
        """Render the conversation as alternating "User:"/"Assistant:" lines."""

        lines = []
        for turn in self.turns:
            lines.append(f"User: {turn.user_message}")
            lines.append(f"Assistant: {turn.outcome.answer}")
        return "\n".join(lines)
