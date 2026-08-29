"""The framework-agnostic contract every adapter targets.

Any agent's run result can be scored by this toolkit once it is translated
into an :class:`AgentOutcome`. The toolkit never imports a specific agent
framework — translation is the adapter's job (see ``adapters/``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation recorded in a trajectory step."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryStep:
    """One think/act/observe step.

    All fields are optional so partial or differently-shaped agent logs can
    still be represented — a step with only a final ``thought`` and no
    ``action`` is valid (e.g. the last step before a plain-text answer).
    """

    thought: Optional[str] = None
    action: Optional[ToolCall] = None
    observation: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryStep":
        """Build a step from the common ``{thought, action, observation}``

        dict shape most agent loggers already produce, so most adapters can
        do ``TrajectoryStep.from_dict(raw_step)`` and stop there.
        """

        action = data.get("action")
        tool_call = None
        if action:
            tool_call = ToolCall(
                name=action.get("name", ""),
                arguments=action.get("arguments") or action.get("args") or {},
            )
        return cls(
            thought=data.get("thought"),
            action=tool_call,
            observation=data.get("observation"),
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

    def used_tool(self, name: str) -> bool:
        """Whether any trajectory step invoked the named tool."""

        return any(step.action is not None and step.action.name == name for step in self.trajectory)

    def had_error(self, marker: str = "ERROR") -> bool:
        """Whether any observation looks like a tool-level error."""

        return any(
            (step.observation or "").startswith(marker) for step in self.trajectory
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
