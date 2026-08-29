"""Adapter for the sibling ``agent-harness-from-scratch`` ReActAgent.

Deliberately not imported by anything in ``agent_eval`` or by this repo's
own test suite — that would create a hard dependency from a
"framework-agnostic" toolkit onto one specific framework. The import of
``agent`` (the sibling package) is delayed until a factory built here is
actually called, so the rest of this repo works with the sibling project
absent; only code paths that use this adapter need it installed, e.g.:

    pip install -e ../agent-harness-from-scratch
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

from agent_eval.types import AgentOutcome, TrajectoryStep

from .conversation_history import ConversationHistoryProvider


def _import_react_agent():
    try:
        from agent import ReActAgent  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "agent-harness-from-scratch is not importable. Install it "
            "alongside this project, e.g.:\n"
            "  pip install -e ../agent-harness-from-scratch\n"
            "or add its repo root to PYTHONPATH."
        ) from exc
    return ReActAgent


def build_agent_factory(
    llm_factory: Callable[[], Any],
    tools_factory: Callable[[], Any],
    **agent_kwargs: Any,
) -> Callable[[], Any]:
    """Return a zero-arg factory that builds a fresh ReActAgent per task.

    ``llm_factory``/``tools_factory`` are zero-arg callables too, so every
    task gets a fresh LLM/tool-registry instance — mirrors EvalHarness's
    "fresh agent per task" contract one level down, which matters for a
    stateful mock LLM (e.g. one used in the concurrency benchmark).
    """

    ReActAgent = _import_react_agent()

    def factory() -> Any:
        return ReActAgent(llm=llm_factory(), tools=tools_factory(), **agent_kwargs)

    return factory


def build_agent_and_history_factory(
    llm_factory: Callable[[], Any],
    tools_factory: Callable[[], Any],
    **agent_kwargs: Any,
) -> Callable[[], Tuple[Any, ConversationHistoryProvider]]:
    """Return a zero-arg factory building a fresh ``(ReActAgent, history)``

    pair per conversation, for
    :class:`agent_eval.conversation_harness.ConversationHarness`.

    A fresh :class:`ConversationHistoryProvider` is created and wired into
    the agent's ``context_providers`` on every call, so conversations never
    share history — mirrors :func:`build_agent_factory`'s per-task
    isolation, one level up (per-conversation instead of per-task). Any
    ``context_providers`` passed in ``agent_kwargs`` run before the history
    provider, in the order given.
    """

    ReActAgent = _import_react_agent()
    existing_providers = list(agent_kwargs.pop("context_providers", None) or [])

    def factory() -> Tuple[Any, ConversationHistoryProvider]:
        history = ConversationHistoryProvider()
        agent = ReActAgent(
            llm=llm_factory(),
            tools=tools_factory(),
            context_providers=[*existing_providers, history],
            **agent_kwargs,
        )
        return agent, history

    return factory


def adapt(result: Any) -> AgentOutcome:
    """Adapt a ReActAgent ``AgentResult`` into the generic AgentOutcome."""

    trajectory = [TrajectoryStep.from_dict(step) for step in result.trajectory]
    return AgentOutcome(
        answer=result.answer,
        success=result.success,
        stop_reason=result.stop_reason,
        steps=result.steps,
        tokens=result.tokens,
        trajectory=trajectory,
        raw=result,
    )
