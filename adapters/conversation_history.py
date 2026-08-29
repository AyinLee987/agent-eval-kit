"""Threads conversation state through an agent whose ``run()`` is otherwise

stateless per call.

`ReActAgent.run(task)` in the sibling repo builds a brand-new execution
context on every call — it does not remember a prior call on the same
instance. Its ``context_providers`` extension point
(``ContextProvider.prepare(task) -> messages``, injected before the user's
message on every run) is the intended hook for exactly this. This class
implements that protocol structurally — it needs no import from the
sibling repo to do so — and accumulates turns so each new call sees the
full prior exchange.

Not agent-specific beyond assuming a ``context_providers``-shaped hook
exists somewhere to plug it into; ``adapters/react_agent_adapter.py``
wires it into ReActAgent specifically.
"""

from __future__ import annotations

from typing import Any, Dict, List


class ConversationHistoryProvider:
    """Accumulates turns and replays them as prior context on every call.

    ``ConversationHarness`` creates one fresh instance per conversation and
    calls :meth:`append_turn` after each turn completes, before the next
    turn's ``run()`` call.
    """

    def __init__(self) -> None:
        self._messages: List[Dict[str, str]] = []

    def prepare(self, task: str) -> List[Dict[str, str]]:
        """Return the accumulated history; ``task`` (the new turn) is ignored — the history doesn't depend on what's being asked now."""

        return list(self._messages)

    def append_turn(self, user_message: str, assistant_message: str) -> None:
        self._messages.append({"role": "user", "content": user_message})
        self._messages.append({"role": "assistant", "content": assistant_message})
