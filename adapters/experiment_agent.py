"""JSON-configurable factory for the sibling agent; imports are execution-only.

Pass import references for llm_factory and tools_factory. Credentials belong
in the environment handled by those factories, never in experiment JSON.
"""
from agent_eval.execution import resolve
from .react_agent_adapter import _import_react_agent


def build_agent(*, llm_factory, tools_factory, llm_config=None, tools_config=None, **agent_kwargs):
    Agent = _import_react_agent()
    return Agent(llm=resolve(llm_factory)(**(llm_config or {})),
                 tools=resolve(tools_factory)(**(tools_config or {})), **agent_kwargs)
