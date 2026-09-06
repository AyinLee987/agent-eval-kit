"""Observe actual synchronous ReAct calls without changing their return values.

Instrumentation is scoped to one exclusively owned agent during ``run`` and
restored even after failure. There are no global patches or framework imports
in the generic evaluation package. Provider-internal HTTP retries and children
that have not themselves been instrumented remain explicitly outside coverage.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import replace
import threading
from typing import Any, Dict, Iterator, Optional

from agent_eval.types import AgentOutcome, TrajectoryStep


_LIMITATIONS = [
    "Provider-internal HTTP retry attempts are not individual spans.",
    "Child agents need their own instrumentation; a delegation tool span alone does not expose child internals.",
    "Only synchronous run is instrumented; async and streaming entry points are unsupported.",
    "Custom context providers, durable-memory services and tool-internal I/O need additional instrumentation.",
    "Recorded model output is provider-exposed content, not hidden chain of thought.",
]


def _usage(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    fields = ("prompt_tokens", "completion_tokens", "total_tokens", "estimated")
    if isinstance(value, dict):
        return {key: value[key] for key in fields if key in value}
    return {key: getattr(value, key) for key in fields if hasattr(value, key)}


def _response(value: Any) -> Dict[str, Any]:
    return {
        "content": getattr(value, "content", None),
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in getattr(value, "tool_calls", [])
        ],
    }


def _budget(ctx: Any) -> Dict[str, Any]:
    return {
        "run_id": ctx.run_id,
        "step": len(ctx.steps) - 1,
        "tokens_used": ctx.tokens_used,
        "max_steps": ctx.max_steps,
        "max_tokens": ctx.max_tokens,
    }


@contextmanager
def _temporary_attr(owner: Any, name: str, value: Any) -> Iterator[None]:
    previous = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, previous)


@contextmanager
def _span(recorder: Any, name: str, kind: str, **kwargs: Any) -> Iterator[Any]:
    with recorder.span(name, kind, **kwargs) as span:
        try:
            yield span
        except BaseException as error:
            usage = _usage(getattr(error, "usage", None))
            if usage is not None:
                span.finish(status="error", usage=usage, metadata={
                    "exception": {"type": type(error).__name__, "message": str(error)},
                })
            if any(base.__name__ == "ControlSignal" and base.__module__.startswith("agent.")
                   for base in type(error).__mro__):
                status = "suspended" if type(error).__name__ == "SuspendRun" else "cancelled"
                span.finish(status=status, metadata={"control_signal": type(error).__name__})
            raise


class _Proxy:
    def __init__(self, target: Any, recorder: Any) -> None:
        self._target = target
        self._recorder = recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class _LLMProxy(_Proxy):
    @property
    def supports_output_shrink(self) -> bool:
        # The tracing facade exposes a bounded method even for legacy
        # targets. That fallback observes calls; it cannot enforce a cap.
        return callable(getattr(self._target, "chat_with_budget", None)) and (
            getattr(self._target, "supports_output_shrink", True) is not False
        )

    def chat(self, messages: Any, tools: Any = None) -> Any:
        with _span(self._recorder, "llm.chat", "llm", input={"messages": messages, "tools": tools}) as span:
            result = self._target.chat(messages, tools=tools)
            span.finish(output=_response(result), usage=_usage(getattr(result, "usage", None)))
            return result

    def chat_with_budget(self, messages: Any, tools: Any = None, *, max_output_tokens: int) -> Any:
        # Forward the bounded interface through the same trace boundary; a
        # plain __getattr__ would silently skip the LLM span after hardening.
        with _span(self._recorder, "llm.chat", "llm", input={"messages": messages, "tools": tools},
                   metadata={"max_output_tokens": max_output_tokens}) as span:
            bounded = getattr(self._target, "chat_with_budget", None)
            result = (bounded(messages, tools=tools, max_output_tokens=max_output_tokens)
                      if callable(bounded) else self._target.chat(messages, tools=tools))
            span.finish(output=_response(result), usage=_usage(getattr(result, "usage", None)))
            return result

    def embed(self, text: str) -> Any:
        with _span(self._recorder, "llm.embed", "embedding", input={"text": text}) as span:
            result = self._target.embed(text)
            span.finish(output={"dimensions": len(result)}, metadata={"usage_available": False})
            return result


class _RegistryProxy(_Proxy):
    def __contains__(self, name: str) -> bool:
        return name in self._target

    def __len__(self) -> int:
        return len(self._target)

    def dispatch(self, name: str, arguments: Any) -> Any:
        with _span(self._recorder, name, "tool_attempt", input=arguments) as span:
            result = self._target.dispatch(name, arguments)
            span.finish(output=result)
            return result


class _DispatcherProxy(_Proxy):
    def dispatch(self, ctx: Any, name: str, arguments: Any) -> Any:
        details = _budget(ctx)
        pending = ctx.state.get("__pending_tools__", {})
        calls = pending.get("tool_calls", [])
        cursor = pending.get("next_call", 0)
        if isinstance(cursor, int) and 0 <= cursor < len(calls):
            details["tool_call_id"] = calls[cursor].get("id")
        details["previous_failures"] = ctx.state.get(f"retries::{name}", 0)
        with _span(self._recorder, name, "tool", input=arguments, metadata=details) as span:
            result = self._target.dispatch(ctx, name, arguments)
            failed = isinstance(result, str) and result.startswith("ERROR")
            span.finish(output=result, status="error" if failed else "ok",
                        metadata={"budget_after": _budget(ctx), "recoverable_observation": failed})
            return result


class _MemoryProxy(_Proxy):
    def manage(self, messages: Any) -> Any:
        before_count = len(messages)
        with _span(self._recorder, "context.manage", "context", input={"messages": messages}) as span:
            result = self._target.manage(messages)
            span.finish(output={"messages": result},
                        metadata={"messages_before": before_count, "messages_after": len(result)})
            return result


class _CompressorProxy(_Proxy):
    def compress_messages(self, messages: Any, query: str) -> Any:
        with _span(self._recorder, "context.compress", "context",
                   input={"messages": messages, "query": query}) as span:
            result = self._target.compress_messages(messages, query)
            span.finish(output={"messages": result[0]}, metadata={"estimated_tokens_saved": result[1]})
            return result


def _find_native(agent: Any) -> Optional[Any]:
    candidate = agent
    # Support the existing DeepSeek evaluation wrapper without changing its
    # provider accounting or exception handling.
    if not hasattr(candidate, "_loop") and hasattr(candidate, "agent"):
        candidate = candidate.agent
    loop = getattr(candidate, "_loop", None)
    return candidate if loop is not None and hasattr(loop, "_dispatcher") else None


@contextmanager
def _instrument(native: Any, recorder: Any) -> Iterator[None]:
    loop = native._loop
    with ExitStack() as stack:
        llm = loop.llm
        stack.enter_context(_temporary_attr(loop, "llm", _LLMProxy(llm, recorder)))
        memory = loop.short_term
        if hasattr(memory, "_llm") and callable(getattr(memory._llm, "chat", None)):
            stack.enter_context(_temporary_attr(memory, "_llm", _LLMProxy(memory._llm, recorder)))
        stack.enter_context(_temporary_attr(loop, "short_term", _MemoryProxy(memory, recorder)))
        compressor = loop.compressor
        if compressor is not None:
            if getattr(compressor, "llm", None) is not None:
                stack.enter_context(_temporary_attr(compressor, "llm", _LLMProxy(compressor.llm, recorder)))
            stack.enter_context(_temporary_attr(loop, "compressor", _CompressorProxy(compressor, recorder)))
        dispatcher = loop._dispatcher
        stack.enter_context(_temporary_attr(dispatcher, "registry", _RegistryProxy(dispatcher.registry, recorder)))
        stack.enter_context(_temporary_attr(loop, "_dispatcher", _DispatcherProxy(dispatcher, recorder)))
        yield


def _attach(result: Any, metadata: Dict[str, Any]) -> Any:
    if isinstance(result, AgentOutcome):
        return replace(result, metadata={**result.metadata, **metadata})
    existing = getattr(result, "metadata", None)
    if isinstance(existing, dict):
        # Result metadata belongs to this run. No provider accounting is lost.
        try:
            result.metadata = {**existing, **metadata}
        except (AttributeError, TypeError):
            pass
    else:
        try:
            result._eval_trace_metadata = dict(metadata)
        except (AttributeError, TypeError):
            # A slots/frozen third-party result stays usable with its original
            # adapter; the durable worker also attaches these links to outcome.
            pass
    return result


class _TracedAgent:
    def __init__(self, agent: Any, recorder: Any, name: str) -> None:
        self._agent = agent
        self._recorder = recorder
        self._name = name
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        if name in {"iter_run", "aiter_run"}:
            raise NotImplementedError("trace_agent currently instruments synchronous run only")
        return getattr(self._agent, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._agent, name, value)

    def run(self, prompt: str, *args: Any, **kwargs: Any) -> Any:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("A traced agent instance cannot run concurrently; use a fresh agent per task")
        try:
            native = _find_native(self._agent)
            runtime_context = {}
            if native is not None:
                from agent.observability import current_log_context
                runtime_context = current_log_context()
            metadata = {"trace_id": self._recorder.trace_id, "trace_path": str(self._recorder.path),
                        "trace_coverage": "react_sync_calls" if native is not None else "run_only",
                        "trace_limitations": list(_LIMITATIONS)}
            limits = {key: getattr(native, key, None) for key in ("max_steps", "max_tokens")}
            with _span(self._recorder, self._name + ".run", "agent", input={"prompt": prompt},
                       metadata={**metadata, "budget": limits, "runtime_context": runtime_context,
                                 "resumed": kwargs.get("resume_from") is not None}) as span:
                if native is None:
                    span.event("trace.coverage.limited", {"reason": "No supported ReActAgent interface; only run is timed"})
                    result = self._agent.run(prompt, *args, **kwargs)
                else:
                    with _instrument(native, self._recorder):
                        result = self._agent.run(prompt, *args, **kwargs)
                unwrapped = result.result if not hasattr(result, "answer") and hasattr(result, "result") else result
                output = {key: getattr(unwrapped, key, None)
                          for key in ("answer", "success", "stop_reason", "steps", "tokens")}
                span.event("agent.stop", output)
                span.finish(output=output, status="ok" if output["success"] else "error")
                return _attach(result, metadata)
        finally:
            self._lock.release()


def trace_agent(agent: Any, recorder: Any, *, name: str = "agent") -> Any:
    """Return a synchronous wrapper retaining ``run`` result and ``close``.

    The caller owns recorder lifetime and must close it. Do not simultaneously
    use the raw agent, its shared memory/dispatcher, or a second wrapper while
    this wrapper runs. Unknown agent interfaces get explicitly limited run-only
    coverage. Original exceptions propagate and dependencies are restored.

    The native AgentRegistry requires concrete ReActAgent instances. Register
    an explicit ReActAgent subclass that instruments its original run entry
    (see trace_smoke.TracedWorker), rather than registering this wrapper.
    """
    if isinstance(agent, _TracedAgent):
        raise ValueError("Agent is already instrumented")
    return _TracedAgent(agent, recorder, name)


def adapt_traced(result: Any) -> AgentOutcome:
    """Preserve the existing trajectory while linking the recorded trace.

    Provider-specific wrappers should keep their own outcome adapter (for
    example ``deepseek_public_eval:adapt``), which already preserves metadata.
    """
    if isinstance(result, AgentOutcome):
        return result
    metadata = {**getattr(result, "metadata", {}), **getattr(result, "_eval_trace_metadata", {})}
    return AgentOutcome(answer=result.answer, success=result.success, stop_reason=result.stop_reason,
                        steps=result.steps, tokens=result.tokens,
                        trajectory=[step if isinstance(step, TrajectoryStep) else TrajectoryStep.from_dict(step)
                                    for step in result.trajectory], raw=result, metadata=metadata)
