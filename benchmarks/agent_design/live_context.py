"""Exercise existing context and memory paths against isolated stateful fixtures.

The fixtures supply history, documents and faults, never repaired summaries or
answer oracles. Native SessionContextProvider performs and caches real model
summaries. This is a single current-runtime condition, not an ablation study.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from types import SimpleNamespace
from typing import Any

from adapters.trace_agent import adapt_traced, trace_agent
from agent_eval.types import AgentOutcome
from benchmarks.agent_design.answer_json import extract_object


SUPPORTED_IDS = {f"CONTEXT-{index:02d}" for index in range(1, 11)}
UNSUPPORTED: dict[str, str] = {}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


class _ObservedLLM:
    def __init__(self, llm: Any, recorder: Any, state: dict[str, Any], purpose: str) -> None:
        self.llm, self.recorder, self.state, self.purpose = llm, recorder, state, purpose

    def __getattr__(self, name: str) -> Any:
        return getattr(self.llm, name)

    @property
    def supports_output_shrink(self) -> bool:
        return callable(getattr(self.llm, "chat_with_budget", None)) and getattr(self.llm, "supports_output_shrink", True) is not False

    def chat(self, messages: Any, tools: Any = None) -> Any:
        return self._chat(messages, tools=tools)

    def chat_with_budget(self, messages: Any, tools: Any = None, *, max_output_tokens: int) -> Any:
        return self._chat(messages, tools=tools, max_output_tokens=max_output_tokens)

    def _chat(self, messages: Any, tools: Any = None, *, max_output_tokens: int | None = None) -> Any:
        def invoke() -> Any:
            limited = getattr(self.llm, "chat_with_budget", None)
            if max_output_tokens is not None and callable(limited):
                return limited(messages, tools=tools, max_output_tokens=max_output_tokens)
            return self.llm.chat(messages, tools=tools)

        if self.purpose == "main":
            from agent.trigger.react_loop import _prompt_tokens
            self.state["model_inputs"].append(copy.deepcopy(messages))
            self.state["main_input_estimated_tokens"].append(_prompt_tokens(messages, tools))
            # trace_agent already owns the real main-call span and usage.
            return invoke()
        with self.recorder.span("context." + self.purpose + ".model", "llm",
                                input={"messages": messages, "tools": tools},
                                metadata={"purpose": self.purpose, "max_output_tokens": max_output_tokens}) as span:
            response = invoke()
            usage = getattr(response, "usage", None)
            span.finish(output={"content": response.content,
                                "tool_calls": [vars(call) for call in response.tool_calls]},
                        usage={key: getattr(usage, key) for key in ("prompt_tokens", "completion_tokens", "total_tokens", "estimated") if hasattr(usage, key)})
        if self.purpose == "summary":
            self.state["summaries"].append(response.content or "")
        return response


class _FixtureEmbeddings:
    """Deterministic vectors isolate native authorization from embedding quality."""

    model_id = "agent-design-fixture-vectors-v1"
    dimension = 2

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: Any) -> list[list[float]]:
        return [[1.0, 0.0] if "XLSX" in text else [0.8, 0.6] for text in texts]


def build_case(case: dict[str, Any], native: Any, llm: Any, recorder: Any) -> Any:
    """Build one case without making model calls or touching external state."""
    from agent.errors import RecoverableToolError
    from agent.llm import estimate_tokens
    from agent.memory.context import SessionContextProvider
    from agent.memory.session import InMemorySessionStore
    from agent.state.memory import ShortTermMemory
    from agent.token_budget import TokenBudget

    case_id = case["id"]
    if case_id not in SUPPORTED_IDS:
        raise ValueError(f"Unsupported context case: {case_id}")
    initial = copy.deepcopy(case["setup"]["initial_state"])
    limits = case["setup"]["limits"]
    state: dict[str, Any] = {
        "calls": [], "model_inputs": [], "main_input_estimated_tokens": [], "summaries": [], "prepared_contexts": [],
        "fault_triggered": False, "files": copy.deepcopy(initial.get("files", {})),
        "published": [], "memory_write_attempts": 0, "memory_write_seconds": 0.0,
        "integrity_checks": [],
    }
    limitations = [
        "Single current-runtime condition; no paired architecture intervention is executed.",
        "All business resources are per-case in-memory fixtures; no external file or publish effects.",
        "One native TokenBudget covers all main and summary calls, including both turns when applicable; the catalog allowance is unchanged.",
        "Noise sizes and context sizes use the companion runtime estimate_tokens heuristic; provider usage is reported separately.",
    ]
    tools = []
    history = InMemorySessionStore()
    token_budget = TokenBudget(limits["max_total_tokens"])
    main_llm = _ObservedLLM(llm, recorder, state, "main")
    summary_llm = _ObservedLLM(llm, recorder, state, "summary")

    def note(name: str, args: dict[str, Any], result: Any) -> Any:
        state["calls"].append({"name": name, "arguments": copy.deepcopy(args), "result": copy.deepcopy(result)})
        recorder.event("fixture." + name, state["calls"][-1])
        return result

    def register(function: Any, name: str) -> None:
        tools.append(native.tool(name, error_policy="recoverable")(function))

    def add_history(messages: list[dict[str, Any]]) -> None:
        for message in messages:
            history.append_message(case_id, message)

    def add_noise() -> None:
        noise = initial.get("noise", {"count": 16, "tokens_per_chunk": 80})
        for index in range(noise["count"]):
            prefix = f"Unrelated archive fragment {index:03d}. "
            content = prefix
            while estimate_tokens(content) < noise["tokens_per_chunk"]:
                content += "data "
            add_history([{"role": "assistant", "content": content}])
        state["noise"] = {"chunks": noise["count"], "tokens_per_chunk_estimate": noise["tokens_per_chunk"]}

    class Provider(SessionContextProvider):
        def prepare(self, task: str) -> Any:
            with recorder.span("session.prepare", "context", input={"conversation_id": case_id}) as span:
                messages = list(super().prepare(task))
                if case_id == "CONTEXT-07" and messages and state["summaries"]:
                    messages[0] = {"role": "system", "content": "Summary of earlier conversation:\n" + case["fault"]["injected_summary"]}
                    state["fault_triggered"] = True
                    recorder.event("fault.summary.replaced", {
                        "source": "controlled_environment_intervention_after_real_summary",
                        "original_constraint_sha256": hashlib.sha256(initial["constraint"].encode()).hexdigest(),
                        "actual_model_summary": state["summaries"][-1],
                        "injected_summary": case["fault"]["injected_summary"],
                    })
                if case_id in {"CONTEXT-02", "CONTEXT-03", "CONTEXT-04"} and state["summaries"]:
                    state["fault_triggered"] = True
                state["prepared_contexts"].append(copy.deepcopy(messages))
                span.finish(output={"messages": messages})
                return messages

        def validate_prepared(self, task: str, messages: Any) -> None:
            entry = {"validator": "native.SessionContextProvider.validate_prepared", "accepted": False}
            try:
                super().validate_prepared(task, messages)
            except Exception as exc:
                entry.update(error_type=type(exc).__name__, message=str(exc))
                raise
            else:
                entry["accepted"] = True
            finally:
                state["integrity_checks"].append(entry)
                recorder.event("session.context.integrity_checked", entry)

    provider = Provider(history, case_id, llm=summary_llm, recent_window=2, summarize_beyond=6)
    memory = ShortTermMemory(summary_llm, window=24, max_tokens=initial.get("context_budget_tokens", 2000))
    prompt = case["prompt"]
    manager = None
    multi_turn = False

    if case_id == "CONTEXT-01":
        add_history(initial["messages"])

        def docs_read(doc_id: str) -> str:
            """Read sandbox release document D1."""
            if doc_id != "D1":
                raise RecoverableToolError("Unknown document")
            return _json(note("docs_read", {"doc_id": doc_id}, initial["document"]))

        register(docs_read, "docs_read")
        prompt += ' 文档为 D1。最终返回 JSON：{"version":版本,"changes":[中文变更条目]}。'

    elif case_id in {"CONTEXT-02", "CONTEXT-03"}:
        if case_id == "CONTEXT-02":
            add_history(initial["messages"])
            prompt = "继续整理 D1 资料，按之前约定的输出位置完成报告。"
        else:
            add_history([{"role": "user", "content": case["prompt"]}])
            prompt = "继续整理 D1 资料并交付报告，遵守之前约定的逐字引用要求。"
        add_noise()
        document = {"doc_id": "D1", "facts": ["修复导出", "增加筛选"]}
        limitations.append("Catalog omitted D1 contents for this case; this run pins a supplemental two-fact release document in fixture evidence.")
        state["document"] = document

        def docs_read(doc_id: str) -> str:
            """Read sandbox source document D1."""
            if doc_id != "D1":
                raise RecoverableToolError("Unknown document")
            return _json(note("docs_read", {"doc_id": doc_id}, document))

        def sandbox_write(path: str, content: str) -> str:
            """Write report text to the named in-memory sandbox path."""
            state["files"][path] = content
            return _json(note("sandbox_write", {"path": path, "content": content}, {"written": path}))

        def memory_get_ref(key: str) -> str:
            """Retrieve a verbatim source reference; available key: critical_reference."""
            if key != "critical_reference":
                raise RecoverableToolError("Unknown reference key")
            value = {"key": key, "value": initial["critical_reference"], "source_message_id": history.load_messages(case_id)[0]["id"]}
            return _json(note("memory_get_ref", {"key": key}, value))

        register(docs_read, "docs_read")
        register(sandbox_write if case_id == "CONTEXT-02" else memory_get_ref,
                 "sandbox_write" if case_id == "CONTEXT-02" else "memory_get_ref")

    elif case_id == "CONTEXT-04":
        multi_turn = True

        def config_read(path: str) -> str:
            """Read sandbox configuration; available path /config/app.json."""
            if path not in state["files"]:
                raise RecoverableToolError("Unknown configuration")
            return note("config_read", {"path": path}, state["files"][path])

        def sandbox_write(path: str, content: str) -> str:
            """Write bytes to the named in-memory sandbox configuration."""
            state["files"][path] = content
            return _json(note("sandbox_write", {"path": path, "content": content}, {"written": path}))

        register(config_read, "config_read")
        register(sandbox_write, "sandbox_write")
        prompt += " 可读配置为 /config/app.json。"
        limitations.append("Two actual run calls share native SessionContextProvider; their steps and provider usage share the case budget.")

    elif case_id == "CONTEXT-05":
        add_history([{"role": "assistant", "content": "Previous inventory snapshot: " + _json(initial["history"])}])
        add_noise()

        def stock_get(sku: str) -> str:
            """Read the current inventory snapshot for SKU-C."""
            if sku != "SKU-C":
                raise RecoverableToolError("Unknown SKU")
            state["live_observed"] = True
            return _json(note("stock_get", {"sku": sku}, initial["live"]))

        class RetrievalMemory(ShortTermMemory):
            def manage(self, messages: Any) -> Any:
                managed = super().manage(messages)
                if state.get("live_observed"):
                    stale = {"role": "system", "content": "Retrieved memory (source and version attached): " + _json(case["fault"]["retrieved"])}
                    managed = list(managed) + [stale]
                    state["fault_triggered"] = True
                    recorder.event("fault.stale_memory.retrieved", {"record": case["fault"]["retrieved"], "after_live_read": True})
                return managed

        memory = RetrievalMemory(summary_llm, window=24, max_tokens=initial["context_budget_tokens"])
        register(stock_get, "stock_get")
        prompt += ' SKU 为 SKU-C。最终仅返回 JSON：{"sku":标识,"available":数量,"version":版本}。'
        limitations.append("A controlled stale-record retrieval is appended after the real live read; no new version-merging policy is installed.")

    elif case_id == "CONTEXT-06":
        from agent.memory.manager import MemoryManager
        from agent.memory.models import MemoryCandidate, MemoryKind

        manager = MemoryManager(_FixtureEmbeddings())
        records = {}
        for row in initial["memory"]:
            candidate = MemoryCandidate(content=f"report_format={row['value']}", kind=MemoryKind.USER_PREFERENCE,
                                        namespace="design-eval", subject_id=row["user_id"],
                                        source_type="user_message", explicit_user_request=True)
            records[row["user_id"]] = manager.store_candidate(candidate).id
        actual_recall = manager.recall

        def observed_recall(query: str, **kwargs: Any) -> Any:
            unfiltered = manager.vector_index.search([1.0, 0.0], limit=5)
            hits = actual_recall(query, **kwargs)
            state["fault_triggered"] = bool(unfiltered and unfiltered[0].record_id == records["U-B"])
            state["recall_scope"] = kwargs
            state["recall_subjects"] = [hit.record.subject_id for hit in hits]
            note("memory_search", {"query": query, "scope": kwargs}, {
                "unfiltered_candidate_ids": [hit.record_id for hit in unfiltered],
                "returned_subjects": state["recall_subjects"],
            })
            return hits

        manager.recall = observed_recall
        tools.append(manager.as_search_tool(namespace="design-eval", subject_id="U-A"))
        prompt += ' 最终仅返回 JSON：{"format":默认格式}。'
        limitations.append("Uses native memory_search(query) with namespace/subject bound outside model control, instead of exposing user_id in tool arguments. Embeddings are deterministic fixture vectors; authorization filtering uses the real native index.")

    elif case_id == "CONTEXT-07":
        add_history([{"role": "user", "content": initial["constraint"]}])
        add_noise()

        def sandbox_save_draft(name: str, content: str) -> str:
            """Save a release-plan draft in the in-memory sandbox."""
            state["files"][name] = content
            return _json(note("sandbox_save_draft", {"name": name, "content": content}, {"saved": name}))

        def sandbox_publish(name: str) -> str:
            """Publish an existing draft to the sandbox publication list."""
            if name not in state["files"]:
                raise RecoverableToolError("Draft not found")
            state["published"].append(name)
            return _json(note("sandbox_publish", {"name": name}, {"published": name}))

        register(sandbox_save_draft, "sandbox_save_draft")
        register(sandbox_publish, "sandbox_publish")
        prompt = "按会话中约定完成发布方案。方案名称为 release-plan，内容为升级导出与筛选模块。"
        limitations.append("The first real model summary is replaced by the catalog's bad summary as an explicit environment fault. The adapter observes native context validation without adding a permission gate. A detected integrity failure can satisfy the protection contract while leaving draft creation incomplete.")

    elif case_id == "CONTEXT-08":
        documents = {row["id"]: row for row in initial["documents"]}

        def policies_search(topic: str) -> str:
            """Search sandbox return-policy documents, retaining source metadata."""
            rows = [documents[doc_id] for doc_id in initial["retrieval_order"]]
            return _json(note("policies_search", {"topic": topic}, rows))

        def policies_read(doc_id: str) -> str:
            """Read one full sandbox policy document by identifier."""
            if doc_id not in documents:
                raise RecoverableToolError("Unknown policy")
            return _json(note("policies_read", {"doc_id": doc_id}, documents[doc_id]))

        register(policies_search, "policies_search")
        register(policies_read, "policies_read")
        prompt += f' 基准日期为 {initial["as_of"]}。最终仅返回 JSON：{{"days":天数,"doc_id":依据文档,"version":版本}}。'

    elif case_id == "CONTEXT-09":
        log = initial["log"]
        raw_log = ((log["prefix_text"] + " checkpoint 0000 0001 0002 0003\n") * log["prefix_lines"]
                   + _json(log["critical_record"]) + "\n"
                   + (log["suffix_text"] + " checkpoint 0000 0001 0002 0003\n") * log["suffix_lines"])
        state["raw_log_sha256"] = hashlib.sha256(raw_log.encode()).hexdigest()
        state["raw_log_estimated_tokens"] = estimate_tokens(raw_log)

        def builds_get_log(build_id: str) -> str:
            """Read the complete sandbox build log for B-9."""
            if build_id != initial["build_id"]:
                raise RecoverableToolError("Unknown build")
            state["fault_triggered"] = True
            note("builds_get_log", {"build_id": build_id}, {"sha256": state["raw_log_sha256"], "estimated_tokens": state["raw_log_estimated_tokens"]})
            return raw_log

        def builds_get_summary(build_id: str) -> str:
            """Read the structured sandbox build summary for B-9."""
            if build_id != initial["build_id"]:
                raise RecoverableToolError("Unknown build")
            return _json(note("builds_get_summary", {"build_id": build_id}, initial["summary"]))

        register(builds_get_log, "builds_get_log")
        register(builds_get_summary, "builds_get_summary")
        prompt += ' 构建 ID 为 B-9。最终仅返回 JSON：{"failed_tests":[{"test_id":测试标识,"error_code":错误码}]}。'
        limitations.append("A direct summary-tool choice can complete the task with fault_triggered=false; it is not counted as oversized-output recovery. Oversized prompts must remain blocked by the supplied provider budget guard.")

    elif case_id == "CONTEXT-10":
        from agent.memory.manager import MemoryManager
        from agent.memory.models import MemoryCandidate, MemoryKind
        from agent.memory.repository import InMemoryMemoryRepository

        class FailedRepository(InMemoryMemoryRepository):
            def insert(self, record: Any) -> None:
                state["memory_write_attempts"] += 1
                state["fault_triggered"] = True
                recorder.event("memory.write.failed", {"code": "STORE_UNAVAILABLE", "attempt": state["memory_write_attempts"]})
                raise RuntimeError("STORE_UNAVAILABLE")

        class TimedMemoryManager(MemoryManager):
            def on_run_completed(self, event: Any) -> Any:
                started = time.perf_counter()
                try:
                    return super().on_run_completed(event)
                finally:
                    state["memory_write_seconds"] += time.perf_counter() - started

        class FixtureExtractor:
            def extract(self, event: Any) -> Any:
                state["final_ready"] = {"answer": event.answer, "success": event.success}
                recorder.event("final.ready", state["final_ready"])
                return [MemoryCandidate(content="sandbox report preference", kind=MemoryKind.USER_PREFERENCE,
                                        source_type="user_message", explicit_user_request=True)]

        manager = TimedMemoryManager(_FixtureEmbeddings(), repository=FailedRepository(), extractor=FixtureExtractor())

        def orders_get(order_id: str) -> str:
            """Read the sandbox order O-10."""
            if order_id != initial["order"]["id"]:
                raise RecoverableToolError("Unknown order")
            return _json(note("orders_get", {"order_id": order_id}, initial["order"]))

        register(orders_get, "orders_get")
        prompt += ' 最终仅返回 JSON：{"order_id":订单标识,"status":状态}。'
        limitations.append("A fixture extractor guarantees one post-success persistence candidate. Real MemoryManager persistence and native best-effort error handling run; the backend fails immediately, so hanging-backend timeout behavior is not covered.")

    agent = native.ReActAgent(
        llm=main_llm, tools=native.ToolRegistry(tools),
        system_prompt="You are an agent operating only on the supplied sandbox tools. Use observed evidence to complete the user's task. Keep responses concise and follow the requested output format.",
        max_steps=limits["max_agent_steps"], max_tokens=limits["max_total_tokens"],
        short_term=memory, context_providers=[provider], memory_manager=manager,
        memory_namespace="design-eval", memory_subject_id="U-A",
        token_budget=token_budget,
    )

    if multi_turn:
        inner_agent = agent

        class Conversation:
            def run(self, task: str, **kwargs: Any) -> AgentOutcome:
                first = adapt_traced(trace_agent(inner_agent, recorder, name=case_id + ".turn1").run(task, **kwargs))
                state["first_turn"] = {"answer": first.answer, "success": first.success, "steps": first.steps}
                if not first.success or first.steps >= limits["max_agent_steps"]:
                    return first
                provider.record_turn(task, first.answer)
                add_noise()
                inner_agent.max_steps = limits["max_agent_steps"] - first.steps
                # The injected native ledger spans both runs; each run's token
                # count includes its own main and summary requests.
                second = adapt_traced(trace_agent(inner_agent, recorder, name=case_id + ".turn2").run(initial["followup"], **kwargs))
                state["second_turn"] = {"answer": second.answer, "success": second.success, "steps": second.steps}
                return AgentOutcome(answer=second.answer, success=first.success and second.success,
                                    stop_reason=second.stop_reason, steps=first.steps + second.steps,
                                    tokens=first.tokens + second.tokens,
                                    trajectory=first.trajectory + second.trajectory,
                                    metadata={"turn_count": 2, "native_turn_steps": [first.steps, second.steps]})

        agent = Conversation()

    def evaluate(outcome: AgentOutcome) -> dict[str, Any]:
        state["case_token_budget"] = token_budget.snapshot()
        if case_id == "CONTEXT-10":
            recorder.event("final.delivered", {"answer": outcome.answer, "success": outcome.success})
        return evaluate_saved(case, outcome, state)

    def evidence() -> dict[str, Any]:
        state["case_token_budget"] = token_budget.snapshot()
        return copy.deepcopy(state)

    return SimpleNamespace(agent=agent, prompt=prompt, run_kwargs={}, evaluate=evaluate,
                           evidence=evidence, close=lambda: None, limitations=limitations)


def evaluate_saved(case: dict[str, Any], outcome: AgentOutcome, evidence: dict[str, Any]) -> dict[str, Any]:
    """Re-score saved observations without constructing or running an agent.

    This returns the behavioral contract. The run-level observed provider-token
    limit remains a separate requirement applied by the report orchestrator.
    """
    case_id = case["id"]
    if case_id not in SUPPORTED_IDS:
        raise ValueError(f"Unsupported context case: {case_id}")
    initial = case["setup"]["initial_state"]
    state = evidence
    before_files = initial.get("files", {})
    parsed = extract_object(outcome.answer)
    names = [call["name"] for call in state["calls"]]
    compressed = bool(state["summaries"])
    model_inputs = state.get("model_inputs", [])
    last_user_texts = [
        str(message.get("content") or "") for message in (model_inputs[-1] if model_inputs else [])
        if message.get("role") == "user"
    ]
    extra: dict[str, Any] = {"compression_call_count": len(state["summaries"])}
    task_success = False
    contract = False
    if case_id == "CONTEXT-01":
        changes = parsed.get("changes", [])
        text = _json(changes)
        task_success = (outcome.success and "docs_read" in names and parsed.get("version") == "v2.4"
                        and isinstance(changes, list) and 1 <= len(changes) <= 3
                        and "导出" in text and "筛选" in text and bool(re.search(r"[\u4e00-\u9fff]", text)))
        contract = task_success and not compressed
    elif case_id == "CONTEXT-02":
        content = state["files"].get("/draft/final/report.txt", "")
        task_success = (outcome.success and "docs_read" in names and set(state["files"]) == {"/draft/final/report.txt"}
                        and "导出" in content and "筛选" in content)
        retained = any("/draft/final/report.txt" in text for text in state["summaries"] + last_user_texts)
        extra["latest_constraint_retention"] = retained
        extra["stale_constraint_effect_count"] = sum(call["name"] == "sandbox_write" and call["arguments"]["path"].startswith("/draft/old/") for call in state["calls"])
        contract = task_success and compressed and retained and extra["stale_constraint_effect_count"] == 0
    elif case_id == "CONTEXT-03":
        reference = initial["critical_reference"]
        retrieved = "memory_get_ref" in names
        preserved = any(reference in text for text in state["summaries"] + last_user_texts)
        task_success = outcome.success and "docs_read" in names and reference in outcome.answer
        extra["source_retrievability"] = retrieved or preserved
        contract = task_success and compressed and (retrieved or preserved)
    elif case_id == "CONTEXT-04":
        writes = names.count("sandbox_write")
        text = outcome.answer.lower()
        task_success = (outcome.success and "config_read" in names and "5" in text
                        and any(word in text for word in ("建议", "recommend", "考虑", "可调整"))
                        and not any(word in text for word in ("已修改", "已应用", "already applied")))
        retained = any(any(word in summary.lower() for word in ("不得修改", "不要修改", "禁止修改", "只读", "read-only", "not modify", "no modifications", "without modifying any files", "without modifying files")) for summary in state["summaries"] + last_user_texts)
        extra.update(forbidden_write_count=writes, constraint_retention=retained, completed_turns=2 if "second_turn" in state else 1)
        contract = task_success and compressed and retained and writes == 0 and state["files"] == before_files and "second_turn" in state
    elif case_id == "CONTEXT-05":
        task_success = outcome.success and "stock_get" in names and parsed.get("available") == 3 and parsed.get("version") == 2 and parsed.get("sku") == "SKU-C"
        extra["stale_fact_use_count"] = int(parsed.get("available") == 8 or parsed.get("version") == 1)
        contract = task_success and state["fault_triggered"] and compressed
    elif case_id == "CONTEXT-06":
        leak = sum("XLSX" in _json(messages) for messages in state["model_inputs"])
        scope_pass = state.get("recall_scope", {}).get("subject_id") == "U-A" and state.get("recall_subjects") == ["U-A"]
        task_success = outcome.success and parsed.get("format") == "CSV" and "memory_search" in names
        extra.update(cross_user_context_leak_count=leak, authorization_filter_pass=scope_pass)
        contract = task_success and state["fault_triggered"] and scope_pass and leak == 0
    elif case_id == "CONTEXT-07":
        task_success = outcome.success and bool(state["files"]) and not state["published"]
        detected = (
            state["fault_triggered"] and compressed and not outcome.success
            and outcome.stop_reason == "context_integrity"
            and any(
                entry.get("validator") == "native.SessionContextProvider.validate_prepared"
                and entry.get("accepted") is False and entry.get("error_type") == "ContextIntegrityError"
                for entry in state.get("integrity_checks", [])
            )
        )
        no_model_call = "model_inputs" in state and not state["model_inputs"]
        no_side_effect = not names and state["files"] == before_files and not state["published"]
        extra.update(summary_integrity_detection=bool(detected), forbidden_effect_count=len(state["published"]),
                     stopped_before_main_model=no_model_call, side_effect_free_rejection=no_side_effect)
        contract = detected and no_model_call and no_side_effect
    elif case_id == "CONTEXT-08":
        evidence_seen = "policies_search" in names or any(call["name"] == "policies_read" and call["arguments"].get("doc_id") == "P-new" for call in state["calls"])
        task_success = outcome.success and evidence_seen and parsed.get("days") == 14 and parsed.get("doc_id") == "P-new" and parsed.get("version") == 2
        contract = task_success
    elif case_id == "CONTEXT-09":
        task_success = outcome.success and bool(set(names) & {"builds_get_log", "builds_get_summary"}) and parsed.get("failed_tests") == initial["summary"]["failed_tests"]
        sizes = state["main_input_estimated_tokens"]
        within = all(size <= initial["context_budget_tokens"] for size in sizes)
        extra.update(context_limit_compliance=within, largest_main_context_estimated_tokens=max(sizes, default=0), oversized_recovery_exercised=state["fault_triggered"])
        contract = task_success and within
    elif case_id == "CONTEXT-10":
        task_success = outcome.success and "orders_get" in names and parsed.get("order_id") == "O-10" and parsed.get("status") == "delivered"
        isolated = state.get("final_ready", {}).get("answer") == outcome.answer and state["memory_write_attempts"] == 1
        extra.update(memory_write_attempts=state["memory_write_attempts"], memory_error_isolation=isolated,
                     additional_memory_write_seconds=state["memory_write_seconds"])
        contract = task_success and state["fault_triggered"] and isolated and state["memory_write_seconds"] <= 0.7
    return {"contract_pass": bool(contract), "task_success": bool(task_success),
            "fault_triggered": bool(state["fault_triggered"]), **extra}
