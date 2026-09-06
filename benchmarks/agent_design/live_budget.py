"""Run budget contracts against the real loop with live or controlled models.

Exact usage and race boundaries need deterministic stimuli. Those cases never
call a provider and are reported separately from the live-model cases.
"""
from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

from adapters.trace_agent import adapt_traced, trace_agent


SUPPORTED_IDS = {f"BUDGET-{i:02d}" for i in range(1, 11)}
UNSUPPORTED = {}
DETERMINISTIC_IDS = SUPPORTED_IDS - {"BUDGET-01", "BUDGET-02"}


def slow_read(report_id: str, cancel_token: str = "") -> str:
    """Read R-6 from a twenty-second read-only service in an isolated process."""
    time.sleep(20)
    return "R-6 complete"


def build_case(case, native, llm, recorder):
    from agent.errors import FatalToolError, RecoverableToolError
    from agent.llm import LLMResponse, ToolCall, Usage
    from agent.state.memory import ShortTermMemory

    case_id = case["id"]
    events = []
    state = {"fault_triggered": False}
    tools = []
    limits = case["setup"]["limits"]
    run_kwargs = {}
    limitations = []
    prompt = case["prompt"]
    custom_agent = None
    agent_options = {}
    close = lambda: None

    def emit(name, **data):
        event = {"event": name, **data}
        events.append(event)
        recorder.event("fixture." + name, data)

    def response(content="", calls=(), usage=(10, 2)):
        return LLMResponse(content=content, tool_calls=list(calls), usage=Usage(*usage, estimated=True))

    def call(name, args=None, usage=(10, 2)):
        return response(calls=[ToolCall(f"call-{len(events)}", name, args or {})], usage=usage)

    class ControlledLLM:
        # These fixed responses validate a complete reservation; truncating
        # one would change the benchmark stimulus instead of its scheduling.
        supports_output_shrink = False

        def __init__(self, make_response):
            self.count = 0
            self.make_response = make_response
            self.max_output_tokens = 2

        def estimate_input_tokens(self, messages, tools=None):
            # Preview constructs data only: no model-call count, tool effects,
            # usage event, or trace span is produced for a rejected reservation.
            preview = self.make_response(self.count + 1, messages)
            self.max_output_tokens = max(1, preview.usage.completion_tokens)
            return preview.usage.prompt_tokens

        def chat(self, messages, tools=None):
            return self._respond(messages)

        def chat_with_budget(self, messages, tools=None, *, max_output_tokens):
            preview = self.make_response(self.count + 1, messages)
            if preview.usage.completion_tokens > max_output_tokens:
                raise AssertionError("Controlled response exceeds the runtime's reserved completion cap")
            return self._respond(messages)

        def _respond(self, messages):
            self.count += 1
            emit("scripted_model", index=self.count)
            result = self.make_response(self.count, messages)
            emit("controlled_usage", total_tokens=result.usage.total_tokens)
            return result

    def register(fn):
        tools.append(native.tool(fn))
        return fn

    if case_id == "BUDGET-01":
        prompt += '\n最终仅返回 JSON 对象，键为工单ID，值为工具返回的状态。'
        limitations.append("最终回答增加显式JSON映射契约，以独立核验工单与状态配对。")
        @register
        def tickets_read(ticket_id: str) -> str:
            """Read one ticket status, using ticket_id T1 or T2."""
            emit("read", ticket_id=ticket_id)
            return json.dumps({"ticket_id": ticket_id, "status": {"T1": "open", "T2": "closed"}[ticket_id]})

        def evaluate(outcome):
            ids = {event.get("ticket_id") for event in events if event["event"] == "read"}
            try:
                answer = json.loads(outcome.answer)
            except (ValueError, TypeError):
                answer = None
            correct = ids == {"T1", "T2"} and answer == {"T1": "open", "T2": "closed"}
            return {"task_success": outcome.success and correct, "fault_triggered": False,
                    "contract_pass": outcome.success and correct and outcome.tokens <= limits["max_total_tokens"]}

    elif case_id == "BUDGET-02":
        observed = []

        @register
        def chain_fetch(node_id: str, previous_token: str = "") -> str:
            """Read dependency chain L1 -> L2 -> L3, passing the preceding returned token."""
            emit("chain", node_id=node_id, previous_token=previous_token)
            required = {"L1": "", "L2": "K1", "L3": "K2"}
            if node_id not in required or previous_token != required[node_id] or (node_id != "L1" and str(int(node_id[-1]) - 1) not in [n[-1] for n in observed]):
                raise RecoverableToolError("Read the predecessor and pass its returned token")
            observed.append(node_id)
            if node_id == "L3":
                state["fault_triggered"] = True
            return json.dumps({"L1": {"token": "K1"}, "L2": {"token": "K2"}, "L3": {"result_code": "READY-3"}}[node_id])

        def evaluate(outcome):
            success = outcome.success and "READY-3" in outcome.answer and observed == ["L1", "L2", "L3"]
            near = len(outcome.trajectory) >= 3 and sum(bool(s.calls) for s in outcome.trajectory) == 3
            return {"task_success": success, "fault_triggered": state["fault_triggered"] and near,
                    "contract_pass": success and len(events) == 3 and outcome.steps <= 4 and outcome.tokens <= 3000}

    elif case_id in {"BUDGET-03", "BUDGET-04"}:
        @register
        def jobs_get(job_id: str) -> str:
            emit("poll", job_id=job_id)
            state["fault_triggered"] = True
            return json.dumps({"state": "running", "progress": 0})

        @register
        def docs_search(query: str, page: int = 1) -> str:
            emit("search", query=query, page=page)
            state["fault_triggered"] = True
            return json.dumps({"documents": [], "revision": 1})

        tools = [tools[0] if case_id == "BUDGET-03" else tools[1]]
        llm = ControlledLLM(lambda n, messages: call("jobs_get", {"job_id": "J-3"}) if case_id == "BUDGET-03"
                            else call("docs_search", {"query": f"E404 variant {n}", "page": n}))
        if case_id == "BUDGET-04":
            from agent.progress import ObservationProgress
            agent_options["progress_policy"] = ObservationProgress(
                read_only_tools=["docs_search"], repeated_observation_limit=4,
            )
            limitations.append("显式声明docs_search为只读，使用运行时的观察内容停滞检测；不把相同写入回执判断为无进展。")

        def evaluate(outcome):
            count = sum(e["event"] in {"poll", "search"} for e in events)
            guard = "loop_detected" in outcome.stop_reason
            threshold = 3 if case_id == "BUDGET-03" else 4
            honest = not outcome.success and ("running" in outcome.answer or "documents" in outcome.answer)
            return {"task_success": None, "fault_triggered": state["fault_triggered"], "business_calls": count,
                    "loop_guard_triggered": guard, "honest_failure": honest,
                    "contract_pass": guard and count <= threshold and honest}

    elif case_id == "BUDGET-05":
        @register
        def notes_read(note_id: str) -> str:
            """Read N-5."""
            emit("read", note_id=note_id)
            state["fault_triggered"] = True
            return "N-5: 本周部署完成"

        llm = ControlledLLM(lambda n, messages: call("notes_read", {"note_id": "N-5"}, usage=(700, 50)) if n == 1
                            else response("部署完成", usage=(180, 100)))
        limitations.append("受控模型精确预告首请求700输入+50输出、后续180输入+100输出；预演不产生调用和用量。chat_with_budget遵守运行时输出cap；测试预留不足时拒绝请求的路径。")

        def evaluate(outcome):
            return {"task_success": None, "fault_triggered": state["fault_triggered"],
                    "controlled_usage": outcome.tokens, "model_calls": llm.count,
                    "contract_pass": state["fault_triggered"] and llm.count == 1 and outcome.tokens <= 1000}

    elif case_id == "BUDGET-06":
        from agent.tool_execution import ProcessToolExecution
        register(slow_read)

        def observe_process(event):
            emit(event["event"], **{key: value for key, value in event.items() if key != "event"})
            if event["event"] == "tool.process.started":
                state["fault_triggered"] = True
            if event["event"] == "tool.process.completed":
                state["tool_elapsed"] = event["elapsed_seconds"]
                state["process_reaped"] = event.get("reaped") is True
                state["process_outcome"] = event.get("outcome")

        agent_options["tool_execution"] = ProcessToolExecution(
            3, ["slow_read"], on_event=observe_process,
        )

        llm = ControlledLLM(lambda n, messages: call("slow_read", {"report_id": "R-6"}) if n == 1 else response("R-6 complete"))
        limitations.append("显式启用ProcessToolExecution(3秒)运行模块级20秒只读工具；父进程观察真实ready/终止/reap事件，避免序列化TraceRecorder或丢失闭包状态。外层120秒实验保护不计为通过。")

        def evaluate(outcome):
            elapsed = state.get("run_elapsed")
            compliant = isinstance(elapsed, (int, float)) and 0 < elapsed <= 3.2
            reaped = state.get("process_reaped") is True
            return {"task_success": None, "fault_triggered": state["fault_triggered"],
                    "tool_elapsed_seconds": state.get("tool_elapsed"), "run_elapsed_seconds": elapsed,
                    "deadline_compliance": compliant, "process_reaped": reaped,
                    "contract_pass": state["fault_triggered"] and compliant and reaped
                    and state.get("process_outcome") == "timed_out" and outcome.stop_reason == "tool_timeout"}

    elif case_id == "BUDGET-07":
        cancelled = threading.Event()
        committed = []
        timer = None

        @register
        def batch_start(batch_id: str, cancel_token: str = "") -> str:
            """Process B-7, honoring the runtime's injected cancellation signal."""
            nonlocal timer
            began = time.monotonic()

            def cancel():
                state["fault_triggered"] = True
                state["cancelled_at"] = time.monotonic()
                emit("cancel.requested", elapsed=state["cancelled_at"] - began)
                cancelled.set()

            timer = threading.Timer(1.1, cancel)
            timer.start()
            for record_id in ("R1", "R2", "R3", "R4"):
                emit("record.started", record_id=record_id, cancelled=cancelled.is_set())
                if cancelled.wait(1):
                    state["ack_delay"] = time.monotonic() - state["cancelled_at"]
                    emit("cancel.acknowledged", delay=state["ack_delay"])
                    return json.dumps({"state": "cancelled", "committed": committed})
                committed.append(record_id)
                emit("record.committed", record_id=record_id)
            return json.dumps({"state": "completed", "committed": committed})

        llm = ControlledLLM(lambda n, messages: call("batch_start", {"batch_id": "B-7"}))
        run_kwargs["cancellation_event"] = cancelled
        close = lambda: timer.join(2) if timer is not None else None
        limitations.append("在途工具按协议合作检查cancel event；本例不证明无法取消的外部I/O会被强杀。")

        def evaluate(outcome):
            return {"task_success": None, "fault_triggered": state["fault_triggered"],
                    "committed_count": len(committed), "cancel_ack_seconds": state.get("ack_delay"),
                    "contract_pass": outcome.stop_reason == "cancelled" and state.get("ack_delay", 999) <= .5
                    and committed == ["R1"] and not any(e.get("cancelled") for e in events)}

    elif case_id == "BUDGET-08":
        from agent.multi_agent import AgentRegistry, AgentSpec, MultiAgentOrchestrator, RunBudget
        registry = AgentRegistry()
        orchestrator = MultiAgentOrchestrator(registry, RunBudget(max_parallel_tasks=2, max_total_tokens=3000))

        class RootAgent:
            def run(self, prompt):
                with recorder.span("shared_budget_test", "orchestration") as root_span:
                    class Worker(native.ReActAgent):
                        def run(self, prompt, **kwargs):
                            with recorder.span(self.agent_name, "worker", parent_span_id=root_span.span_id):
                                return trace_agent(SimpleNamespace(agent=self, run=super().run), recorder, name=self.agent_name).run(prompt, **kwargs)

                    registry.register(AgentSpec("A", "A"), lambda: Worker(ControlledLLM(lambda n, m: response("A done", usage=(1000, 200))), native.ToolRegistry(), max_tokens=3000, agent_name="A"))
                    registry.register(AgentSpec("B", "B"), lambda: Worker(ControlledLLM(lambda n, m: response("B done", usage=(500, 1000))), native.ToolRegistry(), max_tokens=3000, agent_name="B"))

                    def leader_response(n, messages):
                        if n == 1:
                            return response(calls=[ToolCall("a", "spawn_subagent", {"role": "A", "task": "A"}), ToolCall("b", "spawn_subagent", {"role": "B", "task": "B"})], usage=(400, 100))
                        if n == 2:
                            ids = [json.loads(m["content"])["task_id"] for m in messages if m.get("role") == "tool" and m.get("name") == "spawn_subagent"]
                            return call("wait_subagents", {"task_ids": ids, "timeout_seconds": 5}, usage=(1, 0))
                        return response("Shared budget stopped one worker; the combined task is incomplete.", usage=(100, 100))

                    leader = native.ReActAgent(ControlledLLM(leader_response), native.ToolRegistry(orchestrator.leader_tools()), max_steps=6, max_tokens=3000)
                    result = orchestrator.run_leader(trace_agent(leader, recorder), prompt)
                    state.update(fault_triggered=len(result.subagents) == 2, root_tokens=result.tokens,
                                 worker_tokens=[w.tokens for w in result.subagents],
                                 worker_success=[w.success for w in result.subagents],
                                 worker_stop_reasons=[w.stop_reason for w in result.subagents],
                                 independently_observed_tokens=sum(e["total_tokens"] for e in events if e["event"] == "controlled_usage"))
                    emit("aggregate_usage", **state)
                    return adapt_traced(result)

        custom_agent = RootAgent()
        close = orchestrator.close
        limitations.append("显式启用RunBudget(max_total_tokens=3000)；受控Leader与Worker精确预告输入和输出上界，由真实运行时在请求前共享预留。wait轮用量从0调整为1输入token以消除缺失usage估算歧义，未受限总需求为3401；一个Worker应因预算不足被拒，业务任务不算成功。")

        def evaluate(outcome):
            observed = state.get("independently_observed_tokens")
            complete_usage = isinstance(observed, int) and observed > 0
            accurate = complete_usage and state.get("root_tokens") == observed and outcome.tokens == observed
            rejected = any("budget" in reason for reason in state.get("worker_stop_reasons", []))
            task_success = len(state.get("worker_success", [])) == 2 and all(state["worker_success"])
            return {"task_success": task_success, "fault_triggered": state["fault_triggered"],
                    "root_controlled_tokens": observed, "budget_rejected_worker": rejected,
                    "usage_aggregation_correct": accurate,
                    "contract_pass": complete_usage and observed <= 3000 and state["fault_triggered"]
                    and accurate and rejected and not task_success}

    elif case_id == "BUDGET-09":
        from agent.retry import RetryPolicy, TransientLLMError, call_with_retry

        class ServiceTimeout(Exception):
            status_code = 503

        @register
        def status_get(id: str) -> str:
            """Read S-9 through the existing provider deadline/retry component."""
            began = time.monotonic()

            def attempt(timeout):
                state["fault_triggered"] = True
                emit("attempt.started", timeout=timeout)
                with recorder.span("status.attempt", "tool_attempt", input={"timeout": timeout}) as span:
                    time.sleep(min(2, timeout))
                    emit("attempt.failed", code="TEMPORARY_FAILURE")
                    raise ServiceTimeout("TEMPORARY_FAILURE")

            def wait(seconds):
                emit("backoff", seconds=seconds)
                time.sleep(seconds)

            try:
                call_with_retry(attempt, policy=RetryPolicy(timeout_seconds=2, max_attempts=4, initial_backoff=2,
                    backoff_multiplier=1, jitter=0, total_deadline_seconds=5), operation="status.get", sleep=wait)
            except TransientLLMError as error:
                state["elapsed"] = time.monotonic() - began
                state["error"] = str(error)
                raise FatalToolError(str(error)) from None
            return "unreachable"

        llm = ControlledLLM(lambda n, messages: call("status_get", {"id": "S-9"}))
        limitations.append("显式把现有call_with_retry组件配置到沙箱status工具；测该配置，不宣称所有工具自动继承deadline。使用真实墙钟。")

        def evaluate(outcome):
            count = sum(e["event"] == "attempt.started" for e in events)
            return {"task_success": None, "fault_triggered": state["fault_triggered"], "attempt_count": count,
                    "elapsed_seconds": state.get("elapsed", 999), "contract_pass": state["fault_triggered"] and
                    state.get("elapsed", 999) <= 5.2 and count <= 2 and "deadline" in state.get("error", "") and "TEMPORARY_FAILURE" in state.get("error", "")}

    else:
        imported = []

        @register
        def imports_validate(id: str) -> str:
            """Validate I-10 before importing."""
            emit("validation.failed", id=id)
            state["fault_triggered"] = True
            raise FatalToolError("CHECKSUM_MISMATCH")

        @register
        def imports_write(id: str) -> str:
            """Import I-10 after successful validation."""
            imported.append(id)
            emit("write", id=id)
            return "written"

        llm = ControlledLLM(lambda n, messages: response(calls=[ToolCall("validate", "imports_validate", {"id": "I-10"}), ToolCall("write", "imports_write", {"id": "I-10"})]))

        def evaluate(outcome):
            return {"task_success": None, "fault_triggered": state["fault_triggered"], "unauthorized_effect_count": len(imported),
                    "contract_pass": state["fault_triggered"] and not imported and llm.count == 1 and outcome.stop_reason == "fatal_tool_error" and "CHECKSUM_MISMATCH" in outcome.answer}

    agent = custom_agent or native.ReActAgent(llm, native.ToolRegistry(tools), max_steps=limits["max_agent_steps"],
        max_tokens=limits["max_total_tokens"], system_prompt="Use the supplied tools to complete the task. Return observed facts; never invent successful work.",
        short_term=ShortTermMemory(llm, window=100, max_tokens=limits["max_total_tokens"]), max_tool_retries=1,
        **agent_options)
    if case_id == "BUDGET-06":
        native_agent = agent

        def timed_run(prompt, **kwargs):
            began = time.monotonic()
            try:
                return native_agent.run(prompt, **kwargs)
            finally:
                state["run_elapsed"] = time.monotonic() - began

        agent = SimpleNamespace(agent=native_agent, run=timed_run)
    return SimpleNamespace(agent=agent, prompt=prompt, run_kwargs=run_kwargs, evaluate=evaluate,
        evidence=lambda: {"events": events, "state": state}, close=close, limitations=limitations,
        execution_mode="deterministic_runtime" if case_id in DETERMINISTIC_IDS else "live_model")
