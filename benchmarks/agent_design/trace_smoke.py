"""Five offline runtime demonstrations, separate from the 50 case specifications.

Fixed model responses drive real ReActAgent/Orchestrator code. These checks
demonstrate instrumentation and mechanism contracts, not model capability or
an architecture-ablation score. All state and tools are local fixtures.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from adapters.trace_agent import trace_agent
from agent_eval.tracing import TraceRecorder, read_trace


class ScriptedLLM:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = iter(responses)
        self.requests: list[dict[str, Any]] = []

    def chat(self, messages: Any, tools: Any = None) -> Any:
        self.requests.append(copy.deepcopy({"messages": messages, "tools": tools}))
        return next(self._responses)


def run_smoke(directory: str | Path, harness_path: str | Path) -> dict[str, Any]:
    root = Path(harness_path).resolve(strict=True)
    if not (root / "agent" / "agent.py").is_file():
        raise ValueError("harness_path must contain the companion agent package")
    existing = sys.modules.get("agent")
    if existing is not None and Path(existing.__file__).resolve().parent != root / "agent":
        raise ValueError("A different agent package is already imported")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from agent import ReActAgent, ToolRegistry, tool
    from agent.errors import FatalToolError, RecoverableToolError
    from agent.llm import LLMResponse, ToolCall, Usage
    from agent.state.memory import ShortTermMemory

    directory = Path(directory).resolve()
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError("Smoke output directory already exists; choose a new --out directory") from error
    rows = []

    def answer(text: str) -> Any:
        return LLMResponse(content=text, usage=Usage(10, 2, estimated=True))

    def invoke(name: str, arguments: Any = None, call_id: str = "call-1") -> Any:
        return LLMResponse(tool_calls=[ToolCall(call_id, name, arguments or {})], usage=Usage(10, 2, estimated=True))

    def run_one(case_id: str, agent: Any, prompt: str, check: Any) -> None:
        recorder = TraceRecorder(directory / "traces", task_id=case_id, condition="scripted_real_runtime")
        try:
            result = trace_agent(agent, recorder, name=case_id).run(prompt)
            passed, evidence = check(result)
            recorder.event("smoke.oracle", {"passed": passed, "evidence": evidence})
        finally:
            recorder.close()
        rows.append({"id": case_id, "passed": passed, "agent_success": result.success,
                     "stop_reason": result.stop_reason, "evidence": evidence,
                     "trace_path": str(recorder.path), "trace_id": recorder.trace_id,
                     "trace_complete": read_trace(recorder.path)["complete"]})

    recovery_attempts = []

    @tool
    def lookup() -> str:
        recovery_attempts.append(1)
        if len(recovery_attempts) == 1:
            raise RecoverableToolError("Injected temporary read failure")
        return "record-42"

    recovery_llm = ScriptedLLM([invoke("lookup"), answer("The failure is temporary; retry once."),
                               invoke("lookup", call_id="retry-1"), answer("record-42")])
    run_one("smoke.tool_recovery", ReActAgent(recovery_llm, ToolRegistry([lookup]), max_steps=6),
            "Read the record; one transient read failure is expected.",
            lambda result: (result.success and len(recovery_attempts) == 2,
                            {"actual_attempts": len(recovery_attempts), "answer": result.answer}))

    writes: dict[str, str] = {}
    write_attempts = []

    @tool
    def create_record(idempotency_key: str) -> str:
        write_attempts.append(idempotency_key)
        if idempotency_key not in writes:
            writes[idempotency_key] = "created-record-1"
            raise RecoverableToolError("Injected lost response after the write committed")
        return writes[idempotency_key]

    create_args = {"idempotency_key": "request-1"}
    state_llm = ScriptedLLM([invoke("create_record", create_args), answer("Retry with the same idempotency key."),
                            invoke("create_record", create_args, "retry-2"), answer("created-record-1")])
    run_one("smoke.state_idempotency", ReActAgent(state_llm, ToolRegistry([create_record]), max_steps=6),
            "Create one record for request-1; retries must retain its idempotency key.",
            lambda result: (result.success and len(writes) == 1 and len(write_attempts) == 2,
                            {"state": dict(writes), "attempted_keys": list(write_attempts),
                             "idempotency_owner": "fixture tool; this smoke does not prove automatic agent-level deduplication"}))

    @tool
    def compute() -> str:
        return "1056"

    run_one("smoke.budget_stop", ReActAgent(ScriptedLLM([invoke("compute")]), ToolRegistry([compute]), max_steps=1),
            "Compute and return the result.",
            lambda result: (not result.success and "max_steps (1)" in result.stop_reason,
                            {"computed_value": result.trajectory[0]["observation"], "answer": result.answer,
                             "expected_contract": "Budget guard stops execution; successful delivery is not claimed"}))

    @tool
    def read_project() -> str:
        return "Current project: ALPHA"

    context_llm = ScriptedLLM([invoke("read_project"), answer("Preserve the user's project ALPHA."), answer("ALPHA")])

    class EarlierContext:
        def prepare(self, task: str) -> list[dict[str, Any]]:
            return [{"role": "assistant", "content": "An earlier inspection identified the project workspace."}]

    def check_context(result: Any) -> tuple[bool, dict[str, Any]]:
        observed = (len(context_llm.requests) == 3
                    and "Summarize" in context_llm.requests[1]["messages"][0]["content"]
                    and "Summary of earlier conversation" in str(context_llm.requests[2]["messages"]))
        return result.success and result.answer == "ALPHA" and observed, {
            "answer": result.answer, "summary_and_final_prompts_recorded": observed,
            "limitation": "Summary wording is scripted; this tests the context path, not retention quality",
        }

    run_one("smoke.context_summary", ReActAgent(context_llm, ToolRegistry([read_project]),
            short_term=ShortTermMemory(context_llm, window=2), context_providers=[EarlierContext()]),
            "Find project ALPHA.", check_context)

    from agent.multi_agent import AgentRegistry, AgentSpec, MultiAgentOrchestrator, RunBudget
    recorder = TraceRecorder(directory / "traces", task_id="smoke.a2a_failure_isolation",
                             condition="scripted_real_runtime")
    registry = AgentRegistry()
    orchestrator = MultiAgentOrchestrator(registry, RunBudget(max_parallel_tasks=2, subagent_timeout_seconds=10))
    try:
        with recorder.span("a2a.failure_isolation", "a2a") as root_span:
            # The native registry deliberately requires isinstance(ReActAgent).
            # This explicit subclass keeps that contract while routing its
            # original bound run through the reusable adapter.
            class TracedWorker(ReActAgent):
                def run(self, prompt: str, *args: Any, **kwargs: Any) -> Any:
                    with recorder.span("worker.scope", "a2a", parent_span_id=root_span.span_id):
                        entry = SimpleNamespace(agent=self, run=super().run)
                        return trace_agent(entry, recorder, name=self.agent_name).run(prompt, *args, **kwargs)

            @tool
            def broken_source() -> str:
                raise FatalToolError("Injected worker-local fatal failure")

            registry.register(AgentSpec("broken", "Fails inside one worker"),
                              lambda: TracedWorker(ScriptedLLM([invoke("broken_source")]),
                                                   ToolRegistry([broken_source]), agent_name="broken"))
            registry.register(AgentSpec("healthy", "Independent working source"),
                              lambda: TracedWorker(ScriptedLLM([answer("verified-42")]),
                                                   ToolRegistry(), agent_name="healthy"))

            class LeaderLLM:
                def __init__(self) -> None:
                    self.turn = 0

                def chat(self, messages: Any, tools: Any = None) -> Any:
                    self.turn += 1
                    if self.turn == 1:
                        return LLMResponse(tool_calls=[
                            ToolCall("spawn-broken", "spawn_subagent", {"role": "broken", "task": "Read source A"}),
                            ToolCall("spawn-healthy", "spawn_subagent", {"role": "healthy", "task": "Read source B"}),
                        ], usage=Usage(10, 2, estimated=True))
                    if self.turn == 2:
                        task_ids = [json.loads(message["content"])["task_id"] for message in messages
                                    if message.get("role") == "tool" and message.get("name") == "spawn_subagent"]
                        return invoke("wait_subagents", {"task_ids": task_ids, "timeout_seconds": 5.0}, "wait-all")
                    return answer("Source A failed; source B returned verified-42.")

            leader = ReActAgent(LeaderLLM(), ToolRegistry(orchestrator.leader_tools()), max_steps=5)
            result = orchestrator.run_leader(trace_agent(leader, recorder, name="leader"), "Collect both sources.")
            worker_results = {worker.agent_name: worker.to_dict(include_trajectory=True) for worker in result.subagents}
            passed = result.success and len(worker_results) == 2 and not worker_results["broken"]["success"] and worker_results["healthy"]["success"]
            evidence = {"root_run_id": result.root_run_id, "workers": worker_results,
                        "tokens_including_workers": result.tokens,
                        "transport": "in-process MultiAgentOrchestrator; no external A2A protocol compatibility is claimed"}
            recorder.event("a2a.results", evidence)
            root_span.finish(output=evidence, status="ok" if passed else "error")
    finally:
        orchestrator.close()
        recorder.close()
    rows.append({"id": "smoke.a2a_failure_isolation", "passed": passed, "agent_success": result.success,
                 "stop_reason": result.stop_reason, "evidence": evidence,
                 "trace_path": str(recorder.path), "trace_id": recorder.trace_id,
                 "trace_complete": read_trace(recorder.path)["complete"]})
    records = [{"task_id": row["id"], "condition": "scripted_real_runtime", "trial_id": 0,
                "trace": {"trace_id": row["trace_id"], "path": row["trace_path"]},
                "scores": {}, "metric_statuses": {}, "execution_error": None,
                "smoke_only": True, "smoke_contract_passed": row["passed"]} for row in rows]
    report = {"schema_version": 1, "mode": "offline_scripted_real_runtime", "model_api_calls": 0,
              "is_architecture_ablation": False, "case_count": len(rows),
              "smoke_only": True, "records": records,
              "passed": sum(row["passed"] and row["trace_complete"] for row in rows), "cases": rows}
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", "--output", dest="output", default="results/agent-design-trace-smoke")
    parser.add_argument("--harness-path", default=str(Path(__file__).resolve().parents[3] / "agent" / "agent-harness-from-scratch"))
    args = parser.parse_args()
    result = run_smoke(args.output, args.harness_path)
    print(json.dumps({"passed": result["passed"], "case_count": result["case_count"],
                      "model_api_calls": 0, "report": str(Path(args.output).resolve() / "report.json")}, ensure_ascii=False))
    return 0 if result["passed"] == result["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
