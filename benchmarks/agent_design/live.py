"""Execute declared design scenarios, keeping live and controlled runs distinct.

One durable experiment per case permits isolated concurrent execution without
sharing mutable fixtures. Unsupported capabilities stay in the planned report.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import threading
import time

from agent_eval.experiments import create_experiment, fingerprint, read_records, run_experiment
from agent_eval.score_reporting import write_json_report
from agent_eval.trace_viewer import export_trace_html
from agent_eval.tracing import TraceRecorder, read_trace, sanitize_trace_value
from agent_eval.types import AgentOutcome
from adapters.trace_agent import adapt_traced, trace_agent
from .catalog import load_catalog, CATEGORY_LABELS


MODULES = {"tools": "live_tools", "state": "live_state", "budget": "live_budget", "context": "live_context"}
CONDITION = "current_agent"


class AccountedLLM:
    """Flush provider accounting per call, including calls before a hard kill."""
    def __init__(self, llm, client, recorder):
        self.llm, self.client, self.recorder = llm, client, recorder
        self.lock = threading.Lock()

    def chat(self, messages, tools=None):
        return self._chat(messages, tools)

    @property
    def max_output_tokens(self):
        return self.client.max_output_tokens

    @property
    def supports_output_shrink(self):
        return callable(getattr(self.llm, "chat_with_budget", None)) and getattr(self.llm, "supports_output_shrink", True) is not False

    def estimate_input_tokens(self, messages, tools=None):
        from agent.token_budget import estimate_input_tokens
        estimator = getattr(self.llm, "estimate_input_tokens", None)
        return estimator(messages, tools=tools) if estimator else estimate_input_tokens(messages, tools)

    def chat_with_budget(self, messages, tools=None, *, max_output_tokens):
        return self._chat(messages, tools, max_output_tokens)

    def _chat(self, messages, tools=None, max_output_tokens=None):
        with self.lock:
            before = len(self.client.calls)
            allowed = (not self.client.failure and before < self.client.max_calls
                       and sum(c["usage"].get("total_tokens") or 0 for c in self.client.calls) < self.client.max_tokens)
            if allowed:
                self.recorder.event("provider.request.started", {"call_index": before})
            try:
                if max_output_tokens is not None:
                    return self.llm.chat_with_budget(messages, tools=tools, max_output_tokens=max_output_tokens)
                return self.llm.chat(messages, tools=tools)
            finally:
                for call in self.client.calls[before:]:
                    self.recorder.event("provider.request.finished", call)


def now():
    return datetime.now(timezone.utc).isoformat()


def module_for(case):
    return importlib.import_module("benchmarks.agent_design." + MODULES[case["category"]])


def coverage(case):
    if case["category"] == "a2a":
        return {"status": "unsupported", "reason": "尚无实际A2A协议端点、传输和能力profile；本地编排不能替代本协议用例。"}
    module = module_for(case)
    if case["id"] not in module.SUPPORTED_IDS:
        return {"status": "unsupported", "reason": getattr(module, "UNSUPPORTED", {}).get(case["id"], "执行适配器未实现")}
    mode = "deterministic_runtime" if case["id"] in getattr(module, "DETERMINISTIC_IDS", set()) else "live_model"
    return {"status": "planned", "mode": mode}


class CaseAgent:
    def __init__(self, *, case, harness_path):
        self.case = case
        self.harness_path = harness_path
        self.recorder = None

    def run(self, prompt):
        if self.recorder is None:
            raise RuntimeError("Design scenarios require the trace instrumentation hook")
        from adapters.deepseek_public_eval import build_agent as build_provider
        import sys
        root = Path(self.harness_path).resolve(strict=True)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        native = importlib.import_module("agent")
        if Path(native.__file__).resolve().parent != root / "agent":
            raise ValueError("A different agent package was imported")
        mode = coverage(self.case)["mode"]
        provider = None
        llm = None
        if mode == "live_model":
            provider = build_provider(harness_path=self.harness_path, max_steps=24,
                max_tokens=self.case["setup"]["limits"]["max_total_tokens"], max_output_tokens=1024, timeout_seconds=40)
            llm = AccountedLLM(provider.agent.llm, provider.client, self.recorder)
        scenario = module_for(self.case).build_case(self.case, native, llm, self.recorder)
        execution_error = None
        started = time.monotonic()
        try:
            with self.recorder.span("design.scenario", "scenario", input={"prompt": scenario.prompt},
                    metadata={"case_id": self.case["id"], "mode": mode, "limits": self.case["setup"]["limits"],
                              "limitations": scenario.limitations}) as span:
                try:
                    raw = trace_agent(scenario.agent, self.recorder).run(scenario.prompt, **getattr(scenario, "run_kwargs", {}))
                    outcome = adapt_traced(raw)
                except Exception as exc:
                    execution_error = {"stage": "agent", "type": type(exc).__name__, "message": str(sanitize_trace_value(str(exc)))}
                    outcome = AgentOutcome("", False, "agent_exception", 0, 0)
                calls = provider.client.calls if provider else []
                provider_error = provider.client.failure if provider else None
                budget_stops = {"call_budget", "token_budget", "length"}
                if provider_error:
                    if provider_error["type"] in budget_stops:
                        execution_error = None
                        outcome = replace(outcome, success=False, stop_reason=provider_error["type"])
                    else:
                        execution_error = dict(provider_error)
                runtime_tokens = outcome.tokens
                known_tokens = sum(call["usage"].get("total_tokens") or 0 for call in calls)
                usage_complete = all(call["usage_complete"] for call in calls)
                if provider:
                    outcome = replace(outcome, tokens=known_tokens)
                try:
                    scores = scenario.evaluate(outcome)
                except Exception as exc:
                    execution_error = {"stage": "oracle", "type": type(exc).__name__, "message": str(sanitize_trace_value(str(exc)))}
                    scores = {}
                observed_tokens = known_tokens if provider else runtime_tokens
                token_limit = self.case["setup"]["limits"]["max_total_tokens"]
                scores.update(observed_total_tokens=observed_tokens,
                              observed_token_limit_pass=observed_tokens <= token_limit if usage_complete else None,
                              observed_token_overshoot=max(0, observed_tokens - token_limit))
                if observed_tokens > token_limit:
                    scores["contract_pass"] = False
                # Metrics must follow the current external failure, not a lucky
                # state left by a partially completed provider call.
                if execution_error:
                    scores.update(contract_pass=None, task_success=None)
                scores.setdefault("fault_triggered", None)
                scores.setdefault("task_success", None)
                scores.setdefault("contract_pass", None)
                if self.case["fault"] is not None and scores["fault_triggered"] is False:
                    scores["contract_pass"] = None
                evidence = scenario.evidence()
                metadata = {**outcome.metadata, "case_id": self.case["id"], "category": self.case["category"],
                    "execution_mode": mode, "model": "deepseek-v4-flash" if mode == "live_model" else None,
                    "thinking": "disabled" if mode == "live_model" else None, "design_scores": scores,
                    "case_evidence": evidence, "limitations": scenario.limitations, "api_calls": calls,
                    "usage_complete": usage_complete, "provider_error": provider_error,
                    "execution_error": execution_error, "runtime_reported_tokens": runtime_tokens,
                    "controlled_usage_is_not_billed": mode == "deterministic_runtime",
                    "scenario_elapsed_seconds": time.monotonic() - started,
                    "case_deadline_scope": "scenario-specific controlled/environment clock; separate hard process protection=120s",
                    "api_limits": {"max_calls": 24, "max_output_tokens_per_call": 1024, "timeout_seconds": 40,
                                   "observed_total_token_guard": self.case["setup"]["limits"]["max_total_tokens"],
                                   "strict_billing_cap": False}}
                outcome = replace(outcome, tokens=known_tokens, metadata=sanitize_trace_value(metadata))
                self.recorder.event("design.oracle", {"scores": scores, "evidence": evidence})
                span.finish(output={"scores": scores, "answer": outcome.answer, "stop_reason": outcome.stop_reason},
                            status="ok" if scores["contract_pass"] is True else "error" if scores["contract_pass"] is False else "unassessed")
                return outcome
        finally:
            close = getattr(scenario, "close", None)
            if close:
                close()


def build_agent(**kwargs):
    return CaseAgent(**kwargs)


def instrument(agent, recorder):
    agent.recorder = recorder
    return agent


def adapt(result):
    return result


class DesignScorer:
    metric_names = ("contract_pass", "task_success", "fault_triggered")

    def score(self, task, outcome):
        return outcome.metadata["design_scores"]


def prepare(root, harness_path, *, condition=CONDITION):
    if not isinstance(condition, str) or not condition.strip():
        raise ValueError("condition must be a nonempty label")
    root = Path(root).resolve()
    catalog = load_catalog()
    root.mkdir(parents=True, exist_ok=False)
    entries = []
    for case in catalog["cases"]:
        entry = {"task_id": case["id"], "category": case["category"], "title": case["title"], **coverage(case)}
        if entry["status"] == "planned":
            relative = "cases/" + case["id"]
            config = {"trials": 1, "seed": 20260906, "timeout_seconds": 120,
                "conditions": [{"id": condition, "agent": "benchmarks.agent_design.live:build_agent",
                    "outcome_adapter": "benchmarks.agent_design.live:adapt",
                    "config": {"case": case, "harness_path": str(Path(harness_path).resolve())}}],
                "trace": {"enabled": True, "instrumentation": "benchmarks.agent_design.live:instrument"},
                "scorers": [{"factory": "benchmarks.agent_design.live:DesignScorer", "metric_names": list(DesignScorer.metric_names)}],
                "source_roots": [str(Path(harness_path).resolve())],
                "metadata": {"catalog_fingerprint": fingerprint(catalog), "execution_mode": entry["mode"],
                             "purpose": "one regression trial; not a causal architecture comparison"}}
            create_experiment(root / relative, [{"id": case["id"], "prompt": case["prompt"], "group_id": case["id"]}], config)
            entry["experiment"] = relative
        entries.append(entry)
    manifest = {"schema_version": 1, "created_at": now(), "started_at": None, "finished_at": None,
        "model": "deepseek-v4-flash", "thinking": "disabled", "condition": condition, "trials": 1,
        "catalog_recommended_trials": 5, "planned_cases": 50, "catalog_fingerprint": fingerprint(catalog),
        "catalog": catalog, "entries": entries, "is_architecture_ablation": False, "sessions": []}
    write_json_report(root / "run_manifest.json", manifest)
    return manifest


def summarize(root):
    root = Path(root)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    records = []
    summary_rows = []
    trace_dir = root / "traces"
    trace_dir.mkdir(exist_ok=True)
    for entry in manifest["entries"]:
        row = dict(entry)
        if entry["status"] == "unsupported":
            summary_rows.append(row)
            continue
        saved = read_records(root / entry["experiment"])
        if not saved:
            row["status"] = "pending"
        else:
            record = saved[0]
            records.append(record)
            meta = record["outcome"]["metadata"]
            scores = record["scores"]
            row.update(scores=scores, answer=record["outcome"]["answer"], stop_reason=record["outcome"]["stop_reason"],
                       tokens=record["outcome"]["tokens"], trace=record.get("trace"),
                       limitations=meta.get("limitations", []), execution_error=record["execution_error"],
                       provider_error=meta.get("provider_error"), evidence=meta.get("case_evidence", {}))
            if record["execution_error"]:
                row["status"] = "execution_error"
            elif record.get("scoring_error") or record.get("scorer_errors"):
                row["status"] = "scoring_error"
                row["scorer_errors"] = record.get("scorer_errors")
            elif scores.get("contract_pass") is True:
                row["status"] = "passed"
            elif scores.get("contract_pass") is False:
                row["status"] = "failed"
            else:
                row["status"] = "unassessed"
            reference = record.get("trace", {})
            if reference:
                original = Path(reference["path"])
                for source in (original, original.with_suffix(".manifest.json")):
                    if source.exists():
                        (trace_dir / source.name).write_bytes(source.read_bytes())
        summary_rows.append(row)
    calls = []
    unknown_accounting_runs = []
    for record in records:
        meta = record["outcome"]["metadata"]
        if "api_calls" in meta:
            calls.extend(meta["api_calls"])
            continue
        entry = next(row for row in manifest["entries"] if row["task_id"] == record["task_id"])
        if entry.get("mode") != "live_model":
            continue
        unknown_accounting_runs.append(record["task_id"])
        # Reconstruct only durable provider events, never double-count the
        # enclosing LLM/summary spans or pretend an unfinished request was free.
        recovered = {}
        reference = record.get("trace") or {}
        if reference.get("path") and Path(reference["path"]).exists():
            for event in read_trace(reference["path"])["events"]:
                data = event.get("data", {})
                if event.get("name") == "provider.request.started":
                    recovered[data["call_index"]] = {"call_index": data["call_index"], "usage": {}, "usage_complete": False}
                elif event.get("name") == "provider.request.finished":
                    recovered[data["call_index"]] = data
        calls.extend(recovered.values())
    usage = {key: sum(call["usage"].get(key) or 0 for call in calls) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    report = {"schema_version": 1, "model": manifest["model"], "condition": manifest.get("condition", CONDITION), "is_architecture_ablation": False,
        "trials": 1, "planned": 50, "counts": dict(Counter(row["status"] for row in summary_rows)),
        "by_category": {category: dict(Counter(row["status"] for row in summary_rows if row["category"] == category)) for category in CATEGORY_LABELS},
        "by_mode": {mode: dict(Counter(row["status"] for row in summary_rows if row.get("mode") == mode)) for mode in ("live_model", "deterministic_runtime")},
        "real_api_attempts": len(calls), "requests_with_usage": sum(call["usage_complete"] for call in calls),
        "known_usage": usage, "usage_complete": not unknown_accounting_runs and all(call["usage_complete"] for call in calls),
        "unknown_accounting_runs": unknown_accounting_runs,
        "actual_response_models": dict(Counter(call.get("model") for call in calls if call.get("model"))),
        "cases": summary_rows, "records": records, "reported_at": now(),
        "interpretation": "Passed means this specific mechanism contract passed. Honest failure can pass a guard contract. Unsupported and untriggered cases are not passes. This is one regression trial."}
    write_json_report(root / "report.json", report)
    (root / "REPORT.md").write_text(render(report), encoding="utf-8")
    if list(trace_dir.glob("*.jsonl")):
        export_trace_html(root, root / "index.html")
    return report


def render(report):
    lines = ["# Agent 设计用例实际运行", "", f"每条已支持用例执行 Agent（{report.get('condition', CONDITION)}）1 次，未进行架构消融。精确时序/用量场景使用固定响应驱动真实运行时，其余使用 DeepSeek V4 Flash 非思考模式。", "",
        f"计划 50 条；状态：`{json.dumps(report['counts'], ensure_ascii=False)}`。", "",
        "| 方向 | 通过 | 失败 | 未完整评定 | 不支持 | 执行错误 | 评分错误 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for category, label in CATEGORY_LABELS.items():
        counts = report["by_category"][category]
        lines.append("| " + label + " | " + " | ".join(str(counts.get(key, 0)) for key in ("passed", "failed", "unassessed", "unsupported", "execution_error", "scoring_error")) + " |")
    lines += ["", "真实模型和确定性机制测试分开统计：", "", "```json", json.dumps(report["by_mode"], ensure_ascii=False, indent=2), "```", "",
        f"记录到真实 API 尝试 {report['real_api_attempts']} 次；收到用量 {report['requests_with_usage']} 次；已知 token {report['known_usage']['total_tokens']:,}（输入 {report['known_usage']['prompt_tokens']:,}，输出 {report['known_usage']['completion_tokens']:,}）。用量完整：{report['usage_complete']}。未知请求用量不按零估计。确定性模型的虚拟 token 不计费、不加入该总数。", "",
        "| 用例 | 模式 | 状态 | 结果/原因 |", "|---|---|---|---|"]
    for row in report["cases"]:
        detail = row.get("reason") or row.get("answer") or str(row.get("execution_error") or row.get("scores", {}))
        detail = detail.replace("|", "&#124;").replace("\n", " ")[:260]
        lines.append(f"| {row['task_id']} {row['title']} | {row.get('mode', '—')} | {row['status']} | {detail} |")
    lines += ["", "[查看逐步轨迹](index.html)。完整状态判定、限制、故障触发记录、用量与停止原因见 report.json。", "",
        "说明：工具/状态沙箱是受控环境，其服务端幂等不等于Agent自动具备幂等。部分用例增加了JSON输出契约或显式虚拟等待工具，逐题 limitations/evidence 保留这些差异。任务没有触发目标故障、尚无完整oracle的用例标为 unassessed，不能称通过。协议A2A尚未实现。", ""]
    return "\n".join(lines)


def run(root, harness_path, *, workers=4, preflight=False):
    from benchmarks.public_data.run_deepseek_dev import load_deepseek_credential
    from agent_eval.experiments import _lock
    root = Path(root).resolve()
    with _lock(root):
        manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
        load_deepseek_credential(harness_path)
        os.environ["AGENT_LOG_PER_RUN"] = "false"
        manifest["started_at"] = manifest["started_at"] or now()
        session = {"started_at": now(), "preflight": preflight, "workers": workers, "errors": []}
        manifest["sessions"].append(session)
        write_json_report(root / "run_manifest.json", manifest)
        entries = [entry for entry in manifest["entries"] if entry["status"] == "planned"]
        if preflight:
            entries = [entry for entry in entries if entry["task_id"] == "TOOL-01"]
        paused = threading.Event()
        lock = threading.Lock()
        failures = 0

        def progress(record):
            nonlocal failures
            with lock:
                error = record.get("execution_error")
                if error:
                    failures += 1
                    if any(code in json.dumps(error) for code in ("401", "402", "403", "429")) or failures >= 4:
                        paused.set()
                print(json.dumps({"case": record["task_id"], "scores": record["scores"], "tokens": record["outcome"]["tokens"],
                                  "execution_error": error, "paused": paused.is_set()}, ensure_ascii=False), flush=True)

        def execute_entry(entry):
            if paused.is_set():
                return
            try:
                run_experiment(root / entry["experiment"], progress=progress)
            except Exception as exc:
                with lock:
                    session["errors"].append({"case": entry["task_id"], "type": type(exc).__name__, "message": str(sanitize_trace_value(str(exc)))})
                paused.set()

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(execute_entry, entry) for entry in entries]
                for future in as_completed(futures):
                    future.result()
        finally:
            session["finished_at"] = now()
            report = summarize(root)
            if report["counts"].get("pending", 0) == 0:
                manifest["finished_at"] = now()
            write_json_report(root / "run_manifest.json", manifest)
        print(json.dumps({"counts": report["counts"], "known_usage": report["known_usage"], "session_errors": session["errors"]}, ensure_ascii=False), flush=True)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--harness-path", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--condition", default=CONDITION, help="label for a newly prepared run; resume uses its frozen manifest")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        parser.error("workers must be 1..8")
    root = Path(args.out)
    if args.summarize:
        summarize(root)
        return 0
    if not args.resume:
        manifest = prepare(root, args.harness_path, condition=args.condition)
        print(json.dumps(Counter(e["status"] for e in manifest["entries"])), flush=True)
    if args.prepare_only:
        return 0
    report = run(root, args.harness_path, workers=args.workers, preflight=args.preflight)
    return 2 if report["counts"].get("pending", 0) and not args.preflight else 0


if __name__ == "__main__":
    raise SystemExit(main())
