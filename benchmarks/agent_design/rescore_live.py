"""Derive revised design scores from saved answers and environment evidence.

No agent is constructed, tools executed, or model endpoint contacted. The first
run and its embedded oracle events remain immutable. The new report states
which scores changed and why this is a post-hoc measurement correction.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import shutil

from agent_eval.experiments import rescore_experiment
from agent_eval.score_reporting import write_json_report
from agent_eval.trace_viewer import export_trace_html
from . import live


class EvidenceScorer:
    metric_names = ("contract_pass", "task_success", "fault_triggered")

    def __init__(self, case):
        self.case = case

    def score(self, task, outcome):
        meta = outcome.metadata
        if self.case["category"] == "budget":
            scores = dict(meta["design_scores"])
        else:
            scores = live.module_for(self.case).evaluate_saved(self.case, outcome, meta["case_evidence"])
        # Summary and worker model calls belong to the same root budget. The
        # controlled test's stimuli remain separate from the provider bill.
        observed = (meta["runtime_reported_tokens"] if meta["execution_mode"] == "deterministic_runtime" else outcome.tokens)
        limit = self.case["setup"]["limits"]["max_total_tokens"]
        scores.update(observed_total_tokens=observed,
                      observed_token_limit_pass=observed <= limit if meta["usage_complete"] else None,
                      observed_token_overshoot=max(0, observed - limit))
        if observed > limit:
            scores["contract_pass"] = False
        if self.case["fault"] is not None and scores.get("fault_triggered") is False:
            scores["contract_pass"] = None
        return scores


def derive(source, destination, workers=4):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    original = json.loads((source / "report.json").read_text(encoding="utf-8"))
    if original["counts"].get("pending"):
        raise ValueError("Finish the original run before reviewing its scores")
    destination.mkdir(parents=True, exist_ok=False)
    catalog = {case["id"]: case for case in manifest["catalog"]["cases"]}
    entries = [entry for entry in manifest["entries"] if entry.get("experiment")]

    def score_entry(entry):
        return rescore_experiment(source / entry["experiment"], destination / "cases" / (entry["task_id"] + ".json"),
            scorers=[{"factory": "benchmarks.agent_design.rescore_live:EvidenceScorer",
                      "config": {"case": catalog[entry["task_id"]]}}])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        payloads = list(pool.map(score_entry, entries))
    records = [record for payload in payloads for record in payload["records"]]
    record_map = {record["task_id"]: record for record in records}
    report = deepcopy(original)
    changes = []
    for row in report["cases"]:
        if row["status"] == "unsupported":
            continue
        record = record_map[row["task_id"]]
        old_scores = dict(row["scores"])
        row["original_scores"] = old_scores
        row["scores"] = record["scores"]
        if record.get("execution_error"):
            row["status"] = "execution_error"
        elif record.get("scorer_errors") or record.get("scoring_error"):
            row["status"] = "scoring_error"
        else:
            value = row["scores"].get("contract_pass")
            row["status"] = "passed" if value is True else "failed" if value is False else "unassessed"
        delta = {key: {"original": value, "revised": row["scores"].get(key)} for key, value in old_scores.items()
                 if key in row["scores"] and value != row["scores"][key]}
        if delta:
            changes.append({"task_id": row["task_id"], "metrics": delta})
    report.update(records=records, counts=dict(Counter(row["status"] for row in report["cases"])),
                  original_counts=original["counts"], reported_at=live.now())
    report["by_category"] = {category: dict(Counter(row["status"] for row in report["cases"] if row["category"] == category))
                             for category in live.CATEGORY_LABELS}
    report["by_mode"] = {mode: dict(Counter(row["status"] for row in report["cases"] if row.get("mode") == mode))
                         for mode in ("live_model", "deterministic_runtime")}
    report["score_review"] = {"version": 2, "source_report": str(source / "report.json"),
        "additional_model_calls": 0, "changes": changes,
        "reason": "Post-hoc oracle fixes: accept one complete unambiguous JSON object with explanatory text; correct fixture issue matching and explicit state aliases; recognize preserved read-only constraint. No change to observed actions, side effects, fault coverage, or token limits.",
        "scope": "A measurement correction on the same baseline trajectories; not an independent trial or a demonstrated architecture improvement."}
    write_json_report(destination / "report.json", report)
    explanation = ("本报告对同一批已保存的回答、工具观察和最终状态重新评分，新增模型调用 **0** 次。"
                   "修正了说明文字包围JSON导致的假失败、TOOL-10检查错误问题类型、状态字段合法别名，以及只读约束同义表述。"
                   "原始评分和轨迹内的旧oracle事件保留，逐项变更见 report.json 的 score_review。"
                   "这是评估器修正，不是Agent能力提升，也不是额外一次trial。\n\n")
    (destination / "REPORT.md").write_text(explanation + live.render(report), encoding="utf-8")
    shutil.copytree(source / "traces", destination / "traces")
    export_trace_html(destination, destination / "index.html")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = derive(args.source, args.out)
    print(json.dumps({"counts": report["counts"], "by_mode": report["by_mode"],
                      "changed_cases": [row["task_id"] for row in report["score_review"]["changes"]],
                      "additional_model_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
