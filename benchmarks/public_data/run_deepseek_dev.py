"""Run the frozen public development splits through the companion ReActAgent.

Each shard uses the ordinary durable experiment runner. Shards let independent
tasks run concurrently without sharing agents, SQLite writers, or credentials
on disk. A preflight is part of the experiment and is skipped on resume.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import random
import threading
import time

from agent_eval.experiments import create_experiment, fingerprint, run_experiment
from agent_eval.dataset_scoring import load_public_tasks
from agent_eval.score_reporting import write_json_report


def _now():
    return datetime.now(timezone.utc).isoformat()


def load_deepseek_credential(harness_path):
    """Use the existing process key, or parse only its line in the sibling .env."""
    if os.environ.get("DEEPSEEK_API_KEY"):
        return
    path = Path(harness_path) / ".env"
    if path.is_file():
        from dotenv import dotenv_values
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                key, sep, _ = line.strip().removeprefix("export ").partition("=")
                if sep and key.strip() == "DEEPSEEK_API_KEY":
                    value = dotenv_values(stream=io.StringIO(line), interpolate=False).get("DEEPSEEK_API_KEY")
                    if value:
                        os.environ["DEEPSEEK_API_KEY"] = value
                        return
    raise RuntimeError("DEEPSEEK_API_KEY is unavailable in the environment or companion .env")


def make_config(harness_path):
    return {
        "trials": 1, "seed": 20260906, "timeout_seconds": 270, "score_timeout_seconds": 30,
        "source_roots": [str(Path(harness_path).resolve())],
        "conditions": [{"id": "deepseek-v4-flash", "agent": "adapters.deepseek_public_eval:build_agent",
                        "outcome_adapter": "adapters.deepseek_public_eval:adapt", "config": {
                            "harness_path": str(Path(harness_path).resolve()), "model": "deepseek-v4-flash",
                            "temperature": 0, "max_output_tokens": 2048, "max_steps": 4,
                            "max_tokens": 20000, "timeout_seconds": 60}}],
        "scorers": [{"factory": "agent_eval.scoring:RuleScorer"},
                    {"factory": "agent_eval.scoring:TrajectoryScorer"},
                    {"factory": "agent_eval.dataset_scoring:PublicDatasetScorer"}],
        "metadata": {"provider": "DeepSeek official API", "model": "deepseek-v4-flash",
                     "thinking": "disabled", "dataset_split": "dev", "toolset": "bounded_calculator_only",
                     "judge": "none; deterministic public benchmark scoring", "api_retries": 0,
                     "purpose": "Single-condition development baseline, not a version comparison"},
    }


def prepare_run(root, harness_path, *, shards_per_dataset=4):
    root = Path(root)
    if (root / "run_manifest.json").exists():
        raise ValueError("Run already exists; use --resume or --preflight to run pending work")
    if root.exists():
        raise ValueError("Use a new output directory")
    if not 1 <= shards_per_dataset <= 8:
        raise ValueError("shards_per_dataset must be in 1..8")
    config = make_config(harness_path)
    # Freeze all planned identities before any request can be sent.
    selected = {}
    for dataset in ("gsm8k", "hotpotqa"):
        tasks = load_public_tasks(Path(__file__).parent / dataset / "dev.jsonl")
        if len(tasks) != 400 or any(t["public_case"]["split"] != "dev" for t in tasks):
            raise ValueError("Expected the frozen 400-case development split")
        selected[dataset] = tasks
    root.mkdir(parents=True)
    datasets = []
    for dataset, tasks in selected.items():
        shuffled = list(tasks)
        random.Random(20260906).shuffle(shuffled)
        paths = []
        for index in range(shards_per_dataset):
            relative = f"{dataset}/shard-{index:02d}"
            paths.append(relative)
            create_experiment(root / relative, shuffled[index::shards_per_dataset], config)
        datasets.append({"dataset": dataset, "expected_ids": [t["id"] for t in tasks],
                         "dataset_fingerprint": fingerprint(tasks), "shards": paths})
    manifest = {"schema_version": 1, "scoring_version": 2, "model": "deepseek-v4-flash",
                "condition": "deepseek-v4-flash", "split": "dev", "trials": 1,
                "thinking": "disabled", "started_at": None, "finished_at": None,
                "created_at": _now(), "datasets": datasets, "config": config,
                "source_manifest": json.loads((Path(__file__).parent / "manifest.json").read_text(encoding="utf-8")),
                "sessions": []}
    write_json_report(root / "run_manifest.json", manifest)
    return manifest


class _PauseRun(RuntimeError):
    pass


def run_shards(root, harness_path, *, workers=8, preflight=False):
    from agent_eval.experiments import _lock
    root = Path(root)
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in 1..8")
    load_deepseek_credential(harness_path)
    with _lock(root):
        path = root / "run_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["model"] != "deepseek-v4-flash" or manifest["split"] != "dev":
            raise ValueError("Unexpected model or dataset split")
        paths = ([d["shards"][0] for d in manifest["datasets"]] if preflight else
                 [p for d in manifest["datasets"] for p in d["shards"]])
        manifest["started_at"] = manifest["started_at"] or _now()
        manifest["finished_at"] = None
        session = {"started_at": _now(), "ended_at": None, "workers": min(workers, len(paths)),
                   "preflight": preflight, "completed_this_session": 0, "errors": []}
        manifest["sessions"].append(session)
        write_json_report(path, manifest)
        state_lock = threading.Lock()
        paused = threading.Event()
        token_sum = 0
        last_log = time.monotonic()
        failed = 0

        def progress(record):
            nonlocal token_sum, failed, last_log
            with state_lock:
                session["completed_this_session"] += 1
                token_sum += record["outcome"]["tokens"]
                if record["execution_error"]:
                    failed += 1
                    message = json.dumps(record["execution_error"], ensure_ascii=False).lower()
                    # Do not send hundreds of requests with a rejected key, exhausted
                    # balance, or a sustained service failure.
                    if any(code in message for code in ("401", "402", "403", "429")) or failed >= 8:
                        paused.set()
                now = time.monotonic()
                if preflight or session["completed_this_session"] % 20 == 0 or now - last_log > 20:
                    print(json.dumps({"completed_this_session": session["completed_this_session"],
                                      "task": record["task_id"], "known_tokens": token_sum,
                                      "execution_errors": failed, "paused": paused.is_set()}, ensure_ascii=False), flush=True)
                    last_log = now
                if paused.is_set():
                    raise _PauseRun("Paused after provider rejection or repeated failures; committed outcomes retained")

        def run_one(relative):
            if paused.is_set():
                return None
            try:
                return run_experiment(root / relative, max_runs=1 if preflight else None, progress=progress)
            except Exception as exc:
                paused.set()
                with state_lock:
                    session["errors"].append({"shard": relative, "type": type(exc).__name__, "message": str(exc)})
                return None

        try:
            with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as pool:
                futures = [pool.submit(run_one, relative) for relative in paths]
                for future in as_completed(futures):
                    future.result()
        finally:
            session["ended_at"] = _now()
            from .summarize_run import summarize_run
            # Persist session timing before deriving an aggregate snapshot.
            write_json_report(path, manifest)
            summary = summarize_run(root)
            if summary.get("status") == "completed":
                manifest["finished_at"] = _now()
                write_json_report(path, manifest)
                summary = summarize_run(root)
            from .summarize_run import render_markdown
            write_json_report(root / "summary.json", summary)
            (root / "report.md").write_text(render_markdown(summary), encoding="utf-8")
        print(json.dumps({"session": session, "summary": str(root / "summary.json")}, ensure_ascii=False), flush=True)
        return 2 if session["errors"] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--harness-path", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shards-per-dataset", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not args.resume:
        prepare_run(args.out, args.harness_path, shards_per_dataset=args.shards_per_dataset)
    if args.plan_only:
        print("Planned 800 development tasks; no API requests sent.")
        return 0
    return run_shards(args.out, args.harness_path, workers=args.workers, preflight=args.preflight)


if __name__ == "__main__":
    raise SystemExit(main())
