"""CLI for legacy scorecards and durable, repeatable experiments."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from .execution import resolve as _resolve, build_scorers, default_scorers
from .harness import EvalHarness
from .score_reporting import write_json_report


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _view(path):
    from .experiments import _load, planned_keys, read_records
    path = Path(path)
    if path.is_dir():
        manifest, _ = _load(path)
        return {"records": read_records(path), "planned_keys": planned_keys(manifest),
                "measurement_protocol": manifest["measurement_protocol"],
                "dataset_fingerprint": manifest["dataset_fingerprint"]}
    return _read(path)


def _select(view, condition):
    records = view["records"]
    names = {r["condition"] for r in records}
    if condition is None:
        if len(names) != 1:
            raise ValueError("Specify a condition when a report has zero or multiple conditions")
        condition = next(iter(names))
    if names and condition not in names:
        raise ValueError(f"Unknown or entirely unfinished condition: {condition}")
    return [r for r in records if r["condition"] == condition]


def _write_or_print(value, out):
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_report(path, value)
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="agent-eval")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Legacy single-run scorecard (JSON or public JSONL)")
    for name in ("tasks", "agent", "outcome-adapter"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--dump")
    exp = sub.add_parser("experiment", help="Create a durable experiment from JSON config")
    for name in ("tasks", "config", "out"):
        exp.add_argument("--" + name, required=True)
    exp.add_argument("--plan-only", action="store_true")
    exp.add_argument("--max-runs", type=int)
    resume = sub.add_parser("resume", help="Resume pending trials or scoring")
    resume.add_argument("experiment")
    resume.add_argument("--max-runs", type=int)
    resume.add_argument("--recover-interrupted", action="store_true", help="Acknowledge uncertain external effects after a crash; live workers prevent recovery")
    show = sub.add_parser("report", help="Rebuild report from committed results")
    show.add_argument("experiment")
    rescore = sub.add_parser("rescore", help="Score saved outcomes without running agents")
    rescore.add_argument("experiment")
    rescore.add_argument("--out", required=True)
    rescore.add_argument("--scorers", help="JSON factory/config specs; custom LLM scorers can incur cost")
    rescore.add_argument("--timeout", type=float, default=120)
    compare = sub.add_parser("compare", help="Compare matched tasks/trials at group level")
    for name in ("a", "b", "metric"):
        compare.add_argument("--" + name, required=True)
    compare.add_argument("--a-condition")
    compare.add_argument("--b-condition")
    compare.add_argument("--allow-incomplete", action="store_true")
    compare.add_argument("--lower-is-better", action="store_true")
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--resamples", type=int, default=10000)
    compare.add_argument("--out")
    review = sub.add_parser("review-template", help="Export an unlabeled human review document")
    review.add_argument("experiment")
    review.add_argument("--dimensions", nargs="+")
    review.add_argument("--dimensions-file", help="JSON dimension specifications")
    review.add_argument("--out", required=True)
    calibration = sub.add_parser("calibrate", help="Evaluate judges against reviewed references")
    calibration.add_argument("--records", required=True)
    calibration.add_argument("--gold", required=True)
    calibration.add_argument("--out")
    concurrency = sub.add_parser("concurrency", help="Paired repeated serial/parallel benchmark")
    concurrency.add_argument("--case-factory", required=True)
    concurrency.add_argument("--workers", type=int, default=3)
    concurrency.add_argument("--repeats", type=int, default=3)
    concurrency.add_argument("--seed", type=int, default=0)
    concurrency.add_argument("--out")
    trace = sub.add_parser("trace-view", help="Export local JSONL traces to a portable HTML viewer")
    trace.add_argument("source", help="A trace JSONL, trace directory, or experiment directory")
    trace.add_argument("--out", required=True)
    catalog = sub.add_parser("design-cases", help="Validate and render the 50 Agent design case specifications")
    catalog.add_argument("--cases", help="Optional custom catalog JSON")
    catalog.add_argument("--out", help="Write the Chinese case catalog as Markdown")
    args = parser.parse_args(argv)
    try:
        from .experiments import create_experiment, export_report, load_tasks, rescore_experiment, run_experiment
        if args.command == "run":
            tasks = load_tasks(args.tasks)
            card = EvalHarness(_resolve(args.agent), _resolve(args.outcome_adapter), tasks, build_scorers(default_scorers(tasks))).run_all()
            print(card.render())
            if args.dump:
                card.dump(args.dump)
        elif args.command == "experiment":
            manifest = create_experiment(args.out, load_tasks(args.tasks), _read(args.config))
            print(f"Planned {len(manifest['schedule'])} runs in {args.out}")
            if not args.plan_only:
                report = run_experiment(args.out, max_runs=args.max_runs, progress=_progress)
                print(json.dumps({"states": report["states"], "conditions": report["conditions"]}, indent=2))
        elif args.command == "resume":
            report = run_experiment(args.experiment, recover_interrupted=args.recover_interrupted, max_runs=args.max_runs, progress=_progress)
            print(json.dumps({"states": report["states"], "conditions": report["conditions"]}, indent=2))
        elif args.command == "report":
            report = export_report(args.experiment)
            _write_or_print({k: v for k, v in report.items() if k != "records"}, None)
        elif args.command == "rescore":
            report = rescore_experiment(args.experiment, args.out, scorers=_read(args.scorers) if args.scorers else None, score_timeout_seconds=args.timeout)
            print(f"Rescored {len(report['records'])} saved outcomes into {args.out}")
        elif args.command == "compare":
            from .comparison import compare_experiments
            a, b = _view(args.a), _view(args.b)
            if not a.get("dataset_fingerprint") or a.get("dataset_fingerprint") != b.get("dataset_fingerprint"):
                raise ValueError("Comparison requires identical dataset fingerprints")
            if a.get("planned_keys") != b.get("planned_keys"):
                raise ValueError("Planned task/trial/group keys differ")
            if not a.get("measurement_protocol") or a["measurement_protocol"] != b.get("measurement_protocol"):
                raise ValueError("Scoring definitions or time budgets differ; rescore saved runs with one protocol or create matched experiments")
            result = compare_experiments(_select(a, args.a_condition), _select(b, args.b_condition),
                                         metric=args.metric, expected_keys=a.get("planned_keys"),
                                         allow_incomplete=args.allow_incomplete, seed=args.seed,
                                         n_resamples=args.resamples, higher_is_better=not args.lower_is_better)
            _write_or_print(result, args.out)
        elif args.command == "review-template":
            from .calibration import export_review_template
            from .experiments import _load, read_records
            manifest, tasks = _load(args.experiment)
            task_map = {t["id"]: t for t in tasks}
            records = read_records(args.experiment)
            for record in records:
                record["task"] = task_map[record["task_id"]]
            dimensions = _read(args.dimensions_file) if args.dimensions_file else args.dimensions
            result = export_review_template(records, dimensions=dimensions, dataset_fingerprint=manifest["dataset_fingerprint"])
            _write_or_print(result, args.out)
        elif args.command == "calibrate":
            from .calibration import evaluate_calibration
            result = evaluate_calibration(_view(args.records)["records"], _read(args.gold))
            _write_or_print(result, args.out)
        elif args.command == "concurrency":
            from .concurrency_bench import benchmark
            report = benchmark(max_workers=args.workers, repeats=args.repeats, seed=args.seed, case_factory=_resolve(args.case_factory))
            print(report.render())
            if args.out:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                report.dump(args.out)
        elif args.command == "trace-view":
            from .trace_viewer import export_trace_html
            print(export_trace_html(args.source, args.out))
        elif args.command == "design-cases":
            from benchmarks.agent_design.catalog import load_catalog, render_markdown
            catalog = load_catalog(args.cases) if args.cases else load_catalog()
            if args.out:
                path = Path(args.out)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(render_markdown(catalog), encoding="utf-8")
            print(f"Validated {len(catalog['cases'])} case specifications; this does not execute or score an Agent.")
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, ImportError, AttributeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        if getattr(exc, "diagnostics", None):
            print(json.dumps(exc.diagnostics, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


def _progress(record):
    state = "execution_error" if record["execution_error"] else "scorer_error" if record["scorer_errors"] else "scored"
    print(f"{record['condition']} / {record['task_id']} / trial {record['trial_id']}: {state}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
