"""Run controlled offline cases: python -m benchmarks.runtime_regression.run_benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path

from agent_eval.concurrency_bench import benchmark
from benchmarks.runtime_regression.cases import make_cases, make_healthy_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--healthy-only", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = benchmark(max_workers=args.workers, repeats=args.repeats, seed=args.seed,
                       case_factory=make_healthy_cases if args.healthy_only else make_cases)
    if args.out:
        report.dump(args.out)
    print("Controlled synthetic runtime fixture; no public-data or model-quality claim.")
    print(report.render())


if __name__ == "__main__":
    main()
