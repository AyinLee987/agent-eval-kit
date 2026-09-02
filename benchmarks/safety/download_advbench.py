"""Fetches AdvBench's harmful_behaviors.csv (Zou et al. 2023, llm-attacks)
and writes a deterministic sample to prompts.json — the same "fetch once,
check in the JSON" pattern as rag_recall_beir/download_nfcorpus.py, so the
benchmark itself never depends on network access.

AdvBench ships 520 single-turn harmful-behavior requests ("goal") plus a
scripted compliance opener ("target") we don't use — this benchmark only
needs the goal text as the prompt. We sample a subset (default 50) rather
than running the full 520 for cost control (each prompt is one real
Bailian call plus one real DeepSeek classifier call).

Usage (from the evaluation/ repo root):

    python benchmarks/safety/download_advbench.py [--n 50] [--seed 0]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path

SOURCE_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
OUT_PATH = Path(__file__).resolve().parent / "prompts.json"


def fetch_csv(url: str, *, retries: int = 3, timeout: float = 20.0) -> str:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as exc:  # pragma: no cover - network flake
            last_exc = exc
    raise SystemExit(f"Failed to fetch {url} after {retries} attempts: {last_exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Sample size (default 50 of 520).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw = fetch_csv(SOURCE_URL)
    reader = csv.DictReader(io.StringIO(raw))
    goals = [row["goal"].strip() for row in reader if row.get("goal")]
    print(f"Fetched {len(goals)} AdvBench prompts from {SOURCE_URL}")

    rng = random.Random(args.seed)
    sample = rng.sample(goals, min(args.n, len(goals)))
    sample.sort()  # deterministic file content regardless of sample order

    records = [{"id": f"advbench-{i:03d}", "prompt": text} for i, text in enumerate(sample)]
    OUT_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(records)} sampled prompts to {OUT_PATH}")


if __name__ == "__main__":
    main()
