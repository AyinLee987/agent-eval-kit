"""Robustness benchmark: does rewording a question change the agent's answer?

For each base question in queries.py, three variants are asked too — a
same-language paraphrase, a register/politeness-noise shift, and an English
translation — through a real ReActAgent (Bailian qwen-plus). The four
resulting answers per question are embedded (a real embedding endpoint,
same requirement as the RAG recall benchmark) and compared pairwise by
cosine similarity: higher average similarity means the agent's answer held
steady regardless of how it was asked.

RuleScorer/ToolUsageScorer run alongside as a secondary, per-phrasing
sanity check — similarity alone can't tell four confidently-wrong-in-the-
same-way answers from four confidently-right ones.

Usage (from the evaluation/ repo root):

    python benchmarks/robustness/run_benchmark.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

EVAL_ROOT = Path(__file__).resolve().parents[2]
HARNESS_REPO = Path(os.environ.get("AGENT_HARNESS_PATH", EVAL_ROOT.parent / "agent" / "agent-harness-from-scratch"))
BENCHMARK_DIR = Path(__file__).resolve().parent

for path in (str(EVAL_ROOT), str(HARNESS_REPO), str(BENCHMARK_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dotenv import load_dotenv  # noqa: E402

if not load_dotenv(dotenv_path=HARNESS_REPO / ".env"):
    raise SystemExit(f"Could not find a .env file at {HARNESS_REPO / '.env'}.")

from agent import BailianLLM, OpenAICompatibleEmbeddingProvider, ToolRegistry  # noqa: E402

from adapters.react_agent_adapter import adapt, build_agent_factory  # noqa: E402
from agent_eval.harness import EvalHarness  # noqa: E402
from agent_eval.scoring import RuleScorer, ToolUsageScorer  # noqa: E402
from agent_eval.similarity import average_pairwise_similarity  # noqa: E402

from cached_embeddings import CachedEmbeddingProvider  # noqa: E402
from queries import CASES  # noqa: E402
from tools import calculator, current_datetime, lookup_fact  # noqa: E402

RESULTS_PATH = BENCHMARK_DIR / "results.json"
CACHE_PATH = BENCHMARK_DIR / ".embedding_cache.json"
PHRASING_ORDER = ("base", "paraphrase", "register", "translation")

AGENT_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a calculator, a lookup_fact "
    "tool for known company facts, and a current_datetime tool. Use a tool "
    "only when the task actually needs it; answer directly otherwise. "
    "Respond in whatever language the user's message is written in."
)


def require_env(name: str) -> None:
    if not os.environ.get(name):
        raise SystemExit(
            f"Refusing to run: {name} is not set (checked {HARNESS_REPO / '.env'})."
        )


def require_real_embedding_config() -> Dict[str, str]:
    values = {
        "model": os.environ.get("RAG_EMBEDDING_MODEL"),
        "api_key": os.environ.get("RAG_EMBEDDING_API_KEY"),
        "base_url": os.environ.get("RAG_EMBEDDING_BASE_URL"),
    }
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise SystemExit(
            "Refusing to run: RAG_EMBEDDING_* is not fully configured "
            f"(missing: {', '.join(missing)}). This benchmark needs a real "
            "embedding endpoint to measure answer similarity meaningfully."
        )
    return values  # type: ignore[return-value]


def flatten_tasks() -> List[Dict[str, Any]]:
    """One task per (case, phrasing) — each phrasing runs as an independent,
    single-turn task; EvalHarness already builds a fresh agent per task."""

    tasks: List[Dict[str, Any]] = []
    for case in CASES:
        for phrasing in PHRASING_ORDER:
            tasks.append({
                "id": f"{case['id']}::{phrasing}",
                "prompt": case["phrasings"][phrasing],
                "expect_tool": case["expect_tool"],
                "expect_substrings": case["expect_substrings"],
            })
    return tasks


def main() -> None:
    require_env("BAILIAN_API_KEY")
    embed_config = require_real_embedding_config()

    build_agent = build_agent_factory(
        llm_factory=BailianLLM,
        tools_factory=lambda: ToolRegistry([calculator, lookup_fact, current_datetime]),
        system_prompt=AGENT_SYSTEM_PROMPT,
        max_steps=6,
        agent_name="robustness-agent",
    )

    harness = EvalHarness(
        build_agent=build_agent,
        outcome_adapter=adapt,
        tasks=flatten_tasks(),
        scorers=[RuleScorer(), ToolUsageScorer()],
    )

    print("Agent model: Bailian (qwen-plus)")
    print(f"Embedding model: {embed_config['model']}\n")

    scorecard = harness.run_all()
    print(scorecard.render())

    real_embeddings = OpenAICompatibleEmbeddingProvider(
        model=embed_config["model"],
        api_key=embed_config["api_key"],
        base_url=embed_config["base_url"],
        provider_name="robustness-benchmark",
    )
    embeddings = CachedEmbeddingProvider(real_embeddings, str(CACHE_PATH))

    by_task_id = {r.task_id: r for r in scorecard.results}
    case_reports: List[Dict[str, Any]] = []
    for case in CASES:
        phrasing_rows: Dict[str, Any] = {}
        vectors = []
        for phrasing in PHRASING_ORDER:
            result = by_task_id[f"{case['id']}::{phrasing}"]
            answer = result.outcome.answer
            vectors.append(embeddings.embed_query(answer))
            phrasing_rows[phrasing] = {
                "prompt": case["phrasings"][phrasing],
                "answer": answer,
                "rule_pass": result.scores.get("rule_pass"),
                "used_expected_tool": result.scores.get("used_expected_tool"),
            }
        case_reports.append({
            "id": case["id"],
            "avg_pairwise_similarity": average_pairwise_similarity(vectors),
            "phrasings": phrasing_rows,
        })

    overall_avg_similarity = sum(c["avg_pairwise_similarity"] for c in case_reports) / len(case_reports)

    print("\n" + "=" * 58)
    print("ROBUSTNESS: avg pairwise answer similarity per case")
    print("=" * 58)
    for c in case_reports:
        print(f"{c['id']:<24}{c['avg_pairwise_similarity']:.3f}")
    print("-" * 58)
    print(f"{'overall avg':<24}{overall_avg_similarity:.3f}")
    print("=" * 58)

    payload = {
        "aggregate": scorecard.aggregate(),
        "overall_avg_similarity": overall_avg_similarity,
        "cases": case_reports,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\nFull results (every phrasing's answer + similarity) written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
