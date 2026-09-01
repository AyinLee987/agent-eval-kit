"""Adds a 5th pipeline arm to the BEIR NFCorpus ablation: hybrid RRF +
LLM-as-reranker (llm_reranker.py), instead of the default HeuristicReranker
or no reranker at all. Answers: does asking the chat model itself to score
candidate relevance beat the lexical-overlap heuristic, or plain RRF?

Reuses run_benchmark.py's corpus/embedding setup as-is (all cached, no new
embedding API calls) and re-evaluates all five pipelines against the same
323 queries for a directly comparable run -- but this one makes one real
LLM call per query for the llm_rerank arm (~323 calls total), on top of the
zero-cost cached-embedding pipelines. Requires DEEPSEEK_API_KEY in the
sibling repo's .env.

Usage (from the evaluation/ repo root):

    python benchmarks/rag_recall_beir/run_benchmark_llm_rerank.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_benchmark import (  # noqa: E402
    K_VALUES,
    build_cases,
    build_pipelines,
    build_repository_and_retrievers,
    ingest_corpus,
    load_json,
    make_retrieve_fn,
    report_chunking,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agent import DeepSeekLLM, CallableReranker  # noqa: E402
from agent.rag.pipeline import RAGConfig, RAGPipeline  # noqa: E402

from agent_eval.retrieval_metrics import evaluate_retrieval  # noqa: E402
from agent_eval.stats import bootstrap_ci, paired_bootstrap_test  # noqa: E402

from llm_reranker import RerankCache, build_llm_rerank_scorer  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "results_llm_rerank.json"
RERANK_CACHE_PATH = HERE / ".llm_rerank_cache.json"


def main() -> None:
    corpus = load_json("corpus.json")
    queries = load_json("queries.json")
    qrels = load_json("qrels.json")

    repository, ingestion, bm25, dense, embeddings = build_repository_and_retrievers()
    print(f"Ingesting {len(corpus)} documents (all embeddings cached -- no API calls)...")
    logical_id_to_chunk_ids = ingest_corpus(ingestion, corpus)
    report_chunking(logical_id_to_chunk_ids, corpus)
    cases = build_cases(queries, qrels, logical_id_to_chunk_ids)
    print(f"Scored queries: {len(cases)}\n")

    pipelines = build_pipelines(repository, bm25, dense)

    rerank_llm = DeepSeekLLM()  # deepseek-chat (V3)
    failure_counter = [0]
    rerank_cache = RerankCache(str(RERANK_CACHE_PATH))
    # This is every entry in the cache *file*, not "queries already done for
    # this run" -- the cache key includes the prompt template's own text
    # (see RerankCache's docstring), so an entry from a since-edited prompt
    # is still counted here but won't actually be served to this run; only
    # a rerun with the exact same prompt+model as some prior run gets a
    # real head start from this number.
    print(f"LLM rerank cache file: {rerank_cache.stats()['cached_entries']} total entries on disk "
          f"(across every model/prompt version ever run here) -- entries matching this run's "
          f"current model+prompt are reused; a rerun resumes from wherever this run leaves off.\n")
    scorer = build_llm_rerank_scorer(rerank_llm, failure_counter=failure_counter, cache=rerank_cache)
    config = RAGConfig(candidate_limit=30, evidence_limit=15, minimum_evidence=1)
    pipelines["llm_rerank"] = RAGPipeline(
        repository, lexical=bm25, dense=dense, reranker=CallableReranker(scorer), config=config
    )

    summary: Dict[str, dict] = {}
    mrr_per_query: Dict[str, List[float]] = {}
    for name, pipeline in pipelines.items():
        retrieve = make_retrieve_fn(pipeline)
        started = time.perf_counter()
        overall = evaluate_retrieval(cases, retrieve, name=name, k_values=K_VALUES)
        elapsed = time.perf_counter() - started
        print(overall.render() + f"  ({elapsed:.1f}s, {elapsed / len(cases) * 1000:.0f}ms/query)")

        mrr_values = [row["mrr"] for row in overall.per_query]
        mrr_per_query[name] = mrr_values
        mrr_ci = bootstrap_ci(mrr_values, seed=0)
        print(f"  mrr 95% CI: {mrr_ci.render()}")

        summary[name] = {
            "overall": overall.average(),
            "mrr_95ci": {"low": mrr_ci.low, "high": mrr_ci.high},
            "elapsed_seconds": elapsed,
        }
        print()

    print(f"llm_rerank parse failures (fell back to plain RRF order for that query): "
          f"{failure_counter[0]}/{len(cases)}")

    print("\nPaired bootstrap significance (MRR, same queries both sides):")
    significance: Dict[str, dict] = {}
    pairs = (
        ("hybrid_rrf", "llm_rerank"),
        ("hybrid_rerank", "llm_rerank"),
    )
    for a_name, b_name in pairs:
        test = paired_bootstrap_test(mrr_per_query[a_name], mrr_per_query[b_name], seed=0)
        print(f"  {a_name} -> {b_name}: {test.render()}")
        significance[f"{a_name}_vs_{b_name}"] = {"mean_diff": test.mean_diff, "p_value": test.p_value}
    summary["_significance"] = significance
    summary["_meta"] = {
        "n_documents": len(corpus),
        "n_queries": len(cases),
        "llm_rerank_model": getattr(rerank_llm, "model", "unknown"),
        "llm_rerank_parse_failures": failure_counter[0],
    }

    embeddings.flush()
    rerank_cache.flush()
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
