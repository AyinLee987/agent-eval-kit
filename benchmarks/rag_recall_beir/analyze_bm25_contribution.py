"""Symmetric check to analyze_rrf_reordering.py: that script found cases
where BM25 costs hybrid a correct MRR hit. This one asks the opposite
question -- are there queries where BM25 finds the right answer and dense
alone does not, and does hybrid actually keep that win?

Uses only cached embeddings -- no new API calls.

Usage (from the evaluation/ repo root):

    python benchmarks/rag_recall_beir/analyze_bm25_contribution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_benchmark import (  # noqa: E402
    build_cases,
    build_repository_and_retrievers,
    ingest_corpus,
    load_json,
    report_chunking,
)


def main() -> None:
    corpus = load_json("corpus.json")
    queries = load_json("queries.json")
    qrels = load_json("qrels.json")

    repository, ingestion, bm25, dense, embeddings = build_repository_and_retrievers()
    print(f"Ingesting {len(corpus)} documents (all embeddings cached -- no API calls)...")
    logical_id_to_chunk_ids = ingest_corpus(ingestion, corpus)
    report_chunking(logical_id_to_chunk_ids, corpus)
    cases = build_cases(queries, qrels, logical_id_to_chunk_ids)
    query_text_to_relevant = {c.query: c.relevant_ids for c in cases}

    from agent.rag.pipeline import RAGConfig, RAGPipeline
    from agent.rag.query import MedicalQueryPlanner

    planner = MedicalQueryPlanner()
    config = RAGConfig(candidate_limit=30, evidence_limit=15, minimum_evidence=1)

    class IdentityReranker:
        def rerank(self, query, evidence):
            return list(evidence)

    hybrid = RAGPipeline(repository, lexical=bm25, dense=dense, reranker=IdentityReranker(), config=config)

    bm25_wins_dense_misses = []  # bm25-only MRR=1.0, dense-only MRR=0 at rank 1
    hybrid_keeps_bm25_win = 0
    hybrid_loses_bm25_win = 0

    examined = 0
    for case in cases:
        relevant = query_text_to_relevant[case.query]
        planned = planner.plan(case.query)
        bm25_hits = bm25.search(planned, 30)
        dense_hits = dense.search(planned, 30)
        if not bm25_hits or not dense_hits:
            continue
        examined += 1

        bm25_top_ok = bm25_hits[0].chunk_id in relevant
        dense_top_ok = dense_hits[0].chunk_id in relevant
        if not (bm25_top_ok and not dense_top_ok):
            continue  # only care about "bm25 right, dense wrong at rank 1"

        hybrid_bundle = hybrid.retrieve(case.query)
        hybrid_ranked = [e.chunk.id for e in hybrid_bundle.evidence]
        hybrid_top_ok = bool(hybrid_ranked) and hybrid_ranked[0] in relevant

        bm25_wins_dense_misses.append({
            "query": case.query,
            "bm25_chunk": bm25_hits[0].chunk_id,
            "dense_chunk": dense_hits[0].chunk_id,
            "dense_rank_of_bm25_chunk": next(
                (r for r, h in enumerate(dense_hits, 1) if h.chunk_id == bm25_hits[0].chunk_id), None
            ),
            "hybrid_top": hybrid_ranked[0] if hybrid_ranked else None,
            "hybrid_kept_win": hybrid_top_ok,
        })
        if hybrid_top_ok:
            hybrid_keeps_bm25_win += 1
        else:
            hybrid_loses_bm25_win += 1

    print(f"\nExamined {examined} queries.")
    print(f"Queries where BM25's #1 was correct and dense-only's #1 was NOT: "
          f"{len(bm25_wins_dense_misses)} / {examined}")
    print(f"  hybrid RRF kept BM25's correct #1 in first place: {hybrid_keeps_bm25_win}")
    print(f"  hybrid RRF lost it (some other chunk displaced it): {hybrid_loses_bm25_win}")

    print("\nExamples:")
    for row in bm25_wins_dense_misses[:6]:
        print(f"\n=== Query: {row['query']!r} ===")
        print(f"  BM25's #1 (correct): chunk {row['bm25_chunk']}"
              f" -- dense's own rank for this chunk: {row['dense_rank_of_bm25_chunk'] or 'not in top 30'}")
        print(f"  dense-only's #1 instead: chunk {row['dense_chunk']} (not a labeled answer)")
        print(f"  hybrid RRF's #1: chunk {row['hybrid_top']} "
              f"({'kept BM25 correct pick' if row['hybrid_kept_win'] else 'lost it too'})")


if __name__ == "__main__":
    main()
