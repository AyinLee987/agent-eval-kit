"""Finds concrete queries where dense-only's #1 hit is a real answer, but
RRF fusion (see RESULTS.md's "A genuine surprise" section) pushed it down
by combining a different chunk's BM25 + dense scores. Verifies that claim
against real per-retriever rank data instead of leaving it as theory.

Uses only cached embeddings -- no new API calls.

Usage (from the evaluation/ repo root):

    python benchmarks/rag_recall_beir/analyze_rrf_reordering.py
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

MEDICAL_QUERY_PLANNER_IMPORT = "agent.rag.query"


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

    from agent.rag.query import MedicalQueryPlanner
    from agent.rag.pipeline import RAGConfig, RAGPipeline

    planner = MedicalQueryPlanner()
    config = RAGConfig(candidate_limit=30, evidence_limit=15, minimum_evidence=1)
    hybrid = RAGPipeline(repository, lexical=bm25, dense=dense, config=config)

    def rank_map(hits):
        return {hit.chunk_id: (rank, hit.score) for rank, hit in enumerate(hits, start=1)}

    found = 0
    examined = 0
    for case in cases:
        relevant = query_text_to_relevant[case.query]
        planned = planner.plan(case.query)
        dense_hits = dense.search(planned, 30)
        bm25_hits = bm25.search(planned, 30)
        if not dense_hits:
            continue
        examined += 1

        dense_top = dense_hits[0].chunk_id
        if dense_top not in relevant:
            continue  # dense's #1 wasn't even a real answer -- not the case we're after

        hybrid_bundle = hybrid.retrieve(case.query)
        hybrid_ranked = [e.chunk.id for e in hybrid_bundle.evidence]
        if not hybrid_ranked or hybrid_ranked[0] == dense_top:
            continue  # hybrid agreed with dense -- no reordering happened

        dense_ranks = rank_map(dense_hits)
        bm25_ranks = rank_map(bm25_hits)
        displacer = hybrid_ranked[0]

        found += 1
        print(f"\n=== Query: {case.query!r} ===")
        print(f"  dense-only's #1 (a real answer, MRR=1.0 for dense-only): chunk {dense_top}")
        print(f"    bm25 rank for this chunk: {bm25_ranks.get(dense_top, 'not in top 30')}")
        print(f"  hybrid RRF's #1 instead: chunk {displacer}"
              f"{' (also a real answer)' if displacer in relevant else ' (NOT a labeled answer)'}")
        print(f"    dense rank: {dense_ranks.get(displacer, 'not in top 30')}, "
              f"bm25 rank: {bm25_ranks.get(displacer, 'not in top 30')}")
        if found >= 5:
            break

    print(f"\nExamined {examined} queries with a dense-only hit; found {found} where RRF "
          "reordered dense's real #1 answer out of first place (stopped at 5 examples).")


if __name__ == "__main__":
    main()
