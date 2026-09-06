"""RAG recall ablation: BM25-only vs dense-only vs hybrid(RRF) vs hybrid+rerank.

Ingests the synthetic corpus (corpus.py) into the sibling
agent-harness-from-scratch's RAG pipeline, resolves each labeled query in
queries.py to its ground-truth chunk id, and reports Recall@K / MRR / nDCG@K
per pipeline configuration — overall and broken down by query style
(lexical vs paraphrase), since only the paraphrase style should show a real
semantic-retrieval advantage.

Requires a real embedding endpoint (RAG_EMBEDDING_MODEL / _API_KEY /
_BASE_URL in the sibling repo's .env) — refuses to run against the
zero-dependency MockLLM hash embedding, which would make "dense" and
"hybrid" numbers meaningless.

Usage (from the evaluation/ repo root):

    python benchmarks/rag_recall/run_benchmark.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

EVAL_ROOT = Path(__file__).resolve().parents[2]
HARNESS_REPO = Path(os.environ.get("AGENT_HARNESS_PATH", EVAL_ROOT.parent / "agent" / "agent-harness-from-scratch"))

for path in (str(EVAL_ROOT), str(HARNESS_REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dotenv import load_dotenv  # noqa: E402

if not load_dotenv(dotenv_path=HARNESS_REPO / ".env"):
    raise SystemExit(f"Could not find a .env file at {HARNESS_REPO / '.env'}.")

from agent import (  # noqa: E402
    BM25Retriever,
    DenseRetriever,
    InMemoryRAGRepository,
    MedicalParentChildChunker,
    OpenAICompatibleEmbeddingProvider,
    RAGConfig,
    RAGIngestionService,
    RAGPipeline,
)
from agent.rag.models import MedicalQuery  # noqa: E402

from agent_eval.retrieval_metrics import RetrievalCase, RetrievalReport, evaluate_retrieval  # noqa: E402
from agent_eval.stats import bootstrap_ci, paired_permutation_test  # noqa: E402

from cached_embeddings import CachedEmbeddingProvider  # noqa: E402
from corpus import DOCUMENTS, render_markdown  # noqa: E402
from queries import CASES  # noqa: E402

RESULTS_PATH = Path(__file__).with_name("results.json")
CACHE_PATH = Path(__file__).with_name(".embedding_cache.json")
K_VALUES = (3, 5, 10)


class IdentityReranker:
    """No-op reranker: preserves reciprocal-rank-fusion order untouched.

    Used for the "hybrid, no rerank" ablation arm so the only difference
    from "hybrid + rerank" is the reranking step itself.
    """

    def rerank(self, query: MedicalQuery, evidence):
        return list(evidence)


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
            f"(missing: {', '.join(missing)}). This benchmark requires a real "
            "embedding endpoint — see the sibling repo's README RAG section."
        )
    return values  # type: ignore[return-value]


def build_repository_and_retrievers():
    embed_config = require_real_embedding_config()
    repository = InMemoryRAGRepository()
    chunker = MedicalParentChildChunker()

    bm25 = BM25Retriever(repository)
    real_embeddings = OpenAICompatibleEmbeddingProvider(
        model=embed_config["model"],
        api_key=embed_config["api_key"],
        base_url=embed_config["base_url"],
        provider_name="rag-recall-benchmark",
    )
    embeddings = CachedEmbeddingProvider(real_embeddings, str(CACHE_PATH))
    dense = DenseRetriever(repository, embeddings)

    ingestion = RAGIngestionService(repository, chunker, [bm25, dense])
    for document in DOCUMENTS:
        ingestion.ingest_text(
            logical_id=document["logical_id"],
            title=document["title"],
            content=render_markdown(document),
            publisher="synthetic-benchmark",
            document_type="handbook",
            jurisdiction="",
            language="en",
            version="1",
        )

    return repository, bm25, dense, embeddings


def check_one_chunk_per_section(repository) -> None:
    """Fail loudly if a section didn't become exactly one child chunk.

    ``queries.py`` assumes fact-substring ground truth resolves to exactly
    one chunk; if chunking ever groups multiple sections together (e.g.
    after a corpus edit), that assumption silently breaks and Recall
    numbers become meaningless without this check.
    """

    expected_children = sum(len(doc["sections"]) for doc in DOCUMENTS)
    actual_children = sum(1 for chunk in repository.active_chunks() if chunk.chunk_type == "child")
    if actual_children != expected_children:
        raise SystemExit(
            f"Expected {expected_children} child chunks (one per section) but got "
            f"{actual_children}. Chunking assumptions in queries.py no longer hold — "
            "inspect corpus.py section lengths before trusting these results."
        )


def resolve_relevant_ids(fact: str, repository) -> set[str]:
    """Find the chunk(s) whose text contains ``fact`` (case-insensitive).

    Raises if a fact matches zero or more than one chunk — either means the
    ground truth in queries.py doesn't uniquely identify a chunk, which
    would make Recall/MRR/nDCG for that query meaningless rather than just
    imprecise.
    """

    needle = fact.lower()
    matches = {
        chunk.id
        for chunk in repository.active_chunks()
        if chunk.chunk_type == "child" and needle in chunk.text.lower()
    }
    if len(matches) != 1:
        raise SystemExit(
            f"Fact {fact!r} matched {len(matches)} chunks (expected exactly 1). "
            "Fix the fact substring in queries.py so it's unique."
        )
    return matches


def build_pipelines(repository, bm25, dense) -> Dict[str, RAGPipeline]:
    config = RAGConfig(candidate_limit=30, evidence_limit=15, minimum_evidence=1)
    return {
        "bm25_only": RAGPipeline(repository, lexical=bm25, dense=None, config=config),
        "dense_only": RAGPipeline(repository, lexical=None, dense=dense, config=config),
        "hybrid_rrf": RAGPipeline(
            repository, lexical=bm25, dense=dense, reranker=IdentityReranker(), config=config
        ),
        "hybrid_rerank": RAGPipeline(repository, lexical=bm25, dense=dense, config=config),
    }


def make_retrieve_fn(pipeline: RAGPipeline):
    def retrieve(query_text: str) -> List[str]:
        bundle = pipeline.retrieve(query_text)
        return [item.chunk.id for item in bundle.evidence]

    return retrieve


#: The two comparisons RESULTS.md's headline claims actually rest on:
#: "hybrid RRF beats BM25-only" and "the heuristic reranker hurts relative
#: to plain RRF fusion". Every pipeline is run against the same fixed
#: cases are paired by stable query ID after failed observations are excluded.
SIGNIFICANCE_PAIRS = (("bm25_only", "hybrid_rrf"), ("hybrid_rrf", "hybrid_rerank"))


def finite_metric_by_query(report: RetrievalReport, key: str) -> Dict[str, float]:
    """Select successful finite scores while preserving their query identities."""

    scores: Dict[str, float] = {}
    seen = set()
    for row in report.per_query:
        query_id = row["query_id"]
        if query_id in seen:
            raise ValueError(f"Duplicate query_id {query_id!r} in retrieval report.")
        seen.add(query_id)
        value = row.get(key)
        if (row.get("status") == "ok" and not isinstance(value, bool)
                and isinstance(value, (int, float)) and math.isfinite(value)):
            scores[query_id] = float(value)
    return scores


def main() -> None:
    query_ids = [str(case["id"]) for case in CASES]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("Synthetic query IDs must be unique.")
    repository, bm25, dense, embeddings = build_repository_and_retrievers()
    check_one_chunk_per_section(repository)

    cases = [
        RetrievalCase(
            query=case["query"], query_id=str(case["id"]),
            relevant_ids=resolve_relevant_ids(case["fact"], repository),
        )
        for case in CASES
    ]
    style_by_query_id = {str(case["id"]): case["style"] for case in CASES}
    pipelines = build_pipelines(repository, bm25, dense)

    summary: Dict[str, dict] = {}
    mrr_per_query: Dict[str, Dict[str, float]] = {}
    print(f"Corpus: {len(DOCUMENTS)} documents, {len(cases)} labeled queries.")
    print(f"Embedding model: {embeddings.model_id}  ({embeddings.stats()['cached_vectors']} vectors cached)\n")

    for name, pipeline in pipelines.items():
        retrieve = make_retrieve_fn(pipeline)
        overall = evaluate_retrieval(cases, retrieve, name=name, k_values=K_VALUES)
        for row in overall.per_query:
            row["style"] = style_by_query_id[row["query_id"]]
        print(overall.render())

        scores = finite_metric_by_query(overall, "mrr")
        mrr_per_query[name] = scores
        mrr_values = list(scores.values())
        mrr_ci = bootstrap_ci(mrr_values, seed=0) if len(mrr_values) >= 2 else None
        print(f"  mrr 95% CI: {mrr_ci.render() if mrr_ci else 'not applicable (n < 2)'}")

        by_style = {}
        by_style_counts = {}
        for style in ("lexical", "paraphrase"):
            report = RetrievalReport(
                name=f"{name}:{style}", k_values=overall.k_values,
                per_query=[row for row in overall.per_query if row["style"] == style],
            )
            by_style[style] = report.average()
            by_style_counts[style] = report.counts()
            print("  " + report.render())

        summary[name] = {
            "overall": overall.average(),
            "counts": overall.counts(),
            "per_query": overall.per_query,
            "by_style": by_style,
            "by_style_counts": by_style_counts,
            "mrr_95ci": {
                "low": mrr_ci.low, "high": mrr_ci.high,
                "n": mrr_ci.n, "method": mrr_ci.method,
            } if mrr_ci else None,
        }
        print()

    print("Paired permutation significance (MRR, matched successful query IDs):")
    significance: Dict[str, dict] = {}
    for a_name, b_name in SIGNIFICANCE_PAIRS:
        left, right = mrr_per_query[a_name], mrr_per_query[b_name]
        paired_ids = [
            case.query_id for case in cases if case.query_id in left and case.query_id in right
        ]
        pair_key = f"{a_name}_vs_{b_name}"
        pair_meta = {
            "method": "paired_permutation", "paired_query_ids": paired_ids,
            "paired_count": len(paired_ids),
            "excluded_query_ids": [
                case.query_id for case in cases if case.query_id not in paired_ids
            ],
        }
        if len(paired_ids) < 2:
            significance[pair_key] = {
                **pair_meta, "status": "not_applicable", "reason": "paired n < 2",
            }
            continue
        test = paired_permutation_test(
            [left[query_id] for query_id in paired_ids],
            [right[query_id] for query_id in paired_ids], seed=0,
        )
        print(f"  {a_name} -> {b_name}: {test.render()}")
        significance[pair_key] = {
            **pair_meta, "status": "ok", "method": test.method,
            "mean_diff": test.mean_diff, "p_value": test.p_value,
        }
    summary["_significance"] = significance
    summary["_meta"] = {
        "scoring_version": 2, "level": "chunk", "ndcg_gain": "linear",
        "judgment_source": "synthetic fact-substring matches to ingested child chunks",
        "official_document_qrels": False,
        "n_documents": len(DOCUMENTS), "n_queries": len(cases),
        "query_styles": ["lexical", "paraphrase"],
        "style_reports": "slices of the same per-query observations; no retrieval reruns",
    }
    embeddings.flush()
    print(
        "\nThe paired permutation test conditions on matched successful queries "
        "and assumes labels are exchangeable within each pair under the null. "
        "Inspect query failure counts before interpreting its p-value."
    )

    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
