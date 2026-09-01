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

from agent_eval.retrieval_metrics import RetrievalCase, evaluate_retrieval  # noqa: E402
from agent_eval.stats import bootstrap_ci, paired_bootstrap_test  # noqa: E402

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
#: ``cases`` list in the same order, so their per_query rows line up
#: index-for-index -- exactly what paired_bootstrap_test needs.
SIGNIFICANCE_PAIRS = (("bm25_only", "hybrid_rrf"), ("hybrid_rrf", "hybrid_rerank"))


def main() -> None:
    repository, bm25, dense, embeddings = build_repository_and_retrievers()
    check_one_chunk_per_section(repository)

    cases = [
        RetrievalCase(query=case["query"], relevant_ids=resolve_relevant_ids(case["fact"], repository))
        for case in CASES
    ]
    style_by_query = {case["query"]: case["style"] for case in CASES}

    pipelines = build_pipelines(repository, bm25, dense)

    summary: Dict[str, dict] = {}
    mrr_per_query: Dict[str, List[float]] = {}
    print(f"Corpus: {len(DOCUMENTS)} documents, {len(cases)} labeled queries.")
    print(f"Embedding model: {embeddings.model_id}  ({embeddings.stats()['cached_vectors']} vectors cached)\n")

    for name, pipeline in pipelines.items():
        retrieve = make_retrieve_fn(pipeline)
        overall = evaluate_retrieval(cases, retrieve, name=name, k_values=K_VALUES)
        print(overall.render())

        mrr_values = [row["mrr"] for row in overall.per_query]
        mrr_per_query[name] = mrr_values
        mrr_ci = bootstrap_ci(mrr_values, seed=0)
        print(f"  mrr 95% CI: {mrr_ci.render()}")

        by_style = {}
        for style in ("lexical", "paraphrase"):
            subset = [c for c in cases if style_by_query[c.query] == style]
            report = evaluate_retrieval(subset, retrieve, name=f"{name}:{style}", k_values=K_VALUES)
            by_style[style] = report.average()
            print("  " + report.render())

        summary[name] = {
            "overall": overall.average(),
            "by_style": by_style,
            "mrr_95ci": {"low": mrr_ci.low, "high": mrr_ci.high},
        }
        print()

    print("Paired bootstrap significance (MRR, same 24 queries both sides):")
    significance: Dict[str, dict] = {}
    for a_name, b_name in SIGNIFICANCE_PAIRS:
        test = paired_bootstrap_test(mrr_per_query[a_name], mrr_per_query[b_name], seed=0)
        print(f"  {a_name} -> {b_name}: {test.render()}")
        significance[f"{a_name}_vs_{b_name}"] = {
            "mean_diff": test.mean_diff,
            "p_value": test.p_value,
        }
    summary["_significance"] = significance
    embeddings.flush()  # catch whatever's accumulated since the last periodic flush
    print(
        "\nRead p >= 0.05 as 'this benchmark's n=24 queries cannot rule out "
        "resampling noise producing a gap this size' -- not as 'no real "
        "difference exists'. See RESULTS.md's caveats section."
    )

    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
