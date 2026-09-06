"""RAG recall on BEIR NFCorpus -- a published, non-self-authored benchmark.

Same four-pipeline ablation as ../rag_recall (BM25-only / dense-only /
hybrid RRF / hybrid + HeuristicReranker), against the sibling project's real
RAGPipeline and a real embedding endpoint, but on NFCorpus (Boteva et al.,
2016) instead of a hand-authored synthetic corpus: 3,633 real biomedical
documents, 323 real relevance-judged test queries, graded relevance (score
0/1/2) from the original BEIR release -- not a fact-substring resolved at
ingestion time. This is what ../rag_recall/RESULTS.md's "synthetic corpus"
caveat looks like addressed directly, at ~13x the query count.

Run `python benchmarks/rag_recall_beir/download_nfcorpus.py` once first to
fetch corpus.json / queries.json / qrels.json (checked into the repo after
that -- this script itself makes no HuggingFace requests).

Usage (from the evaluation/ repo root):

    python benchmarks/rag_recall_beir/run_benchmark.py
"""

from __future__ import annotations

import collections
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, TYPE_CHECKING

EVAL_ROOT = Path(__file__).resolve().parents[2]
HARNESS_REPO = Path(os.environ.get("AGENT_HARNESS_PATH", EVAL_ROOT.parent / "agent" / "agent-harness-from-scratch"))

if TYPE_CHECKING:
    from agent import RAGIngestionService, RAGPipeline
    from agent.rag.models import MedicalQuery
    from agent_eval.retrieval_metrics import RetrievalCase


def initialize_environment() -> None:
    """Load live benchmark dependencies and credentials only on explicit run."""

    for path in (str(EVAL_ROOT), str(HARNESS_REPO)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from dotenv import load_dotenv

    if not load_dotenv(dotenv_path=HARNESS_REPO / ".env"):
        raise SystemExit(f"Could not find a .env file at {HARNESS_REPO / '.env'}.")


HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "results.json"
CACHE_PATH = HERE / ".embedding_cache.json"
K_VALUES = (3, 5, 10)
# Recall/MRR treat scores > 0 as relevant. nDCG retains published grades
# with trec_eval's linear-gain convention.
RELEVANCE_THRESHOLD = 1


class IdentityReranker:
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
            "embedding endpoint -- see the sibling repo's README RAG section."
        )
    return values  # type: ignore[return-value]


def load_json(name: str) -> list:
    path = HERE / name
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. Run "
            "`python benchmarks/rag_recall_beir/download_nfcorpus.py` first."
        )
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def render_markdown(doc: dict) -> str:
    """Render one corpus document for ingestion; it may produce several chunks."""

    heading = doc["title"].strip() or doc["id"]
    return f"## {heading}\n{doc['text']}"


def build_repository_and_retrievers():
    from agent import (
        BM25Retriever, DenseRetriever, InMemoryRAGRepository,
        MedicalParentChildChunker, OpenAICompatibleEmbeddingProvider,
        RAGIngestionService,
    )
    from benchmarks.rag_recall.cached_embeddings import CachedEmbeddingProvider

    embed_config = require_real_embedding_config()
    repository = InMemoryRAGRepository()
    chunker = MedicalParentChildChunker()

    bm25 = BM25Retriever(repository)
    real_embeddings = OpenAICompatibleEmbeddingProvider(
        model=embed_config["model"],
        api_key=embed_config["api_key"],
        base_url=embed_config["base_url"],
        provider_name="rag-recall-beir-benchmark",
    )
    embeddings = CachedEmbeddingProvider(real_embeddings, str(CACHE_PATH))
    dense = DenseRetriever(repository, embeddings)

    ingestion = RAGIngestionService(repository, chunker, [bm25, dense])
    return repository, ingestion, bm25, dense, embeddings


def ingest_corpus(ingestion: RAGIngestionService, corpus: List[dict]) -> Dict[str, Set[str]]:
    """Ingest every document; return {corpus_id: {child chunk ids}}."""

    logical_id_to_chunk_ids: Dict[str, Set[str]] = {}
    total = len(corpus)
    for i, doc in enumerate(corpus, start=1):
        result = ingestion.ingest_text(
            logical_id=doc["id"],
            title=doc["title"].strip() or doc["id"],
            content=render_markdown(doc),
            publisher="beir-nfcorpus",
            document_type="reference",
            jurisdiction="",
            language="en",
            version="1",
        )
        child_ids = {c.id for c in result.chunks if c.chunk_type == "child"}
        logical_id_to_chunk_ids[doc["id"]] = child_ids
        if i % 200 == 0 or i == total:
            print(f"  ingested {i}/{total} documents...")
    return logical_id_to_chunk_ids


def report_chunking(logical_id_to_chunk_ids: Dict[str, Set[str]], corpus: List[dict]) -> None:
    """Describe chunking without converting document judgments into chunk labels."""

    counts = collections.Counter(len(ids) for ids in logical_id_to_chunk_ids.values())
    zero = counts.get(0, 0)
    multi = sum(n for count, n in counts.items() if count >= 2)
    print(f"  chunking: {counts.get(1, 0)} docs -> 1 chunk, {multi} docs -> 2+ chunks, "
          f"{zero} docs -> 0 chunks (chunk-count distribution: {dict(sorted(counts.items()))})")
    if zero:
        print(f"  {zero} documents produced no child chunk; their qrels and queries "
              "remain in document-level evaluation and coverage reporting.")


def build_cases(
    queries: List[dict],
    qrels: List[dict],
    logical_id_to_chunk_ids: Optional[Mapping[str, Set[str]]] = None,
) -> List[RetrievalCase]:
    """Keep original document-level grades and every supplied query.

    The optional third argument remains accepted for older callers. Ingested
    chunks never determine the qrels denominator: unavailable relevant
    documents remain relevant, and their missing coverage is reported.
    """

    from agent_eval.retrieval_metrics import RetrievalCase

    grades_by_query: Dict[str, Dict[str, float]] = {}
    for row in qrels:
        score = row["score"]
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or score < 0):
            raise ValueError("Qrels scores must be finite nonnegative numbers.")
        query_id, corpus_id = str(row["query_id"]), str(row["corpus_id"])
        grades = grades_by_query.setdefault(query_id, {})
        if corpus_id in grades and grades[corpus_id] != score:
            raise ValueError(f"Conflicting qrels for query {query_id!r}, document {corpus_id!r}.")
        grades[corpus_id] = float(score)

    cases = []
    query_ids: Set[str] = set()
    for query in queries:
        query_id = str(query["id"])
        if query_id in query_ids:
            raise ValueError(f"Duplicate query_id {query_id!r}.")
        query_ids.add(query_id)
        cases.append(RetrievalCase(
            query=query["text"], query_id=query_id,
            relevant_ids=grades_by_query.get(query_id, {}),
        ))
    return cases


def build_chunk_to_document(
    logical_id_to_chunk_ids: Mapping[str, Set[str]],
) -> Dict[str, str]:
    """Invert ingestion identities; a chunk must belong to exactly one document."""

    chunk_to_document: Dict[str, str] = {}
    for document_id, chunk_ids in logical_id_to_chunk_ids.items():
        for chunk_id in chunk_ids:
            existing = chunk_to_document.get(chunk_id)
            if existing is not None and existing != document_id:
                raise ValueError(f"Chunk {chunk_id!r} belongs to multiple documents.")
            chunk_to_document[chunk_id] = document_id
    return chunk_to_document


def rank_document_ids(
    chunk_ids: Sequence[str], chunk_to_document: Mapping[str, str],
) -> List[str]:
    """Rank documents by their first retrieved chunk, with stable deduplication."""

    ranked: List[str] = []
    seen: Set[str] = set()
    for chunk_id in chunk_ids:
        if chunk_id not in chunk_to_document:
            raise ValueError(f"Retrieved chunk {chunk_id!r} has no document mapping.")
        document_id = chunk_to_document[chunk_id]
        if document_id not in seen:
            ranked.append(document_id)
            seen.add(document_id)
    return ranked


def document_coverage(
    corpus: List[dict],
    cases: Sequence[RetrievalCase],
    logical_id_to_chunk_ids: Mapping[str, Set[str]],
) -> Dict[str, Any]:
    """Report ingestion losses without removing documents or query judgments."""

    corpus_ids = {str(doc["id"]) for doc in corpus}
    indexed_ids = {
        document_id for document_id, chunks in logical_id_to_chunk_ids.items() if chunks
    }
    relevant_documents: Set[str] = set()
    per_query = []
    for case in cases:
        if isinstance(case.relevant_ids, Mapping):
            relevant = {doc for doc, score in case.relevant_ids.items() if score > 0}
        else:
            relevant = set(case.relevant_ids)
        relevant_documents.update(relevant)
        available = relevant & indexed_ids
        per_query.append({
            "query_id": case.query_id,
            "relevant_document_count": len(relevant),
            "retrievable_relevant_document_count": len(available),
            "relevant_document_coverage": len(available) / len(relevant) if relevant else None,
            "missing_relevant_document_ids": sorted(relevant - indexed_ids),
        })
    return {
        "corpus_document_count": len(corpus_ids),
        "indexed_corpus_document_count": len(corpus_ids & indexed_ids),
        "corpus_document_coverage": (
            len(corpus_ids & indexed_ids) / len(corpus_ids) if corpus_ids else None
        ),
        "missing_corpus_document_ids": sorted(corpus_ids - indexed_ids),
        "relevant_document_count": len(relevant_documents),
        "indexed_relevant_document_count": len(relevant_documents & indexed_ids),
        "relevant_document_coverage": (
            len(relevant_documents & indexed_ids) / len(relevant_documents)
            if relevant_documents else None
        ),
        "positive_judgment_count": sum(row["relevant_document_count"] for row in per_query),
        "indexed_positive_judgment_count": sum(
            row["retrievable_relevant_document_count"] for row in per_query
        ),
        "queries_without_retrievable_relevant_documents": [
            row["query_id"] for row in per_query
            if row["relevant_document_count"] and not row["retrievable_relevant_document_count"]
        ],
        "per_query": per_query,
    }


def build_pipelines(repository, bm25, dense) -> Dict[str, RAGPipeline]:
    from agent import RAGConfig, RAGPipeline

    config = RAGConfig(candidate_limit=30, evidence_limit=15, minimum_evidence=1)
    return {
        "bm25_only": RAGPipeline(repository, lexical=bm25, dense=None, config=config),
        "dense_only": RAGPipeline(repository, lexical=None, dense=dense, config=config),
        "hybrid_rrf": RAGPipeline(
            repository, lexical=bm25, dense=dense, reranker=IdentityReranker(), config=config
        ),
        "hybrid_rerank": RAGPipeline(repository, lexical=bm25, dense=dense, config=config),
    }


def make_retrieve_fn(
    pipeline: RAGPipeline, logical_id_to_chunk_ids: Mapping[str, Set[str]],
):
    chunk_to_document = build_chunk_to_document(logical_id_to_chunk_ids)

    def retrieve(query_text: str) -> List[str]:
        bundle = pipeline.retrieve(query_text)
        return rank_document_ids(
            [item.chunk.id for item in bundle.evidence], chunk_to_document
        )

    return retrieve


SIGNIFICANCE_PAIRS = (("bm25_only", "hybrid_rrf"), ("hybrid_rrf", "hybrid_rerank"))


def main() -> None:
    initialize_environment()
    from agent_eval.retrieval_metrics import evaluate_retrieval
    from agent_eval.stats import bootstrap_ci, paired_permutation_test

    corpus = load_json("corpus.json")
    queries = load_json("queries.json")
    qrels = load_json("qrels.json")

    repository, ingestion, bm25, dense, embeddings = build_repository_and_retrievers()

    print(f"Ingesting {len(corpus)} NFCorpus documents (cached embeddings reused across runs)...")
    logical_id_to_chunk_ids = ingest_corpus(ingestion, corpus)
    report_chunking(logical_id_to_chunk_ids, corpus)

    cases = build_cases(queries, qrels, logical_id_to_chunk_ids)
    coverage = document_coverage(corpus, cases, logical_id_to_chunk_ids)
    pipelines = build_pipelines(repository, bm25, dense)

    summary: Dict[str, dict] = {}
    mrr_per_query: Dict[str, Dict[str, float]] = {}
    print(f"\nCorpus: {len(corpus)} documents. Retained queries: {len(cases)} "
          f"(of {len(queries)} supplied; positive grades determine applicability).")
    print(f"Embedding model: {embeddings.model_id}  ({embeddings.stats()['cached_vectors']} vectors cached)\n")

    for name, pipeline in pipelines.items():
        retrieve = make_retrieve_fn(pipeline, logical_id_to_chunk_ids)
        overall = evaluate_retrieval(cases, retrieve, name=name, k_values=K_VALUES)
        print(overall.render())

        mrr_values = [row["mrr"] for row in overall.per_query if row["status"] == "ok"]
        mrr_per_query[name] = {
            row["query_id"]: row["mrr"] for row in overall.per_query if row["status"] == "ok"
        }
        mrr_ci = bootstrap_ci(mrr_values, seed=0) if len(mrr_values) >= 2 else None
        print(f"  mrr 95% CI: {mrr_ci.render() if mrr_ci else 'not applicable (n < 2)'}")

        summary[name] = {
            "overall": overall.average(), "counts": overall.counts(),
            "per_query": overall.per_query,
            "mrr_95ci": {"low": mrr_ci.low, "high": mrr_ci.high,
                         "method": mrr_ci.method, "n": mrr_ci.n,
                         "n_resamples": mrr_ci.n_resamples} if mrr_ci else None,
        }
        print()

    print("Paired permutation significance (MRR, same queries both sides):")
    significance: Dict[str, dict] = {}
    for a_name, b_name in SIGNIFICANCE_PAIRS:
        left, right = mrr_per_query[a_name], mrr_per_query[b_name]
        paired_ids = [case.query_id for case in cases if case.query_id in left and case.query_id in right]
        paired_key = f"{a_name}_vs_{b_name}"
        pair_meta = {
            "method": "paired_permutation", "paired_query_ids": paired_ids,
            "paired_count": len(paired_ids),
            "excluded_query_ids": [case.query_id for case in cases if case.query_id not in paired_ids],
        }
        if len(paired_ids) < 2:
            significance[paired_key] = {
                **pair_meta, "status": "not_applicable", "reason": "paired n < 2",
            }
            continue
        test = paired_permutation_test(
            [left[query_id] for query_id in paired_ids],
            [right[query_id] for query_id in paired_ids], seed=0,
        )
        print(f"  {a_name} -> {b_name}: {test.render()}")
        significance[paired_key] = {
            **pair_meta, "status": "ok", "mean_diff": test.mean_diff, "p_value": test.p_value,
            "sampling_method": test.method, "n_resamples": test.n_resamples,
        }
    summary["_significance"] = significance
    summary["_meta"] = {
        "n_documents": len(corpus), "n_queries": len(cases),
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "scoring_version": 2, "level": "document",
        "judgment_level": "document", "ndcg_gain": "linear",
        "document_ranking": "first_occurrence_in_chunk_ranking",
        "chunk_judgments": "not provided by NFCorpus; no chunk-level score is reported",
        "coverage": coverage,
    }

    embeddings.flush()  # catch whatever's accumulated since the last periodic flush
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
