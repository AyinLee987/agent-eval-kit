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
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_recall"))
from cached_embeddings import CachedEmbeddingProvider  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "results.json"
CACHE_PATH = HERE / ".embedding_cache.json"
K_VALUES = (3, 5, 10)
# NFCorpus's published qrels use 0/1/2; treat 1 ("partially relevant") and 2
# ("highly relevant") both as relevant, matching how BEIR's own baselines
# binarize this corpus's graded judgments for Recall/MRR-style metrics
# (this toolkit's retrieval_metrics.py only supports binary relevance --
# see this benchmark's RESULTS.md for what that simplification costs).
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
    """One ``## title`` section per document -- mirrors ../rag_recall/
    corpus.py's convention, which is what makes "each document becomes
    exactly one child chunk" a checkable assumption (see
    check_one_chunk_per_document below) instead of a hope."""

    heading = doc["title"].strip() or doc["id"]
    return f"## {heading}\n{doc['text']}"


def build_repository_and_retrievers():
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
    """Unlike ../rag_recall's short synthetic sections, NFCorpus abstracts
    often exceed MedicalParentChildChunker's max_tokens=400 and split into
    several child chunks -- "one document, one chunk" is not a safe
    assumption here, so this just reports the distribution rather than
    aborting on it. build_cases() below already treats every chunk under a
    relevant document's logical_id as relevant, which is the correct
    ground truth regardless of how many chunks one document became."""

    counts = collections.Counter(len(ids) for ids in logical_id_to_chunk_ids.values())
    zero = counts.get(0, 0)
    multi = sum(n for count, n in counts.items() if count >= 2)
    print(f"  chunking: {counts.get(1, 0)} docs -> 1 chunk, {multi} docs -> 2+ chunks, "
          f"{zero} docs -> 0 chunks (chunk-count distribution: {dict(sorted(counts.items()))})")
    if zero:
        print(f"  WARNING: {zero} documents produced no child chunk and can never be "
              "retrieved -- any query whose only relevant document is one of these "
              "will be excluded by build_cases() below.")


def build_cases(
    queries: List[dict],
    qrels: List[dict],
    logical_id_to_chunk_ids: Dict[str, Set[str]],
) -> List[RetrievalCase]:
    relevant_by_query: Dict[str, Set[str]] = {}
    for row in qrels:
        if row["score"] < RELEVANCE_THRESHOLD:
            continue
        chunk_ids = logical_id_to_chunk_ids.get(row["corpus_id"])
        if not chunk_ids:
            continue  # corpus id not ingested (shouldn't happen; download_nfcorpus.py checks this)
        relevant_by_query.setdefault(row["query_id"], set()).update(chunk_ids)

    cases = []
    dropped = 0
    for query in queries:
        relevant = relevant_by_query.get(query["id"], set())
        if not relevant:
            dropped += 1
            continue
        cases.append(RetrievalCase(query=query["text"], relevant_ids=relevant))
    if dropped:
        print(f"  {dropped}/{len(queries)} judged queries had no relevant chunk after "
              f"thresholding at score>={RELEVANCE_THRESHOLD} -- excluded.")
    return cases


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


SIGNIFICANCE_PAIRS = (("bm25_only", "hybrid_rrf"), ("hybrid_rrf", "hybrid_rerank"))


def main() -> None:
    corpus = load_json("corpus.json")
    queries = load_json("queries.json")
    qrels = load_json("qrels.json")

    repository, ingestion, bm25, dense, embeddings = build_repository_and_retrievers()

    print(f"Ingesting {len(corpus)} NFCorpus documents (cached embeddings reused across runs)...")
    logical_id_to_chunk_ids = ingest_corpus(ingestion, corpus)
    report_chunking(logical_id_to_chunk_ids, corpus)

    cases = build_cases(queries, qrels, logical_id_to_chunk_ids)
    pipelines = build_pipelines(repository, bm25, dense)

    summary: Dict[str, dict] = {}
    mrr_per_query: Dict[str, List[float]] = {}
    print(f"\nCorpus: {len(corpus)} documents. Scored queries: {len(cases)} "
          f"(of {len(queries)} judged, score>={RELEVANCE_THRESHOLD} threshold).")
    print(f"Embedding model: {embeddings.model_id}  ({embeddings.stats()['cached_vectors']} vectors cached)\n")

    for name, pipeline in pipelines.items():
        retrieve = make_retrieve_fn(pipeline)
        overall = evaluate_retrieval(cases, retrieve, name=name, k_values=K_VALUES)
        print(overall.render())

        mrr_values = [row["mrr"] for row in overall.per_query]
        mrr_per_query[name] = mrr_values
        mrr_ci = bootstrap_ci(mrr_values, seed=0)
        print(f"  mrr 95% CI: {mrr_ci.render()}")

        summary[name] = {"overall": overall.average(), "mrr_95ci": {"low": mrr_ci.low, "high": mrr_ci.high}}
        print()

    print("Paired bootstrap significance (MRR, same queries both sides):")
    significance: Dict[str, dict] = {}
    for a_name, b_name in SIGNIFICANCE_PAIRS:
        test = paired_bootstrap_test(mrr_per_query[a_name], mrr_per_query[b_name], seed=0)
        print(f"  {a_name} -> {b_name}: {test.render()}")
        significance[f"{a_name}_vs_{b_name}"] = {"mean_diff": test.mean_diff, "p_value": test.p_value}
    summary["_significance"] = significance
    summary["_meta"] = {"n_documents": len(corpus), "n_queries": len(cases), "relevance_threshold": RELEVANCE_THRESHOLD}

    embeddings.flush()  # catch whatever's accumulated since the last periodic flush
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
