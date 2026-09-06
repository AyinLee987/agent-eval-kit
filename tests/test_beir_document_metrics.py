"""Offline tests for BEIR document identity and judgment coverage."""

from __future__ import annotations

import importlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from agent_eval.retrieval_metrics import evaluate_retrieval
from benchmarks.rag_recall_beir.run_benchmark import (
    build_cases, build_chunk_to_document, document_coverage,
    make_retrieve_fn, rank_document_ids,
)


def _fixture():
    corpus = [{"id": "d1"}, {"id": "d2"}, {"id": "d3"}]
    queries = [
        {"id": "q1", "text": "partially covered"},
        {"id": "q2", "text": "only missing relevant document"},
        {"id": "q3", "text": "no positive judgments"},
        {"id": "q4", "text": "no judgments"},
    ]
    qrels = [
        {"query_id": "q1", "corpus_id": "d1", "score": 2},
        {"query_id": "q1", "corpus_id": "d2", "score": 1},
        {"query_id": "q2", "corpus_id": "d2", "score": 2},
        {"query_id": "q3", "corpus_id": "d1", "score": 0},
    ]
    mapping = {"d1": {"chunk-1", "chunk-2"}, "d2": set(), "d3": {"chunk-3"}}
    return corpus, queries, qrels, mapping


def test_build_cases_keeps_original_document_grades_and_all_queries():
    _, queries, qrels, mapping = _fixture()
    cases = build_cases(queries, qrels, mapping)
    assert len(cases) == len(queries)
    assert cases[0].relevant_ids == {"d1": 2, "d2": 1}
    assert cases[1].relevant_ids == {"d2": 2}
    assert cases[2].relevant_ids == {"d1": 0}
    assert cases[3].relevant_ids == {}
    assert [case.query_id for case in cases] == ["q1", "q2", "q3", "q4"]
    assert build_cases(queries, qrels, {}) == cases


def test_document_rank_collapses_repeated_chunks_in_first_occurrence_order():
    mapping = build_chunk_to_document({"d1": {"c1", "c2"}, "d2": {"c3"}})
    assert rank_document_ids(["c2", "c1", "c3", "c2"], mapping) == ["d1", "d2"]
    assert rank_document_ids(["c3", "c1"], mapping) == ["d2", "d1"]


def test_ambiguous_or_missing_chunk_mapping_fails_loudly():
    with pytest.raises(ValueError, match="multiple documents"):
        build_chunk_to_document({"d1": {"same"}, "d2": {"same"}})
    with pytest.raises(ValueError, match="no document mapping"):
        rank_document_ids(["unknown"], {})


def test_make_retrieve_fn_scores_documents_instead_of_fabricated_chunk_qrels():
    class Pipeline:
        def retrieve(self, query):
            return SimpleNamespace(evidence=[
                SimpleNamespace(chunk=SimpleNamespace(id=chunk_id))
                for chunk_id in ["chunk-2", "chunk-1", "chunk-3"]
            ])

    _, queries, qrels, mapping = _fixture()
    cases = build_cases(queries, qrels, mapping)
    retrieve = make_retrieve_fn(Pipeline(), mapping)
    assert retrieve("q") == ["d1", "d3"]
    report = evaluate_retrieval(cases, retrieve, k_values=(3,))
    assert report.per_query[0]["recall@3"] == 0.5
    assert report.per_query[1]["recall@3"] == 0.0
    assert report.per_query[1]["status"] == "ok"
    assert report.per_query[2]["status"] == "not_applicable"
    assert report.counts()["total_queries"] == 4
    assert report.counts()["scored_queries"] == 2


def test_coverage_exposes_missing_docs_and_affected_queries_without_changing_scores():
    corpus, queries, qrels, mapping = _fixture()
    cases = build_cases(queries, qrels, mapping)
    coverage = document_coverage(corpus, cases, mapping)
    assert coverage["corpus_document_count"] == 3
    assert coverage["indexed_corpus_document_count"] == 2
    assert coverage["missing_corpus_document_ids"] == ["d2"]
    assert coverage["relevant_document_coverage"] == 0.5
    assert coverage["positive_judgment_count"] == 3
    assert coverage["indexed_positive_judgment_count"] == 1
    assert coverage["queries_without_retrievable_relevant_documents"] == ["q2"]
    assert coverage["per_query"][0]["relevant_document_coverage"] == 0.5
    assert coverage["per_query"][1]["relevant_document_coverage"] == 0.0
    assert coverage["per_query"][2]["relevant_document_coverage"] is None


def test_conflicting_qrels_or_duplicate_query_identity_are_rejected():
    with pytest.raises(ValueError, match="Conflicting qrels"):
        build_cases([{"id": "q", "text": "query"}], [
            {"query_id": "q", "corpus_id": "d", "score": 1},
            {"query_id": "q", "corpus_id": "d", "score": 2},
        ])
    with pytest.raises(ValueError, match="Duplicate query_id"):
        build_cases([{"id": "q", "text": "a"}, {"id": "q", "text": "b"}], [])


@pytest.mark.parametrize("name", [
    "benchmarks.rag_recall_beir.run_benchmark",
    "benchmarks.rag_recall_beir.run_benchmark_llm_rerank",
])
def test_importing_benchmarks_does_not_load_env_or_import_live_agent(name, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Importing a benchmark must not load credentials.")

    dotenv = ModuleType("dotenv")
    dotenv.load_dotenv = forbidden
    monkeypatch.setitem(sys.modules, "dotenv", dotenv)
    monkeypatch.setitem(sys.modules, "agent", None)
    path_before = list(sys.path)
    module = importlib.import_module(name)
    importlib.reload(module)
    assert sys.path == path_before


@pytest.mark.parametrize("failing_arm", [None, "bm25_only"])
def test_offline_benchmark_output_includes_every_query_and_coverage(tmp_path, monkeypatch, failing_arm):
    from benchmarks.rag_recall_beir import run_benchmark as benchmark

    corpus, queries, qrels, mapping = _fixture()

    class Embeddings:
        model_id = "offline"

        def stats(self):
            return {"cached_vectors": 0}

        def flush(self):
            pass

    class Pipeline:
        def __init__(self, name):
            self.name = name

        def retrieve(self, query):
            if self.name == failing_arm and query == queries[0]["text"]:
                raise RuntimeError("one retriever failed this query")
            return SimpleNamespace(evidence=[
                SimpleNamespace(chunk=SimpleNamespace(id="chunk-1")),
            ])

    output = tmp_path / "offline-results.json"
    monkeypatch.setattr(benchmark, "RESULTS_PATH", output)
    monkeypatch.setattr(benchmark, "initialize_environment", lambda: None)
    monkeypatch.setattr(benchmark, "load_json", lambda name: {
        "corpus.json": corpus, "queries.json": queries, "qrels.json": qrels,
    }[name])
    monkeypatch.setattr(
        benchmark, "build_repository_and_retrievers",
        lambda: (None, None, None, None, Embeddings()),
    )
    monkeypatch.setattr(benchmark, "ingest_corpus", lambda *_: mapping)
    monkeypatch.setattr(benchmark, "build_pipelines", lambda *_: {
        name: Pipeline(name) for name in ("bm25_only", "dense_only", "hybrid_rrf", "hybrid_rerank")
    })
    benchmark.main()
    data = json.loads(output.read_text())
    assert data["_meta"]["judgment_level"] == "document"
    assert data["_meta"]["coverage"]["missing_corpus_document_ids"] == ["d2"]
    assert data["bm25_only"]["counts"]["total_queries"] == 4
    assert len(data["bm25_only"]["per_query"]) == 4
    assert data["bm25_only"]["per_query"][2]["mrr"] is None
    assert data["_meta"]["scoring_version"] == 2
    assert data["_meta"]["level"] == "document"
    pair = data["_significance"]["bm25_only_vs_hybrid_rrf"]
    if failing_arm:
        assert data["bm25_only"]["counts"]["failed_queries"] == 1
        assert pair["paired_query_ids"] == ["q2"]
        assert pair["paired_count"] == 1
        assert pair["status"] == "not_applicable"
    else:
        assert pair["paired_query_ids"] == ["q1", "q2"]
        assert pair["method"] == "paired_permutation"
