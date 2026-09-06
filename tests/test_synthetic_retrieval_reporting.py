"""Exercise synthetic benchmark main without importing live setup or credentials."""

from __future__ import annotations

import ast
import json
import math
from collections import Counter
from pathlib import Path

import pytest

from agent_eval.retrieval_metrics import RetrievalCase, RetrievalReport, evaluate_retrieval
from agent_eval.stats import bootstrap_ci, paired_permutation_test


def _offline_functions():
    source = Path(__file__).resolve().parents[1] / "benchmarks/rag_recall/run_benchmark.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    names = {"finite_metric_by_query", "main"}
    nodes = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(
        body=[ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ), *nodes],
        type_ignores=[],
    )
    namespace = {
        "json": json, "math": math, "RetrievalCase": RetrievalCase,
        "RetrievalReport": RetrievalReport, "evaluate_retrieval": evaluate_retrieval,
        "bootstrap_ci": bootstrap_ci, "paired_permutation_test": paired_permutation_test,
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace


def test_finite_score_selection_excludes_null_failures_and_nan():
    namespace = _offline_functions()
    report = RetrievalReport("fake", (1,), per_query=[
        {"query_id": "ok", "status": "ok", "mrr": 0.5},
        {"query_id": "failed", "status": "failed", "mrr": None},
        {"query_id": "unjudged", "status": "not_applicable", "mrr": None},
        {"query_id": "invalid", "status": "ok", "mrr": float("nan")},
        {"query_id": "infinite", "status": "ok", "mrr": float("inf")},
    ])
    assert namespace["finite_metric_by_query"](report, "mrr") == {"ok": 0.5}


def test_finite_score_selection_rejects_duplicate_query_ids():
    namespace = _offline_functions()
    report = RetrievalReport("fake", (1,), per_query=[
        {"query_id": "duplicate", "status": "ok", "mrr": 1.0},
        {"query_id": "duplicate", "status": "failed", "mrr": None},
    ])
    with pytest.raises(ValueError, match="Duplicate query_id"):
        namespace["finite_metric_by_query"](report, "mrr")


def test_styles_reuse_one_observation_and_failed_queries_stay_accounted_for(tmp_path):
    namespace = _offline_functions()
    calls = Counter()
    paired_calls = []
    queries = [
        {"id": "one", "query": "q1", "fact": "target1", "style": "lexical"},
        {"id": "two", "query": "q2", "fact": "target2", "style": "lexical"},
        {"id": "three", "query": "q3", "fact": "target3", "style": "paraphrase"},
        {"id": "four", "query": "q4", "fact": "", "style": "paraphrase"},
    ]
    names = ("bm25_only", "dense_only", "hybrid_rrf", "hybrid_rerank")

    class Embeddings:
        model_id = "offline"

        def stats(self):
            return {"cached_vectors": 0}

        def flush(self):
            pass

    def make_retrieve_fn(name):
        def retrieve(query):
            calls[(name, query)] += 1
            if calls[(name, query)] > 1:
                raise AssertionError("Style reports must not rerun retrieval.")
            if (name, query) in {("bm25_only", "q2"), ("hybrid_rrf", "q3")}:
                raise RuntimeError("One arm failed this query.")
            return {
                "q1": ["target1"], "q2": ["noise", "target2"],
                "q3": ["noise", "noise2", "target3"], "q4": [],
            }[query]
        return retrieve

    def paired(left, right, **kwargs):
        paired_calls.append((left, right))
        return paired_permutation_test(left, right, **kwargs)

    output = tmp_path / "synthetic-results.json"
    namespace.update({
        "CASES": queries, "DOCUMENTS": [{"id": "synthetic"}],
        "K_VALUES": (1, 3),
        "SIGNIFICANCE_PAIRS": (
            ("bm25_only", "hybrid_rrf"), ("hybrid_rrf", "hybrid_rerank"),
        ),
        "RESULTS_PATH": output,
        "build_repository_and_retrievers": lambda: (None, None, None, Embeddings()),
        "check_one_chunk_per_section": lambda _: None,
        "resolve_relevant_ids": lambda fact, _: {fact} if fact else set(),
        "build_pipelines": lambda *_: {name: name for name in names},
        "make_retrieve_fn": make_retrieve_fn,
        "paired_permutation_test": paired,
    })
    namespace["main"]()

    assert len(calls) == len(names) * len(queries)
    assert set(calls.values()) == {1}
    saved = json.loads(output.read_text())
    bm25 = saved["bm25_only"]
    assert bm25["counts"]["total_queries"] == 4
    assert bm25["counts"]["failed_queries"] == 1
    assert bm25["counts"]["not_applicable_queries"] == 1
    assert bm25["counts"]["scored_queries"] == 2
    assert bm25["per_query"][1]["query_id"] == "two"
    assert bm25["per_query"][1]["status"] == "failed"
    assert bm25["per_query"][1]["mrr"] is None
    assert bm25["by_style"]["lexical"]["mrr"] == 1.0
    assert bm25["by_style_counts"]["lexical"]["failed_queries"] == 1
    assert bm25["by_style_counts"]["paraphrase"]["not_applicable_queries"] == 1
    assert bm25["mrr_95ci"]["n"] == 2
    assert "bootstrap" in bm25["mrr_95ci"]["method"]
    pair = saved["_significance"]["bm25_only_vs_hybrid_rrf"]
    assert pair["paired_query_ids"] == ["one"]
    assert pair["status"] == "not_applicable"
    assert paired_calls == [([1.0, 0.5], [1.0, 0.5])]
    assert saved["_meta"]["scoring_version"] == 2
    assert saved["_meta"]["level"] == "chunk"
    assert saved["_meta"]["official_document_qrels"] is False
