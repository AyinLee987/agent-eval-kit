"""Retrieval regressions for duplicate IDs, graded judgments and query accounting."""

from __future__ import annotations

import math

import pytest

from agent_eval.retrieval_metrics import (
    RetrievalCase, RetrievalReport, evaluate_retrieval, mrr, ndcg_at_k, recall_at_k,
)


def test_recall_at_k_counts_overlap_within_top_k():
    assert recall_at_k(["a", "b", "c", "d"], {"c", "z"}, k=3) == 0.5
    assert recall_at_k(["a", "b", "c", "d"], {"c", "z"}, k=1) == 0.0


def test_no_positive_judgments_make_metrics_undefined_instead_of_zero():
    for judgments in (set(), {"a": 0}):
        assert recall_at_k(["a"], judgments, k=5) is None
        assert ndcg_at_k(["a"], judgments, k=5) is None
        assert mrr(["a"], judgments) is None


def test_mrr_rewards_earlier_unique_hits():
    assert mrr(["x", "a", "b"], {"a"}) == 0.5
    assert mrr(["a", "x"], {"a"}) == 1.0
    assert mrr(["x", "y"], {"a"}) == 0.0
    assert mrr(["x", "x", "a"], {"a"}) == 0.5


def test_ndcg_at_k_is_perfect_when_all_relevant_ids_rank_first():
    assert ndcg_at_k(["a", "b", "z"], {"a", "b"}, k=3) == 1.0


def test_ndcg_at_k_penalizes_relevant_ids_ranked_lower():
    assert 0.0 < ndcg_at_k(["z", "a", "b"], {"a", "b"}, k=3) < 1.0


def test_repeated_relevant_ids_cannot_inflate_ndcg_above_one():
    assert ndcg_at_k(["a", "a", "a"], {"a"}, 3) == 1.0
    assert recall_at_k(["a", "a", "b"], {"a", "b"}, 2) == 1.0
    assert ndcg_at_k(["a", "a", "b"], {"a", "b"}, 2) == 1.0


def test_ndcg_retains_graded_relevance_with_trec_eval_linear_gain():
    qrels = {"high": 2, "partial": 1, "irrelevant": 0}
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert ndcg_at_k(["partial", "high"], qrels, 2) == pytest.approx(expected)
    assert ndcg_at_k(["high", "partial"], qrels, 2) == 1.0
    assert recall_at_k(["irrelevant", "partial"], qrels, 2) == 0.5


def test_graded_ndcg_remains_finite_for_large_finite_gains():
    value = ndcg_at_k(["a", "b"], {"a": 1e308, "b": 1e308}, 2)
    assert value == 1.0


@pytest.mark.parametrize("k", [0, -1, 1.5, "2", True, False, None])
def test_k_must_be_a_positive_integer_even_without_qrels(k):
    with pytest.raises(ValueError, match="positive integer"):
        recall_at_k([], set(), k)
    with pytest.raises(ValueError, match="positive integer"):
        ndcg_at_k([], set(), k)
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_retrieval([], lambda _: [], k_values=(k,))


@pytest.mark.parametrize("score", [-1, float("nan"), float("inf"), True, "1"])
def test_invalid_relevance_grades_are_rejected(score):
    for metric in (
        lambda: recall_at_k(["a"], {"a": score}, 1),
        lambda: ndcg_at_k(["a"], {"a": score}, 1),
        lambda: mrr(["a"], {"a": score}),
    ):
        with pytest.raises(ValueError, match="Relevance"):
            metric()


def test_report_retains_unjudged_queries_without_averaging_metadata():
    called = []
    rankings = {"q1": ["doc-1", "doc-1"], "q2": ["doc-9", "doc-4"]}
    cases = [
        RetrievalCase(query="q1", relevant_ids={"doc-1"}, query_id="one"),
        RetrievalCase(query="q2", relevant_ids={"doc-4"}, query_id="two"),
        RetrievalCase(query="unjudged", relevant_ids=set(), query_id="three"),
        RetrievalCase(query="zero-grade", relevant_ids={"doc-1": 0}, query_id="four"),
    ]

    def retrieve(query):
        called.append(query)
        return rankings.get(query, [])

    report = evaluate_retrieval(cases, retrieve, k_values=(1, 3))
    assert len(called) == len(report.per_query) == 4
    assert [row["query_id"] for row in report.per_query] == ["one", "two", "three", "four"]
    assert report.per_query[0]["duplicate_count"] == 1
    assert report.per_query[0]["retrieved_count"] == 2
    assert report.per_query[0]["unique_retrieved_count"] == 1
    assert report.per_query[2]["status"] == "not_applicable"
    assert report.per_query[2]["recall@1"] is None
    assert report.per_query[2]["reason"] == "no_judgments"
    assert report.per_query[3]["reason"] == "no_positive_judgments"
    assert report.counts() == {
        "total_queries": 4, "scored_queries": 2, "not_applicable_queries": 2,
        "failed_queries": 0,
        "unjudged_queries": 1, "no_positive_judgments_queries": 1,
    }
    avg = report.average()
    assert avg["recall@1"] == 0.5
    assert avg["recall@3"] == 1.0
    assert set(avg) == {"mrr", "recall@1", "ndcg@1", "recall@3", "ndcg@3"}
    assert "scored=2/4" in report.render()


def test_relevant_documents_missing_from_the_ranked_list_stay_in_denominator():
    report = evaluate_retrieval(
        [RetrievalCase("q", {"available": 2, "missing": 1}, query_id="q")],
        lambda _: ["available"], k_values=(5,),
    )
    assert report.per_query[0]["relevant_count"] == 2
    assert report.average()["recall@5"] == 0.5


def test_reports_with_only_undefined_queries_have_explicit_counts_and_no_average():
    report = evaluate_retrieval([RetrievalCase("q", set())], lambda _: ["x"])
    assert report.average() == {}
    assert report.counts()["not_applicable_queries"] == 1
    assert report.per_query[0]["query_id"] == "query-1"


def test_direct_reports_do_not_silently_average_invalid_scores():
    report = RetrievalReport("bad", (1,), per_query=[
        {"mrr": float("nan"), "recall@1": 1.0, "ndcg@1": 1.0},
    ])
    with pytest.raises(ValueError, match="invalid metric"):
        report.average()


def test_a_failed_retrieval_keeps_error_metadata_and_does_not_abort_later_queries():
    calls = []
    cases = [
        RetrievalCase("bad", {"target"}, query_id="failed"),
        RetrievalCase("good", {"target"}, query_id="succeeded"),
        RetrievalCase("unjudged", set(), query_id="unjudged"),
    ]

    def retrieve(query):
        calls.append(query)
        if query == "bad":
            raise RuntimeError("retriever unavailable")
        return ["target"]

    report = evaluate_retrieval(cases, retrieve, k_values=(1,))
    assert calls == ["bad", "good", "unjudged"]
    failed = report.per_query[0]
    assert failed["status"] == "failed"
    assert failed["reason"] == "retrieval_error"
    assert failed["error"] == {"type": "RuntimeError", "message": "retriever unavailable"}
    assert failed["mrr"] is None
    assert failed["retrieved_count"] is None
    assert failed["relevant_count"] == 1
    assert report.average()["mrr"] == 1.0
    assert report.counts()["scored_queries"] == 1
    assert report.counts()["failed_queries"] == 1
    assert report.counts()["not_applicable_queries"] == 1
    assert report.counts()["unjudged_queries"] == 1
