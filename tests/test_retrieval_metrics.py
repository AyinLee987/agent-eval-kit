from agent_eval.retrieval_metrics import (
    RetrievalCase,
    evaluate_retrieval,
    mrr,
    ndcg_at_k,
    recall_at_k,
)


def test_recall_at_k_counts_overlap_within_top_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"c", "z"}
    assert recall_at_k(retrieved, relevant, k=3) == 0.5  # only "c" is in top-3
    assert recall_at_k(retrieved, relevant, k=1) == 0.0


def test_recall_at_k_is_zero_when_no_relevant_ids_are_marked():
    assert recall_at_k(["a", "b"], set(), k=5) == 0.0


def test_mrr_rewards_earlier_hits():
    assert mrr(["x", "a", "b"], {"a"}) == 0.5
    assert mrr(["a", "x"], {"a"}) == 1.0
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_ndcg_at_k_is_perfect_when_all_relevant_ids_rank_first():
    retrieved = ["a", "b", "z"]
    relevant = {"a", "b"}
    assert ndcg_at_k(retrieved, relevant, k=3) == 1.0


def test_ndcg_at_k_penalizes_relevant_ids_ranked_lower():
    retrieved = ["z", "a", "b"]
    relevant = {"a", "b"}
    assert 0.0 < ndcg_at_k(retrieved, relevant, k=3) < 1.0


def test_evaluate_retrieval_averages_across_cases_and_skips_unlabeled_ones():
    corpus = {
        "q1": ["doc-1", "doc-2", "doc-3"],
        "q2": ["doc-9", "doc-4", "doc-5"],
    }
    cases = [
        RetrievalCase(query="q1", relevant_ids={"doc-1"}),
        RetrievalCase(query="q2", relevant_ids={"doc-4"}),
        RetrievalCase(query="unlabeled", relevant_ids=set()),
    ]

    report = evaluate_retrieval(cases, retrieve=lambda q: corpus.get(q, []), k_values=(1, 3))

    assert len(report.per_query) == 2  # the unlabeled case is skipped
    avg = report.average()
    assert avg["recall@1"] == 0.5  # q1 hits at rank 1, q2 does not
    assert avg["recall@3"] == 1.0  # both hit within top-3
