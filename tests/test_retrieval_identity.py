import pytest

from agent_eval.retrieval_metrics import RetrievalCase, evaluate_retrieval


@pytest.mark.parametrize("ids", [("q", "q"), ("", "query-1"), (1, "q")])
def test_duplicate_or_invalid_retrieval_identity_is_rejected_before_execution(ids):
    cases = [RetrievalCase("query", {"doc"}, query_id=query_id) for query_id in ids]
    def retrieve(query):
        pytest.fail("Invalid case identities must be rejected before retrieval.")
    with pytest.raises(ValueError, match="Query IDs"):
        evaluate_retrieval(cases, retrieve)
