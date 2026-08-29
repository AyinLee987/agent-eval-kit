"""Retrieval quality metrics, decoupled from any specific retriever.

These operate on plain ranked-id lists, so they work against a hybrid RAG
pipeline, a single BM25 index, a vector store, or anything else that can
answer "given this query, what ids did you rank first" — that's the whole
interface (:data:`RetrieveFn`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Set

# Given a query string, return a ranked list of ids (best first).
RetrieveFn = Callable[[str], List[str]]


@dataclass(frozen=True)
class RetrievalCase:
    """One labeled query: the ids that count as relevant to it."""

    query: str
    relevant_ids: Set[str]


def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """Fraction of relevant ids present in the top-k retrieved ids.

    0.0 for a case with no relevant ids marked (undefined recall), so
    callers should filter those out before averaging rather than let a
    silent 0.0 skew the mean.
    """

    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def mrr(retrieved: Sequence[str], relevant: Set[str]) -> float:
    """Reciprocal rank of the first relevant id, 0.0 if none appear."""

    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """Binary-relevance nDCG@k: 1.0 if relevant, 0.0 otherwise, per position."""

    def dcg(ids: Sequence[str]) -> float:
        return sum(
            (1.0 if item in relevant else 0.0) / math.log2(rank + 1)
            for rank, item in enumerate(ids[:k], start=1)
        )

    ideal = dcg(list(relevant)[:k])
    if ideal == 0.0:
        return 0.0
    return dcg(retrieved) / ideal


@dataclass
class RetrievalReport:
    """Per-query and averaged retrieval metrics for one retriever."""

    name: str
    k_values: Sequence[int]
    per_query: List[Dict[str, float]] = field(default_factory=list)

    def average(self) -> Dict[str, float]:
        if not self.per_query:
            return {}
        keys = self.per_query[0].keys()
        return {key: sum(row[key] for row in self.per_query) / len(self.per_query) for key in keys}

    def render(self) -> str:
        avg = self.average()
        cells = "  ".join(f"{key}={value:.3f}" for key, value in avg.items())
        return f"{self.name}: {cells} (n={len(self.per_query)})"


def evaluate_retrieval(
    cases: Sequence[RetrievalCase],
    retrieve: RetrieveFn,
    *,
    name: str = "retriever",
    k_values: Sequence[int] = (5, 10),
) -> RetrievalReport:
    """Run ``retrieve`` over every case and average Recall@k / MRR / nDCG@k.

    Cases with no relevant ids are skipped — Recall/nDCG are undefined for
    them and would silently drag the average toward zero.
    """

    per_query: List[Dict[str, float]] = []
    for case in cases:
        if not case.relevant_ids:
            continue
        ranked = retrieve(case.query)
        row: Dict[str, float] = {"mrr": mrr(ranked, case.relevant_ids)}
        for k in k_values:
            row[f"recall@{k}"] = recall_at_k(ranked, case.relevant_ids, k)
            row[f"ndcg@{k}"] = ndcg_at_k(ranked, case.relevant_ids, k)
        per_query.append(row)

    return RetrievalReport(name=name, k_values=k_values, per_query=per_query)
