"""Score ranked retrieval IDs with explicit judgments and per-query coverage.

Binary sets remain supported; graded mappings use linear relevance gain for
nDCG, matching trec_eval's default. Undefined metrics stay visible as null
records instead of silently changing the evaluated query population.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Union

RetrieveFn = Callable[[str], List[str]]
Relevance = Union[Set[str], Mapping[str, float]]


@dataclass(frozen=True)
class RetrievalCase:
    """A query with binary IDs or document IDs mapped to relevance grades."""

    query: str
    relevant_ids: Relevance
    query_id: str = ""


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")


def _grades(relevant: Relevance) -> Dict[str, float]:
    values = dict(relevant) if isinstance(relevant, Mapping) else {
        item: 1.0 for item in relevant
    }
    normalized: Dict[str, float] = {}
    for item, score in values.items():
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"Relevance for {item!r} must be a finite nonnegative number.")
        try:
            value = float(score)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"Relevance for {item!r} must be finite.") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Relevance for {item!r} must be a finite nonnegative number.")
        normalized[item] = value
    return normalized


def unique_ranked_ids(retrieved: Sequence[str]) -> List[str]:
    """Keep only each ID's first occurrence, preserving ranked order."""

    return list(dict.fromkeys(retrieved))


def recall_at_k(retrieved: Sequence[str], relevant: Relevance, k: int) -> Optional[float]:
    """Recall over positively judged IDs, or None when it is undefined."""

    _validate_k(k)
    positive = {item for item, score in _grades(relevant).items() if score > 0}
    if not positive:
        return None
    return len(set(unique_ranked_ids(retrieved)[:k]) & positive) / len(positive)


def mrr(retrieved: Sequence[str], relevant: Relevance) -> Optional[float]:
    """Reciprocal rank on unique IDs; None means there are no positive qrels."""

    positive = {item for item, score in _grades(relevant).items() if score > 0}
    if not positive:
        return None
    for rank, item in enumerate(unique_ranked_ids(retrieved), start=1):
        if item in positive:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Relevance, k: int) -> Optional[float]:
    """Graded nDCG@k using relevance itself as gain (trec_eval convention).

    See https://github.com/usnistgov/trec_eval/blob/master/m_ndcg.c.
    Every document contributes once. Scaling all gains by their maximum
    preserves the ratio and avoids overflow for large finite grades.
    """

    _validate_k(k)
    grades = _grades(relevant)
    maximum = max(grades.values(), default=0.0)
    if maximum == 0:
        return None
    gains = {item: score / maximum for item, score in grades.items() if score > 0}
    ideal = math.fsum(
        score / math.log2(rank + 1)
        for rank, score in enumerate(sorted(gains.values(), reverse=True)[:k], start=1)
    )
    actual = math.fsum(
        gains.get(item, 0.0) / math.log2(rank + 1)
        for rank, item in enumerate(unique_ranked_ids(retrieved)[:k], start=1)
    )
    return min(1.0, actual / ideal)


@dataclass
class RetrievalReport:
    """Scores plus explicit query-level applicability and retrieval counts."""

    name: str
    k_values: Sequence[int]
    per_query: List[Dict[str, Any]] = field(default_factory=list)

    def metric_keys(self) -> List[str]:
        keys = ["mrr"]
        for k in self.k_values:
            _validate_k(k)
            keys.extend((f"recall@{k}", f"ndcg@{k}"))
        return list(dict.fromkeys(keys))

    def average(self) -> Dict[str, float]:
        """Average metric columns only, leaving metadata and counts separate."""

        rows = [row for row in self.per_query if row.get("status", "ok") == "ok"]
        if not rows:
            return {}
        averaged: Dict[str, float] = {}
        for key in self.metric_keys():
            values = [row[key] for row in rows]
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1
                for value in values
            ):
                raise ValueError(f"Scored query contains invalid metric {key!r}.")
            averaged[key] = math.fsum(values) / len(values)
        return averaged

    def counts(self) -> Dict[str, int]:
        scored = sum(row.get("status", "ok") == "ok" for row in self.per_query)
        return {
            "total_queries": len(self.per_query),
            "scored_queries": scored,
            "not_applicable_queries": sum(
                row.get("status") == "not_applicable" for row in self.per_query
            ),
            "failed_queries": sum(row.get("status") == "failed" for row in self.per_query),
            "unjudged_queries": sum(
                row.get("reason") == "no_judgments" for row in self.per_query
            ),
            "no_positive_judgments_queries": sum(
                row.get("reason") == "no_positive_judgments" for row in self.per_query
            ),
        }

    def render(self) -> str:
        cells = "  ".join(f"{key}={value:.3f}" for key, value in self.average().items())
        counts = self.counts()
        return (
            f"{self.name}: {cells} "
            f"(scored={counts['scored_queries']}/{counts['total_queries']}, "
            f"not_applicable={counts['not_applicable_queries']}, "
            f"failed={counts['failed_queries']})"
        )


def evaluate_retrieval(
    cases: Sequence[RetrievalCase],
    retrieve: RetrieveFn,
    *,
    name: str = "retriever",
    k_values: Sequence[int] = (5, 10),
) -> RetrievalReport:
    """Retain every query, including undefined judgments and retrieval failures.

    average() reports quality among successfully retrieved labeled queries.
    counts() separately exposes failures, which are never relabeled unjudged.
    Invalid metric configuration or qrels fails before retrieval begins.
    """

    for k in k_values:
        _validate_k(k)
    ks = tuple(dict.fromkeys(k_values))
    per_query: List[Dict[str, Any]] = []
    prepared = [(case, _grades(case.relevant_ids)) for case in cases]
    ids = [case.query_id or f"query-{index + 1}" for index, (case, _) in enumerate(prepared)]
    if any(not isinstance(query_id, str) for query_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("Query IDs must be unique strings, including generated IDs.")
    for index, (case, grades) in enumerate(prepared):
        relevant_count = sum(score > 0 for score in grades.values())
        reason = None if relevant_count else (
            "no_positive_judgments" if grades else "no_judgments"
        )
        row: Dict[str, Any] = {
            "query_id": case.query_id or f"query-{index + 1}",
            "query": case.query,
            "status": "ok" if relevant_count else "not_applicable",
            "reason": reason,
            "judgment_count": len(grades),
            "relevant_count": relevant_count,
            "retrieved_count": None,
            "unique_retrieved_count": None,
            "duplicate_count": None,
            "mrr": None,
        }
        for k in ks:
            row[f"recall@{k}"] = None
            row[f"ndcg@{k}"] = None
        try:
            ranked = list(retrieve(case.query))
            unique = unique_ranked_ids(ranked)
        except Exception as exc:
            row.update(
                status="failed", reason="retrieval_error",
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            per_query.append(row)
            continue
        row.update(
            retrieved_count=len(ranked), unique_retrieved_count=len(unique),
            duplicate_count=len(ranked) - len(unique),
            mrr=mrr(unique, grades),
        )
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(unique, grades, k)
            row[f"ndcg@{k}"] = ndcg_at_k(unique, grades, k)
        per_query.append(row)
    return RetrievalReport(name=name, k_values=ks, per_query=per_query)
