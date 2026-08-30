"""Embedding cosine similarity — a small, dependency-free primitive.

Used by the robustness benchmark to compare answers across phrasings of the
same question, and available for a future reference-answer similarity
scorer (see TODO.md's toolkit-hardening list) without pulling in numpy for
what's a ~5-line computation.
"""

from __future__ import annotations

import math
from typing import Sequence


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the cosine similarity of two equal-length vectors.

    Returns 0.0 for a zero-magnitude vector rather than raising — a
    degenerate embedding shouldn't crash a benchmark run, it should just
    score as "not similar to anything."
    """

    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    denom = norm_a * norm_b
    return dot / denom if denom else 0.0


def average_pairwise_similarity(vectors: Sequence[Sequence[float]]) -> float:
    """Average cosine similarity over every unordered pair in ``vectors``.

    Returns 1.0 for zero or one vector (trivially "consistent" — there's
    nothing to disagree with), rather than raising or returning 0.0, so a
    caller can't misread "nothing to compare" as "totally inconsistent."
    """

    n = len(vectors)
    if n <= 1:
        return 1.0
    pairs = [
        cosine_similarity(vectors[i], vectors[j])
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return sum(pairs) / len(pairs)
