"""Bootstrap confidence intervals and paired significance testing.

Every benchmark in this toolkit reports a mean over some list of per-item
scores (``RetrievalReport.per_query``, a scorer's per-task results, ...) and,
until now, nothing beyond that mean -- a bare "0.830 vs 0.648" reads as a
real difference, but with n=24 that gap could just as easily be resampling
noise. These two functions turn a list of per-item values (or a list of
paired per-item deltas between two systems evaluated on the *same* items)
into a confidence interval / p-value, using nothing but the standard
library's ``random`` module -- no numpy/scipy, matching this toolkit's
zero-dependency core.

Both use the (nonparametric) bootstrap rather than a normal-theory interval
(t-test, z-test) because per-query metrics like Recall@k are bounded in
[0, 1] and often not close to normally distributed at small n -- resampling
the data itself sidesteps that assumption instead of relying on it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ConfidenceInterval:
    """A point estimate plus a bootstrap confidence interval around it."""

    mean: float
    low: float
    high: float
    n: int
    n_resamples: int
    confidence: float

    def render(self) -> str:
        return f"{self.mean:.3f} [{self.low:.3f}, {self.high:.3f}] (n={self.n})"


@dataclass(frozen=True)
class PairedTestResult:
    """Result of a paired bootstrap test on system_b - system_a per item."""

    mean_diff: float
    p_value: float
    n: int
    n_resamples: int

    def render(self) -> str:
        sig = "significant" if self.p_value < 0.05 else "not significant"
        return f"diff={self.mean_diff:+.3f}  p={self.p_value:.3f} ({sig} at alpha=0.05, n={self.n})"


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> ConfidenceInterval:
    """Return a bootstrap confidence interval for the mean of ``values``.

    Resamples ``values`` with replacement ``n_resamples`` times, takes the
    mean of each resample, and reports the percentile interval of that
    resampled-mean distribution -- the standard nonparametric bootstrap
    (Efron & Tibshirani). ``seed`` makes a run reproducible; pass ``None``
    for a fresh draw each call.

    Raises on fewer than 2 values -- a "confidence interval" around a
    single point is meaningless, and callers should surface that as a
    problem with the benchmark's n, not silently plot a zero-width interval.
    """

    if len(values) < 2:
        raise ValueError(f"Need at least 2 values for a bootstrap CI, got {len(values)}.")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")

    rng = random.Random(seed)
    n = len(values)
    resampled_means = []
    for _ in range(n_resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        resampled_means.append(sum(resample) / n)
    resampled_means.sort()

    tail = (1.0 - confidence) / 2.0
    low_idx = int(tail * n_resamples)
    high_idx = int((1.0 - tail) * n_resamples) - 1
    high_idx = min(high_idx, n_resamples - 1)

    return ConfidenceInterval(
        mean=sum(values) / n,
        low=resampled_means[low_idx],
        high=resampled_means[high_idx],
        n=n,
        n_resamples=n_resamples,
        confidence=confidence,
    )


def paired_bootstrap_test(
    system_a: Sequence[float],
    system_b: Sequence[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> PairedTestResult:
    """Test whether ``system_b``'s per-item scores differ from ``system_a``'s.

    ``system_a`` and ``system_b`` must be the *same length and item order*
    -- e.g. both systems' per-query Recall@5 on the identical query list, so
    element ``i`` of each is a matched pair, not an independent sample. This
    is what makes it a *paired* test: it resamples the per-item deltas
    (b - a), not the two systems' scores independently, which is the
    correct design for "two systems evaluated on the same benchmark" (a
    within-subjects comparison) rather than "two independent samples."

    The p-value is two-sided: the fraction of bootstrap resamples whose mean
    delta lands on the opposite side of zero from the observed mean delta,
    doubled (Davison & Hinkley's percentile approach) and capped at 1.0 --
    "how often would resampling alone flip the sign of the observed gap."
    """

    if len(system_a) != len(system_b):
        raise ValueError(
            f"system_a and system_b must be the same length (paired), got "
            f"{len(system_a)} vs {len(system_b)}."
        )
    n = len(system_a)
    if n < 2:
        raise ValueError(f"Need at least 2 paired items, got {n}.")

    deltas = [b - a for a, b in zip(system_a, system_b)]
    observed = sum(deltas) / n

    rng = random.Random(seed)
    resampled_means = []
    for _ in range(n_resamples):
        resample = [deltas[rng.randrange(n)] for _ in range(n)]
        resampled_means.append(sum(resample) / n)

    if observed >= 0:
        tail_count = sum(1 for m in resampled_means if m <= 0)
    else:
        tail_count = sum(1 for m in resampled_means if m >= 0)
    p_value = min(1.0, 2.0 * tail_count / n_resamples)

    return PairedTestResult(mean_diff=observed, p_value=p_value, n=n, n_resamples=n_resamples)
