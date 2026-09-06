"""Choose uncertainty estimates that match the benchmark's sampling unit.

Continuous per-task scores use a percentile bootstrap interval. Bernoulli
outcomes use Wilson intervals so an all-pass sample still has uncertainty.
Paired comparisons exchange condition labels under the null hypothesis;
resampling the observed differences does not itself generate that null.
All implementations use only the standard library.
"""

from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass
from numbers import Integral, Real
from statistics import NormalDist
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class ConfidenceInterval:
    """A mean and interval, with the method explicit in saved/rendered results."""

    mean: float
    low: float
    high: float
    n: int
    n_resamples: int
    confidence: float
    method: str = "percentile_bootstrap"

    def render(self) -> str:
        return (
            f"{self.mean:.3f} [{self.low:.3f}, {self.high:.3f}] "
            f"(n={self.n}, {self.confidence:.0%} CI, method={self.method})"
        )


@dataclass(frozen=True)
class PairedTestResult:
    """Two-sided test of system_b - system_a with paired labels exchanged.

    n_resamples counts the label assignments actually examined; exact tests
    can enumerate fewer assignments than the requested Monte Carlo budget.
    """

    mean_diff: float
    p_value: float
    n: int
    n_resamples: int
    method: str = "paired_permutation"

    def render(self) -> str:
        sig = "significant" if self.p_value < 0.05 else "not significant"
        return (
            f"diff={self.mean_diff:+.3f}  p={self.p_value:.3g} "
            f"({sig} at alpha=0.05, n={self.n}, method={self.method})"
        )


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def _confidence(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not 0.0 < value < 1.0
        or not math.isfinite(value)
    ):
        raise ValueError(f"confidence must be finite and in (0, 1), got {value!r}.")
    return float(value)


def _finite_values(values: Sequence[float], name: str) -> List[float]:
    converted = []
    for index, value in enumerate(values):
        if not isinstance(value, Real):
            raise ValueError(f"{name}[{index}] must be a finite real number.")
        try:
            number = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{name}[{index}] must be a finite real number.") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}] must be a finite real number.")
        converted.append(number)
    return converted


def _mean(values: Sequence[float]) -> float:
    # Divide before summing so a finite mean does not require an overflowing sum.
    return math.fsum(value / len(values) for value in values)


def _percentile(ordered: Sequence[float], probability: float) -> float:
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: Optional[int] = 0,
) -> ConfidenceInterval:
    """Percentile bootstrap CI for the mean of finite per-item scores.

    Items must be independent sampling units. Average repeated trials within
    each task first when task is the unit of inference. This resamples those
    task scores, not separate trials or correlated calls within one batch.

    Quantiles use linear interpolation. A constant sample has a zero-width
    percentile interval; use wilson_ci for binary success counts, where that
    behavior would conceal uncertainty. seed=None requests a fresh random draw.
    """

    n_resamples = _positive_integer(n_resamples, "n_resamples")
    confidence = _confidence(confidence)
    scores = _finite_values(values, "values")
    n = len(scores)
    if n < 2:
        raise ValueError(f"Need at least 2 values for a bootstrap CI, got {n}.")

    rng = random.Random(seed)
    resampled_means = sorted(
        _mean([scores[rng.randrange(n)] for _ in range(n)])
        for _ in range(n_resamples)
    )
    tail = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        mean=_mean(scores),
        low=_percentile(resampled_means, tail),
        high=_percentile(resampled_means, 1.0 - tail),
        n=n,
        n_resamples=n_resamples,
        confidence=confidence,
    )


def wilson_ci(
    successes: int,
    n: int,
    *,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Wilson score interval for independent Bernoulli outcomes.

    successes and n are counts, not rates or averaged repeated-trial scores.
    n=1 is supported; n=0 has no estimate. The result uses n_resamples=0 because
    this interval is analytic. It does not assume zero uncertainty at 0/n or n/n.
    """

    n = _positive_integer(n, "n")
    if (
        isinstance(successes, bool)
        or not isinstance(successes, Integral)
        or not 0 <= successes <= n
    ):
        raise ValueError("successes must be an integer between 0 and n.")
    confidence = _confidence(confidence)
    # Use the lower tail to avoid rounding (1 + confidence) / 2 up to 1.
    z = -NormalDist().inv_cdf((1.0 - confidence) / 2.0)
    proportion = successes / n
    z_squared_over_n = z * z / n
    denominator = 1.0 + z_squared_over_n
    center = (proportion + z_squared_over_n / 2.0) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / n + z_squared_over_n / (4.0 * n)
    ) / denominator
    return ConfidenceInterval(
        mean=proportion,
        low=0.0 if successes == 0 else max(0.0, center - margin),
        high=1.0 if successes == n else min(1.0, center + margin),
        n=n,
        n_resamples=0,
        confidence=confidence,
        method="wilson",
    )


def paired_permutation_test(
    system_a: Sequence[float],
    system_b: Sequence[float],
    *,
    n_resamples: int = 10_000,
    seed: Optional[int] = 0,
) -> PairedTestResult:
    """Test a mean paired difference by swapping condition labels within pairs.

    Inputs must have identical item order and length. Under the null, A/B
    labels must be exchangeable within each independent pair (for example,
    under randomized paired assignment). Pairing alone does not establish this:
    fixed condition order, provider drift, or dependent repeated trials can
    violate it. For repeated benchmark trials, use task-level paired scores.

    The two-sided statistic is abs(mean(B - A)). All 2**n label assignments
    are enumerated when they fit n_resamples; otherwise independent random
    sign flips use (extreme_count + 1) / (n_resamples + 1), including the
    observed assignment, so a Monte Carlo p-value cannot be zero. This absolute
    statistic convention is explicit; other two-sided conventions exist.
    """

    n_resamples = _positive_integer(n_resamples, "n_resamples")
    left = _finite_values(system_a, "system_a")
    right = _finite_values(system_b, "system_b")
    if len(left) != len(right):
        raise ValueError(
            f"system_a and system_b must be the same length (paired), got "
            f"{len(left)} vs {len(right)}."
        )
    n = len(left)
    if n < 2:
        raise ValueError(f"Need at least 2 paired items, got {n}.")

    deltas = [b - a for a, b in zip(left, right)]
    if not all(math.isfinite(delta) for delta in deltas):
        raise ValueError("Paired differences must remain finite.")
    scaled = [delta / n for delta in deltas]
    observed = math.fsum(scaled)
    threshold = abs(observed) * (1.0 - 1e-14)
    assignments = 1 << n

    if assignments <= n_resamples:
        extreme = sum(
            abs(math.fsum(
                value if mask & (1 << index) else -value
                for index, value in enumerate(scaled)
            )) >= threshold
            for mask in range(assignments)
        )
        p_value = extreme / assignments
        examined = assignments
        method = "paired_permutation_exact"
    else:
        rng = random.Random(seed)
        extreme = sum(
            abs(math.fsum(
                value if rng.getrandbits(1) else -value for value in scaled
            )) >= threshold
            for _ in range(n_resamples)
        )
        p_value = (extreme + 1) / (n_resamples + 1)
        examined = n_resamples
        method = "paired_permutation_monte_carlo"

    return PairedTestResult(
        mean_diff=observed,
        p_value=p_value,
        n=n,
        n_resamples=examined,
        method=method,
    )


def paired_bootstrap_test(
    system_a: Sequence[float],
    system_b: Sequence[float],
    *,
    n_resamples: int = 10_000,
    seed: Optional[int] = 0,
) -> PairedTestResult:
    """Deprecated compatibility name for paired_permutation_test.

    The former bootstrap sign-tail calculation was not a null-label test and
    could declare two positive pairs significant. Calls now use the paired
    permutation test and its exchangeability assumption; saved results record
    the actual method. New callers should use paired_permutation_test directly.
    """

    warnings.warn(
        "paired_bootstrap_test now performs a paired permutation test; "
        "use paired_permutation_test and check its label-exchangeability assumption.",
        DeprecationWarning,
        stacklevel=2,
    )
    return paired_permutation_test(
        system_a, system_b, n_resamples=n_resamples, seed=seed
    )
