"""Tests for the bootstrap CI / paired significance-test primitives."""

from __future__ import annotations

import pytest

from agent_eval import bootstrap_ci, paired_bootstrap_test


def test_bootstrap_ci_centers_on_the_sample_mean():
    values = [0.0, 0.5, 1.0, 0.5, 1.0, 0.0, 0.5]
    result = bootstrap_ci(values, seed=1)
    assert result.mean == pytest.approx(sum(values) / len(values))
    assert result.low <= result.mean <= result.high
    assert result.n == len(values)


def test_bootstrap_ci_is_narrower_with_more_consistent_data():
    consistent = [0.8] * 20
    noisy = [0.0, 1.0] * 10
    tight = bootstrap_ci(consistent, seed=1)
    wide = bootstrap_ci(noisy, seed=1)
    assert (tight.high - tight.low) < (wide.high - wide.low)


def test_bootstrap_ci_is_reproducible_with_a_fixed_seed():
    values = [0.1, 0.9, 0.4, 0.6, 0.7, 0.2]
    a = bootstrap_ci(values, seed=42)
    b = bootstrap_ci(values, seed=42)
    assert a == b


def test_bootstrap_ci_rejects_fewer_than_two_values():
    with pytest.raises(ValueError):
        bootstrap_ci([0.5])
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_bootstrap_ci_rejects_an_out_of_range_confidence():
    with pytest.raises(ValueError):
        bootstrap_ci([0.1, 0.2, 0.3], confidence=1.5)


def test_paired_bootstrap_test_finds_a_large_consistent_gap_significant():
    # system_b beats system_a by 0.3 on every single item, no exceptions --
    # a real, consistent improvement should read as significant.
    system_a = [0.5] * 30
    system_b = [0.8] * 30
    result = paired_bootstrap_test(system_a, system_b, seed=1)
    assert result.mean_diff == pytest.approx(0.3)
    assert result.p_value < 0.05


def test_paired_bootstrap_test_finds_identical_systems_not_significant():
    system_a = [0.1, 0.9, 0.4, 0.6, 0.7, 0.2, 0.5, 0.3]
    result = paired_bootstrap_test(system_a, list(system_a), seed=1)
    assert result.mean_diff == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)


def test_paired_bootstrap_test_is_noisy_with_a_tiny_inconsistent_sample():
    # A gap that only shows up on 1 of 4 paired items shouldn't read as
    # significant -- this is exactly the "n=24, one query is ~4pp" caveat
    # from benchmarks/rag_recall/RESULTS.md, reproduced at n=4 for clarity.
    system_a = [0.0, 0.0, 0.0, 0.0]
    system_b = [1.0, 0.0, 0.0, 0.0]
    result = paired_bootstrap_test(system_a, system_b, seed=1)
    assert result.p_value >= 0.05


def test_paired_bootstrap_test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_test([0.1, 0.2], [0.1, 0.2, 0.3])


def test_paired_bootstrap_test_rejects_fewer_than_two_items():
    with pytest.raises(ValueError):
        paired_bootstrap_test([0.1], [0.2])
