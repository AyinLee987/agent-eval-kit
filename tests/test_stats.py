"""Tests for the bootstrap CI / paired significance-test primitives."""

from __future__ import annotations

import pytest

from agent_eval import bootstrap_ci, paired_bootstrap_test
from agent_eval.stats import paired_permutation_test


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


def test_paired_permutation_test_finds_a_large_consistent_gap_significant():
    # system_b beats system_a by 0.3 on every single item, no exceptions --
    # a real, consistent improvement should read as significant.
    system_a = [0.5] * 30
    system_b = [0.8] * 30
    result = paired_permutation_test(system_a, system_b, seed=1)
    assert result.mean_diff == pytest.approx(0.3)
    assert result.p_value < 0.05


def test_paired_permutation_test_finds_identical_systems_not_significant():
    system_a = [0.1, 0.9, 0.4, 0.6, 0.7, 0.2, 0.5, 0.3]
    result = paired_permutation_test(system_a, list(system_a), seed=1)
    assert result.mean_diff == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)


def test_paired_permutation_test_is_noisy_with_a_tiny_inconsistent_sample():
    # A gap that only shows up on 1 of 4 paired items shouldn't read as
    # significant -- this is exactly the "n=24, one query is ~4pp" caveat
    # from benchmarks/rag_recall/RESULTS.md, reproduced at n=4 for clarity.
    system_a = [0.0, 0.0, 0.0, 0.0]
    system_b = [1.0, 0.0, 0.0, 0.0]
    result = paired_permutation_test(system_a, system_b, seed=1)
    assert result.p_value >= 0.05


def test_paired_permutation_test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_permutation_test([0.1, 0.2], [0.1, 0.2, 0.3])


def test_paired_permutation_test_rejects_fewer_than_two_items():
    with pytest.raises(ValueError):
        paired_permutation_test([0.1], [0.2])


# Expected Wilson values are analytic reference cases for the score interval;
# no SciPy/statsmodels dependency is required to run this regression suite.
@pytest.mark.parametrize("successes,n,expected", [
    (10, 10, (0.7224672001371107, 1.0)),
    (0, 10, (0.0, 0.2775327998628892)),
    (5, 10, (0.236593090512564, 0.7634069094874361)),
    (50, 50, (0.9286524008666414, 1.0)),
    (1, 1, (0.20654931437723745, 1.0)),
])
def test_wilson_interval_matches_reference_values(successes, n, expected):
    from agent_eval.stats import wilson_ci

    result = wilson_ci(successes, n)
    assert (result.low, result.high) == pytest.approx(expected, abs=1e-12)
    assert result.mean == successes / n
    assert result.n == n
    assert result.method == "wilson"
    assert result.n_resamples == 0
    assert "wilson" in result.render()


def test_wilson_intervals_reflect_sample_size_and_confidence():
    from agent_eval.stats import wilson_ci

    assert wilson_ci(50, 50).low > wilson_ci(10, 10).low
    assert wilson_ci(10, 10, confidence=0.99).low < wilson_ci(10, 10).low
    assert wilson_ci(0, 50).high < wilson_ci(0, 10).high
    assert wilson_ci(5, 10, confidence=0.9999999999999999).low >= 0.0


@pytest.mark.parametrize("successes,n", [
    (-1, 10), (11, 10), (1.0, 10), (True, 10),
    (0, 0), (0, -1), (0, 2.0), (0, True),
])
def test_wilson_rejects_invalid_counts(successes, n):
    from agent_eval.stats import wilson_ci

    with pytest.raises(ValueError):
        wilson_ci(successes, n)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), "0.5", None, 1j])
def test_bootstrap_and_paired_tests_reject_nonfinite_or_non_numeric_scores(value):
    from agent_eval.stats import paired_permutation_test

    with pytest.raises(ValueError, match="finite real"):
        bootstrap_ci([0.0, value])
    with pytest.raises(ValueError, match="finite real"):
        paired_permutation_test([0.0, value], [0.0, 1.0])
    with pytest.raises(ValueError, match="finite real"):
        paired_permutation_test([0.0, 1.0], [0.0, value])


@pytest.mark.parametrize("value", [0, -1, 2.5, True, float("nan"), float("inf"), "100"])
def test_resampling_counts_must_be_positive_integers(value):
    from agent_eval.stats import paired_permutation_test

    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_ci([0.0, 1.0], n_resamples=value)
    with pytest.raises(ValueError, match="n_resamples"):
        paired_permutation_test([0.0, 0.0], [1.0, 1.0], n_resamples=value)


@pytest.mark.parametrize("value", [0.0, 1.0, -0.5, float("nan"), float("inf"), True, "0.95", None])
def test_confidence_must_be_a_finite_probability(value):
    from agent_eval.stats import wilson_ci

    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci([0.0, 1.0], confidence=value)
    with pytest.raises(ValueError, match="confidence"):
        wilson_ci(5, 10, confidence=value)


def test_finite_large_scores_do_not_overflow_the_bootstrap_mean():
    result = bootstrap_ci([1e308, 1e308], n_resamples=10)
    assert result.mean == result.low == result.high == 1e308


def test_overflowing_paired_differences_are_rejected():
    from agent_eval.stats import paired_permutation_test

    with pytest.raises(ValueError, match="differences must remain finite"):
        paired_permutation_test([-1e308, 0.0], [1e308, 1.0])


def test_two_positive_pairs_have_exact_p_value_one_half():
    from agent_eval.stats import paired_permutation_test

    # Four equally likely assignments give differences +1, 0, 0, -1.
    result = paired_permutation_test([0.0, 0.0], [1.0, 1.0])
    assert result.mean_diff == 1.0
    assert result.p_value == 0.5
    assert result.n == 2
    assert result.n_resamples == 4
    assert result.method == "paired_permutation_exact"


@pytest.mark.parametrize("deltas,expected", [
    ([1, 1, 1], 0.25),
    ([1, 1, 1, 1], 0.125),
    ([1, -1], 1.0),
    ([0, 0, 0, 0], 1.0),
    ([1, 0, 0, 0], 1.0),
    ([1, 1, 0, 0], 0.5),
])
def test_exact_paired_permutation_matches_enumerated_null_cases(deltas, expected):
    from agent_eval.stats import paired_permutation_test

    result = paired_permutation_test([0] * len(deltas), deltas)
    assert result.p_value == expected


def test_exact_paired_test_is_invariant_to_seed_order_and_condition_swap():
    from agent_eval.stats import paired_permutation_test

    left = [0.0, 0.5, 0.7, 0.2]
    right = [0.8, 0.4, 0.8, 0.8]
    forward = paired_permutation_test(left, right, seed=0)
    reordered = paired_permutation_test(left[::-1], right[::-1], seed=999)
    reverse = paired_permutation_test(right, left)
    assert forward == reordered
    assert reverse.mean_diff == pytest.approx(-forward.mean_diff)
    assert reverse.p_value == forward.p_value


def test_monte_carlo_plus_one_prevents_zero_p_values_and_is_reproducible():
    from agent_eval.stats import paired_permutation_test

    left = [0.0] * 20
    right = [1.0] * 20
    result = paired_permutation_test(left, right, n_resamples=99, seed=0)
    assert result == paired_permutation_test(left, right, n_resamples=99, seed=0)
    assert result.p_value == 1 / 100
    assert result.n_resamples == 99
    assert result.method == "paired_permutation_monte_carlo"

    small = paired_permutation_test(left, right, n_resamples=10_000, seed=0)
    displayed_p = small.render().split("p=")[1].split()[0]
    assert float(displayed_p) > 0
    assert "monte_carlo" in small.render()


def test_deprecated_paired_bootstrap_name_uses_the_correct_null_test():
    from agent_eval.stats import paired_permutation_test

    with pytest.warns(DeprecationWarning, match="permutation"):
        legacy = paired_bootstrap_test([0.0, 0.0], [1.0, 1.0])
    assert legacy == paired_permutation_test([0.0, 0.0], [1.0, 1.0])
    assert legacy.p_value == 0.5


def test_result_dataclasses_keep_positional_construction_compatible():
    from agent_eval.stats import ConfidenceInterval, PairedTestResult

    ci = ConfidenceInterval(0.5, 0.2, 0.8, 20, 1000, 0.95)
    test = PairedTestResult(0.1, 0.001, 20, 1000)
    assert "percentile_bootstrap" in ci.render()
    assert "paired_permutation" in test.render()


def test_safety_summary_uses_wilson_without_importing_model_or_cache_code(capsys):
    import ast
    from pathlib import Path
    from typing import Any, Dict

    from agent_eval.stats import wilson_ci

    # The benchmark has import-time provider setup. Extract only its pure
    # summary function so this regression cannot load .env or call a model.
    source = Path(__file__).resolve().parents[1] / "benchmarks" / "safety" / "run_benchmark.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    summarize = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "summarize"
    )
    namespace = {"wilson_ci": wilson_ci, "Dict": Dict, "Any": Any}
    exec(compile(ast.Module(body=[summarize], type_ignores=[]), str(source), "exec"), namespace)
    namespace["summarize"]({
        "refusal": {str(index): {"verdict": "REFUSED"} for index in range(50)},
        "escalation": {str(index): {"verdict": "INTERRUPTED"} for index in range(10)},
    })
    output = capsys.readouterr().out
    assert "[0.929, 1.000] Wilson 95% CI" in output
    assert "[0.722, 1.000] Wilson 95% CI" in output
