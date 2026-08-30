"""Tests for the cosine-similarity primitives used by the robustness benchmark."""

from __future__ import annotations

import math

import pytest

from agent_eval import average_pairwise_similarity, cosine_similarity


def test_identical_vectors_are_maximally_similar():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_orthogonal_vectors_have_zero_similarity():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_opposite_vectors_have_similarity_negative_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_scale_invariant():
    a = [1.0, 2.0, 3.0]
    b = [2.0, 4.0, 6.0]
    assert cosine_similarity(a, b) == pytest.approx(1.0)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


def test_zero_vector_scores_zero_rather_than_dividing_by_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_average_pairwise_similarity_of_identical_vectors_is_one():
    vectors = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    assert average_pairwise_similarity(vectors) == pytest.approx(1.0)


def test_average_pairwise_similarity_matches_hand_computed_value():
    # cos(a,b)=1, cos(a,c)=0, cos(b,c)=0 -> mean = 1/3
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    assert average_pairwise_similarity(vectors) == pytest.approx(1 / 3)


def test_average_pairwise_similarity_is_one_for_zero_or_one_vector():
    assert average_pairwise_similarity([]) == 1.0
    assert average_pairwise_similarity([[1.0, 2.0]]) == 1.0
