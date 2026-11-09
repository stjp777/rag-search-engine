import pytest

from rag_search.fusion import hybrid_score, normalize, rrf_score


def test_normalize_scales_to_unit_range():
    assert normalize([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]


def test_normalize_equal_scores_map_to_one():
    assert normalize([3.0, 3.0]) == [1.0, 1.0]


def test_normalize_empty():
    assert normalize([]) == []


def test_rrf_score():
    assert rrf_score(1, k=60) == pytest.approx(1 / 61)


def test_hybrid_score_weights():
    assert hybrid_score(1.0, 0.0, alpha=0.25) == pytest.approx(0.25)
    assert hybrid_score(0.0, 1.0, alpha=0.25) == pytest.approx(0.75)
