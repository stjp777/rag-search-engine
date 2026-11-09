import pytest

from rag_search.evaluation import precision_recall_f1


def test_perfect_retrieval():
    assert precision_recall_f1(["A", "B"], ["A", "B"]) == (1.0, 1.0, 1.0)


def test_partial_retrieval():
    precision, recall, f1 = precision_recall_f1(["A", "X"], ["A", "B"])
    assert (precision, recall) == (0.5, 0.5)
    assert f1 == pytest.approx(0.5)


def test_no_hits_does_not_divide_by_zero():
    assert precision_recall_f1(["X"], ["A"]) == (0.0, 0.0, 0.0)


def test_nothing_retrieved():
    assert precision_recall_f1([], ["A"]) == (0.0, 0.0, 0.0)
