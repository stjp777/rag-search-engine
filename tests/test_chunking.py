import pytest

from rag_search.chunking import fixed_size_chunking, semantic_chunking


def test_semantic_chunking_groups_sentences_with_overlap():
    text = "One. Two. Three. Four. Five."
    assert semantic_chunking(text, 3, 1) == ["One. Two. Three.", "Three. Four. Five."]


def test_semantic_chunking_single_sentence_without_punctuation():
    assert semantic_chunking("no punctuation here", 4, 1) == ["no punctuation here"]


def test_semantic_chunking_empty_text():
    assert semantic_chunking("   ", 4, 1) == []


def test_fixed_size_chunking_words():
    assert fixed_size_chunking("a b c d e", 2, 0) == ["a b", "c d", "e"]


def test_fixed_size_chunking_overlap_drops_redundant_tail():
    assert fixed_size_chunking("a b c d", 3, 2) == ["a b c", "b c d"]


@pytest.mark.parametrize("size, overlap", [(3, 3), (3, 5), (0, 0), (3, -1)])
def test_invalid_window_is_rejected(size, overlap):
    with pytest.raises(ValueError):
        semantic_chunking("One. Two.", size, overlap)
    with pytest.raises(ValueError):
        fixed_size_chunking("a b", size, overlap)
