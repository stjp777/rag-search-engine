import pytest

from rag_search.errors import IndexNotFoundError
from rag_search.keyword import InvertedIndex, tokenize, tokenize_term

DOCS = [
    {"id": 1, "title": "Space Bear", "description": "A bear travels through space."},
    {"id": 2, "title": "Garden Party", "description": "A quiet party in the garden."},
    {"id": 3, "title": "Bear Hunt", "description": "A hunt for a bear in the woods."},
]


@pytest.fixture
def index():
    idx = InvertedIndex(DOCS)
    idx.build()
    return idx


def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("The bears are running") == ["bear", "run"]


def test_tokenize_splits_hyphenated_words():
    assert tokenize("spider-man") == ["spider", "man"]


def test_tokenize_term_requires_one_token():
    with pytest.raises(ValueError):
        tokenize_term("the")
    with pytest.raises(ValueError):
        tokenize_term("two words")


def test_term_frequency(index):
    assert index.term_frequency(1, "bear") == 2
    assert index.term_frequency(2, "bear") == 0


def test_unknown_document_id(index):
    with pytest.raises(ValueError):
        index.term_frequency(99, "bear")


def test_bm25_search_ranks_matching_documents_only(index):
    results = index.bm25_search("bear", limit=10)
    assert {doc_id for doc_id, _ in results} == {1, 3}
    assert results == sorted(results, key=lambda r: r[1], reverse=True)


def test_bm25_search_with_only_stopwords_is_empty(index):
    assert index.bm25_search("the a of", limit=5) == []


def test_idf_decreases_with_document_frequency(index):
    assert index.bm25_idf("garden") > index.bm25_idf("bear")


def test_save_and_load_round_trip(index, tmp_path):
    index.save(tmp_path)
    loaded = InvertedIndex(DOCS)
    loaded.load(tmp_path)
    assert loaded.bm25_search("bear", 5) == index.bm25_search("bear", 5)


def test_load_missing_index(tmp_path):
    with pytest.raises(IndexNotFoundError):
        InvertedIndex(DOCS).load(tmp_path)
