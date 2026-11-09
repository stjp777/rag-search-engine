from rag_search.hybrid import HybridSearch
from rag_search.keyword import InvertedIndex

DOCS = [
    {"id": 1, "title": "Dragon Gold", "description": "A dragon guards gold.", "genre": "fantasy"},
    {"id": 2, "title": "Dragon Comedy", "description": "A funny dragon.", "genre": "comedy"},
    {"id": 3, "title": "Sea Tale", "description": "A sailor at sea.", "genre": "comedy"},
]


class FakeSemantic:
    """Ranks documents in id order; stands in for the embedding model."""

    def search_chunks(self, query, limit):
        return [{"id": d["id"], "score": 1.0 - i / 10} for i, d in enumerate(DOCS)][:limit]


def make_search():
    search = HybridSearch.__new__(HybridSearch)
    search.documents = {d["id"]: d for d in DOCS}
    search.semantic = FakeSemantic()
    search.keyword = InvertedIndex(DOCS)
    search.keyword.build()
    return search


def test_rrf_filter_excludes_non_matching_documents():
    results = make_search().rrf_search("dragon", limit=5, filters={"genre": "comedy"})
    assert {r["id"] for r in results} == {2, 3}
    assert results[0]["id"] == 2


def test_weighted_filter_excludes_non_matching_documents():
    results = make_search().weighted_search("dragon", 0.5, 5, filters={"genre": "fantasy"})
    assert [r["id"] for r in results] == [1]


def test_no_filter_returns_everything():
    assert len(make_search().rrf_search("dragon", limit=5)) == 3
