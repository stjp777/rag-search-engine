"""Keyword search: tokenizer, inverted index, TF-IDF and BM25."""

import math
import pickle
import string
from collections import Counter
from collections.abc import Sequence
from functools import cache, lru_cache
from importlib import resources
from pathlib import Path

from nltk.stem import PorterStemmer

from rag_search.config import BM25_B, BM25_K1
from rag_search.errors import IndexNotFoundError

INDEX_FILENAME = "index.pkl"

_STEMMER = PorterStemmer()
# Apostrophes are dropped ("don't" -> "dont"); other punctuation separates words.
_PUNCTUATION = str.maketrans({ch: " " for ch in string.punctuation if ch != "'"} | {"'": None})


@cache
def _stem(token: str) -> str:
    return _STEMMER.stem(token)


def _normalize_text(text: str) -> str:
    return text.lower().translate(_PUNCTUATION)


@lru_cache(maxsize=1)
def load_stopwords() -> frozenset[str]:
    """Load the bundled stopword list, normalised the same way as query text."""
    raw = resources.files("rag_search").joinpath("data", "stopwords.txt").read_text("utf-8")
    return frozenset(_normalize_text(word) for word in raw.split() if word)


def tokenize(text: str) -> list[str]:
    """Lowercase, split, drop stopwords and stem."""
    stopwords = load_stopwords()
    return [_stem(token) for token in _normalize_text(text).split() if token not in stopwords]


def tokenize_term(term: str) -> str:
    """Tokenize a single word, rejecting input that does not reduce to one token."""
    tokens = tokenize(term)
    if len(tokens) != 1:
        raise ValueError(
            f"{term!r} must be a single word that is not a stopword (got {len(tokens)} tokens)"
        )
    return tokens[0]


class InvertedIndex:
    """Term -> document index with per-document term frequencies."""

    def __init__(self, documents: Sequence[dict] = ()) -> None:
        self.documents = {doc["id"]: doc for doc in documents}
        self.index: dict[str, set] = {}
        self.term_frequencies: dict = {}
        self.doc_lengths: dict = {}
        self._avg_doc_length = 0.0

    # Building and persistence ------------------------------------------------

    def build(self) -> None:
        """Index every document's title and description."""
        self.index, self.term_frequencies, self.doc_lengths = {}, {}, {}
        for doc_id, doc in self.documents.items():
            tokens = tokenize(f"{doc['title']} {doc['description']}")
            self.term_frequencies[doc_id] = Counter(tokens)
            self.doc_lengths[doc_id] = len(tokens)
            for token in set(tokens):
                self.index.setdefault(token, set()).add(doc_id)
        self._update_average_length()

    def save(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "index": self.index,
            "term_frequencies": self.term_frequencies,
            "doc_lengths": self.doc_lengths,
        }
        with (cache_dir / INDEX_FILENAME).open("wb") as file:
            pickle.dump(payload, file)

    def load(self, cache_dir: Path) -> None:
        """Load a previously saved index. Only load caches this program wrote."""
        path = cache_dir / INDEX_FILENAME
        if not path.is_file():
            raise IndexNotFoundError(
                "Keyword index not found. Run `rag-search keyword build` first."
            )
        with path.open("rb") as file:
            payload = pickle.load(file)
        self.index = payload["index"]
        self.term_frequencies = payload["term_frequencies"]
        self.doc_lengths = payload["doc_lengths"]
        self._update_average_length()

    @classmethod
    def load_or_build(cls, documents: Sequence[dict], cache_dir: Path) -> "InvertedIndex":
        index = cls(documents)
        try:
            index.load(cache_dir)
        except IndexNotFoundError:
            index.build()
            index.save(cache_dir)
        return index

    def _update_average_length(self) -> None:
        total = sum(self.doc_lengths.values())
        self._avg_doc_length = total / len(self.doc_lengths) if self.doc_lengths else 0.0

    # Term statistics ---------------------------------------------------------

    def _require_document(self, doc_id) -> None:
        if doc_id not in self.doc_lengths:
            raise ValueError(f"Unknown document id: {doc_id!r}")

    def document_frequency(self, term: str) -> int:
        return len(self.index.get(term, ()))

    def term_frequency(self, doc_id, term: str) -> int:
        self._require_document(doc_id)
        return self.term_frequencies[doc_id][term]

    def idf(self, term: str) -> float:
        """Smoothed inverse document frequency."""
        return math.log((len(self.doc_lengths) + 1) / (self.document_frequency(term) + 1))

    def tf_idf(self, doc_id, term: str) -> float:
        return self.term_frequency(doc_id, term) * self.idf(term)

    def bm25_idf(self, term: str) -> float:
        total = len(self.doc_lengths)
        df = self.document_frequency(term)
        return math.log((total - df + 0.5) / (df + 0.5) + 1)

    def bm25_tf(self, doc_id, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        """Term frequency saturated and normalised by document length."""
        tf = self.term_frequency(doc_id, term)
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self._avg_doc_length)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    # Search ------------------------------------------------------------------

    def bm25_search(self, query: str, limit: int) -> list[tuple]:
        """Return up to ``limit`` ``(doc_id, score)`` pairs, best first.

        Only documents containing at least one query term are scored.
        """
        tokens = tokenize(query)
        idf = {token: self.bm25_idf(token) for token in set(tokens)}
        candidates = set().union(*(self.index.get(token, ()) for token in idf))
        scores = {
            doc_id: sum(idf[token] * self.bm25_tf(doc_id, token) for token in tokens)
            for doc_id in candidates
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:limit]
