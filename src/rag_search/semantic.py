"""Semantic search with sentence embeddings."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_search.chunking import semantic_chunking
from rag_search.config import CHUNK_OVERLAP, CHUNK_SENTENCES, EMBEDDING_MODEL
from rag_search.errors import IndexNotFoundError

EMBEDDINGS_FILENAME = "document_embeddings.npy"
CHUNK_EMBEDDINGS_FILENAME = "chunk_embeddings.npy"
CHUNK_OWNERS_FILENAME = "chunk_owners.npy"


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length (all-zero rows are left as zeros)."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_scores(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Cosine similarity between ``vector`` and every row of ``matrix``."""
    return unit_rows(matrix) @ unit_rows(vector)


class SemanticSearch:
    """Embeds whole documents and ranks them by cosine similarity to a query."""

    def __init__(
        self,
        documents: Sequence[dict] = (),
        cache_dir: Path | None = None,
        model_name: str = EMBEDDING_MODEL,
    ) -> None:
        self.documents = list(documents)
        self.cache_dir = cache_dir
        self.model = SentenceTransformer(model_name)
        self.embeddings: np.ndarray | None = None

    def generate_embedding(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("Cannot embed empty text")
        return self.model.encode([text])[0]

    def build_embeddings(self) -> np.ndarray:
        texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        self._save(EMBEDDINGS_FILENAME, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self) -> np.ndarray:
        cached = self._load(EMBEDDINGS_FILENAME)
        if cached is not None and len(cached) == len(self.documents):
            self.embeddings = cached
            return cached
        return self.build_embeddings()

    def search(self, query: str, limit: int) -> list[dict]:
        if self.embeddings is None:
            raise IndexNotFoundError(
                "No embeddings loaded. Call load_or_create_embeddings() first."
            )
        scores = cosine_scores(self.embeddings, self.generate_embedding(query))
        top = np.argsort(-scores)[:limit]
        return [self._result(int(i), float(scores[i])) for i in top]

    def _result(self, doc_index: int, score: float) -> dict:
        doc = self.documents[doc_index]
        return {
            "id": doc["id"],
            "title": doc["title"],
            "description": doc["description"],
            "score": score,
        }

    def _save(self, filename: str, array: np.ndarray) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.cache_dir / filename, array)

    def _load(self, filename: str) -> np.ndarray | None:
        if self.cache_dir is None or not (self.cache_dir / filename).is_file():
            return None
        return np.load(self.cache_dir / filename)


class ChunkedSemanticSearch(SemanticSearch):
    """Embeds sentence chunks; a document scores as high as its best chunk."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.chunk_embeddings: np.ndarray | None = None
        self.chunk_owners: np.ndarray | None = None  # document index of each chunk

    def build_chunk_embeddings(self) -> np.ndarray:
        chunks, owners = [], []
        for doc_index, doc in enumerate(self.documents):
            doc_chunks = semantic_chunking(doc["description"], CHUNK_SENTENCES, CHUNK_OVERLAP)
            chunks.extend(doc_chunks)
            owners.extend([doc_index] * len(doc_chunks))
        if not chunks:
            raise ValueError("No document has a description to embed")

        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_owners = np.array(owners, dtype=np.int32)
        self._save(CHUNK_EMBEDDINGS_FILENAME, self.chunk_embeddings)
        self._save(CHUNK_OWNERS_FILENAME, self.chunk_owners)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self) -> np.ndarray:
        embeddings = self._load(CHUNK_EMBEDDINGS_FILENAME)
        owners = self._load(CHUNK_OWNERS_FILENAME)
        if (
            embeddings is not None
            and owners is not None
            and len(embeddings) == len(owners)
            and owners.max(initial=-1) < len(self.documents)
        ):
            self.chunk_embeddings, self.chunk_owners = embeddings, owners
            return embeddings
        return self.build_chunk_embeddings()

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        if self.chunk_embeddings is None or self.chunk_owners is None:
            raise IndexNotFoundError(
                "No chunk embeddings loaded. Call load_or_create_chunk_embeddings() first."
            )
        chunk_scores = cosine_scores(self.chunk_embeddings, self.generate_embedding(query))
        best = np.full(len(self.documents), -np.inf)
        np.maximum.at(best, self.chunk_owners, chunk_scores)
        top = [i for i in np.argsort(-best)[:limit] if np.isfinite(best[i])]
        return [self._result(int(i), float(best[i])) for i in top]
