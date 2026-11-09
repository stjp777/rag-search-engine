"""Hybrid search combining BM25 keyword scores with chunked semantic scores."""

from collections.abc import Sequence
from pathlib import Path

from rag_search.config import CANDIDATE_MULTIPLIER, RRF_K
from rag_search.fusion import hybrid_score, normalize, rrf_score
from rag_search.keyword import InvertedIndex
from rag_search.semantic import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: Sequence[dict], cache_dir: Path) -> None:
        self.documents = {doc["id"]: doc for doc in documents}
        self.semantic = ChunkedSemanticSearch(documents, cache_dir)
        self.semantic.load_or_create_chunk_embeddings()
        self.keyword = InvertedIndex.load_or_build(documents, cache_dir)

    def _pool_size(self, limit: int) -> int:
        return min(len(self.documents), limit * CANDIDATE_MULTIPLIER)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        """Blend normalised keyword and semantic scores; ``alpha`` weights the keyword side."""
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        pool = self._pool_size(limit)
        keyword = self.keyword.bm25_search(query, pool)
        semantic = self.semantic.search_chunks(query, pool)

        scores: dict = {}
        for (doc_id, _), value in zip(keyword, normalize([s for _, s in keyword]), strict=True):
            scores.setdefault(doc_id, {"keyword_score": 0.0, "semantic_score": 0.0})
            scores[doc_id]["keyword_score"] = value
        for result, value in zip(semantic, normalize([r["score"] for r in semantic]), strict=True):
            scores.setdefault(result["id"], {"keyword_score": 0.0, "semantic_score": 0.0})
            scores[result["id"]]["semantic_score"] = value

        results = [
            {
                **self._base(doc_id),
                **entry,
                "hybrid_score": hybrid_score(
                    entry["keyword_score"], entry["semantic_score"], alpha
                ),
            }
            for doc_id, entry in scores.items()
        ]
        results.sort(key=lambda r: r["hybrid_score"], reverse=True)
        return results[:limit]

    def rrf_search(self, query: str, k: int = RRF_K, limit: int = 10) -> list[dict]:
        """Combine the keyword and semantic rankings with Reciprocal Rank Fusion."""
        pool = self._pool_size(limit)
        keyword = self.keyword.bm25_search(query, pool)
        semantic = self.semantic.search_chunks(query, pool)

        ranks: dict = {}
        for rank, (doc_id, _) in enumerate(keyword, start=1):
            ranks.setdefault(doc_id, {"keyword_rank": None, "semantic_rank": None})
            ranks[doc_id]["keyword_rank"] = rank
        for rank, result in enumerate(semantic, start=1):
            ranks.setdefault(result["id"], {"keyword_rank": None, "semantic_rank": None})
            ranks[result["id"]]["semantic_rank"] = rank

        results = [
            {
                **self._base(doc_id),
                **entry,
                "rrf_score": sum(
                    rrf_score(rank, k)
                    for rank in (entry["keyword_rank"], entry["semantic_rank"])
                    if rank is not None
                ),
            }
            for doc_id, entry in ranks.items()
        ]
        results.sort(key=lambda r: r["rrf_score"], reverse=True)
        return results[:limit]

    def _base(self, doc_id) -> dict:
        doc = self.documents[doc_id]
        return {"id": doc_id, "title": doc["title"], "description": doc["description"]}
