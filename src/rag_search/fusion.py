"""Score normalisation and fusion helpers for hybrid search."""

from collections.abc import Sequence

from rag_search.config import RRF_K


def normalize(scores: Sequence[float]) -> list[float]:
    """Min-max scale scores to the range 0-1.

    If every score is identical there is nothing to scale against, so each
    one maps to 1.0.
    """
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high == low:
        return [1.0] * len(scores)
    span = high - low
    return [(score - low) / span for score in scores]


def rrf_score(rank: int, k: int = RRF_K) -> float:
    """Reciprocal Rank Fusion contribution for a 1-based rank."""
    return 1 / (k + rank)


def hybrid_score(keyword_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    """Weighted blend: ``alpha`` is the weight given to the keyword score."""
    return alpha * keyword_score + (1 - alpha) * semantic_score
