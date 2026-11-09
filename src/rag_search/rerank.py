"""Reranking search results with an LLM or a cross-encoder."""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from rag_search import config
from rag_search.errors import LLMError
from rag_search.llm import ask
from rag_search.util import format_documents, truncate

logger = logging.getLogger(__name__)

METHODS = ("individual", "batch", "cross-encoder")


def parse_score(text: str) -> float:
    """Extract a 0-10 score from a model reply. Uses the last number in the text."""
    out_of_ten = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10\b", text)
    numbers = [out_of_ten.group(1)] if out_of_ten else re.findall(r"\d+(?:\.\d+)?", text)
    if not numbers:
        raise ValueError(f"no number in reply: {text!r}")
    score = float(numbers[-1])
    if not 0 <= score <= 10:
        raise ValueError(f"score out of range: {score}")
    return score


def parse_id_list(text: str) -> list:
    """Extract a JSON array of ids from a model reply, ignoring markdown fences."""
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON array in reply: {text!r}")
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON array in reply: {text!r}") from exc
    if not all(isinstance(item, (int, str)) for item in parsed):
        raise ValueError(f"expected a list of ids, got: {parsed!r}")
    return parsed


def _score_one(query: str, result: dict) -> float | None:
    prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {format_documents([result], config.LLM_DOCUMENT_CHARS)}

Consider:
- Direct relevance to the query
- User intent (what they are looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number, no other text or explanation.

Score:"""
    try:
        return ask(prompt, parse=parse_score)
    except LLMError as exc:
        logger.warning("Could not score %r: %s", result["title"], exc)
        return None


def rerank_individual(query: str, results: list[dict]) -> list[dict]:
    """Ask the LLM to score each result from 0 to 10, then sort by score.

    Requests run concurrently. A result that could not be scored gets
    ``rerank_score = None`` and sorts last.
    """
    with ThreadPoolExecutor(max_workers=config.RERANK_WORKERS) as pool:
        scores = list(pool.map(lambda result: _score_one(query, result), results))
    for result, score in zip(results, scores, strict=True):
        result["rerank_score"] = score
    return sorted(
        results,
        key=lambda r: -1 if r["rerank_score"] is None else r["rerank_score"],
        reverse=True,
    )


def rerank_batch(query: str, results: list[dict]) -> list[dict]:
    """Ask the LLM to order all results in one request.

    Results the model leaves out keep their original order after the ranked ones.
    If the request fails, the original order is kept.
    """
    listing = "\n".join(
        f"{r['id']}: {r['title']} - {truncate(r['description'], config.LLM_DOCUMENT_CHARS)}"
        for r in results
    )
    prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{listing}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of ids.
Do not wrap the JSON in Markdown. Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""
    try:
        ranked_ids = ask(prompt, parse=parse_id_list)
    except LLMError as exc:
        logger.warning("Batch rerank failed, keeping original order: %s", exc)
        ranked_ids = []

    position: dict = {}
    for doc_id in ranked_ids:
        position.setdefault(doc_id, len(position) + 1)
    fallback = len(position) + 1
    for result in results:
        if result["id"] in position:
            result["rerank_rank"] = position[result["id"]]
        else:
            result["rerank_rank"] = fallback
            fallback += 1
    return sorted(results, key=lambda r: r["rerank_rank"])


@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.CROSS_ENCODER_MODEL)


def rerank_cross_encoder(query: str, results: list[dict]) -> list[dict]:
    """Score each (query, document) pair with a local cross-encoder model."""
    pairs = [[query, format_documents([r], config.LLM_DOCUMENT_CHARS)] for r in results]
    scores = _cross_encoder().predict(pairs)
    for result, score in zip(results, scores, strict=True):
        result["cross_encoder_score"] = float(score)
    return sorted(results, key=lambda r: r["cross_encoder_score"], reverse=True)


def rerank(query: str, results: list[dict], method: str) -> list[dict]:
    """Rerank results using one of ``METHODS``."""
    match method:
        case "individual":
            return rerank_individual(query, results)
        case "batch":
            return rerank_batch(query, results)
        case "cross-encoder":
            return rerank_cross_encoder(query, results)
        case _:
            raise ValueError(f"Unknown rerank method: {method!r}")
