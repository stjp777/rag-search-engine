"""Search quality evaluation: LLM relevance grading and precision/recall metrics."""

import json
import logging
import re
from collections.abc import Collection, Sequence
from pathlib import Path

from rag_search import config
from rag_search.errors import DatasetError, LLMError
from rag_search.llm import ask
from rag_search.util import format_documents

logger = logging.getLogger(__name__)


def precision_recall_f1(
    retrieved: Sequence[str], relevant: Collection[str]
) -> tuple[float, float, float]:
    """Precision, recall and F1 for a list of retrieved titles."""
    relevant = set(relevant)
    hits = sum(1 for title in retrieved if title in relevant)
    precision = hits / len(retrieved) if retrieved else 0.0
    recall = hits / len(relevant) if relevant else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def load_golden_dataset(path: str | Path) -> list[dict]:
    """Load test cases: ``{"test_cases": [{"query": ..., "relevant_docs": [titles]}]}``."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Golden dataset not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))["test_cases"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DatasetError(f"{path} must contain a 'test_cases' list") from exc


def _parse_grades(text: str, expected: int) -> list[int]:
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON list in reply: {text!r}")
    grades = json.loads(match.group())
    if (
        not isinstance(grades, list)
        or len(grades) != expected
        or not all(isinstance(g, int) and 0 <= g <= 3 for g in grades)
    ):
        raise ValueError(f"expected {expected} grades of 0-3, got: {grades!r}")
    return grades


def grade_results(query: str, results: list[dict]) -> list[int] | None:
    """Have an LLM grade each result 0-3. Returns None if no valid grades were produced."""
    listing = format_documents(results, config.LLM_DOCUMENT_CHARS)
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{listing}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
    try:
        return ask(prompt, parse=lambda text: _parse_grades(text, len(results)))
    except LLMError as exc:
        logger.warning("Grading failed: %s", exc)
        return None
