"""Small shared helpers."""

from collections.abc import Iterable


def truncate(text: str, limit: int) -> str:
    """Shorten text to at most ``limit`` characters, marking the cut with an ellipsis."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def format_documents(results: Iterable[dict], limit: int, *, numbered: bool = False) -> str:
    """Format search results as plain text for inclusion in an LLM prompt."""
    lines = []
    for position, result in enumerate(results, start=1):
        label = f"[{position}] " if numbered else ""
        description = truncate(result.get("description", ""), limit)
        lines.append(f"{label}{result.get('title', '')} - {description}")
    return "\n".join(lines)
