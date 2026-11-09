"""LLM-based query improvement: spelling fixes, rewriting and expansion."""

import base64
import logging
import mimetypes
from pathlib import Path

from rag_search.errors import LLMError
from rag_search.llm import ask

logger = logging.getLogger(__name__)

METHODS = ("spell", "rewrite", "expand")


def _clean(text: str) -> str:
    return text.strip().strip('"')


def spell_correct(query: str) -> str:
    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""
    return ask(prompt, parse=_clean) or query


def rewrite(query: str) -> str:
    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
    return ask(prompt, parse=_clean) or query


def expand(query: str) -> str:
    prompt = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""
    expansion = ask(prompt, parse=_clean)
    # The examples echo the original query, so avoid appending it twice.
    if expansion.lower().startswith(query.lower()):
        return expansion
    return f"{query} {expansion}".strip()


def enhance_query(query: str, method: str | None) -> str:
    """Apply one of ``METHODS``. On LLM failure the original query is returned."""
    handlers = {"spell": spell_correct, "rewrite": rewrite, "expand": expand}
    if method not in handlers:
        return query
    try:
        return handlers[method](query)
    except LLMError as exc:
        logger.warning("Query enhancement skipped: %s", exc)
        return query


def rewrite_with_image(image_path: str | Path, query: str) -> str:
    """Rewrite a text query using the content of an image."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode()

    instructions = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instructions},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                {"type": "text", "text": query.strip()},
            ],
        }
    ]
    return ask(messages, parse=_clean)
