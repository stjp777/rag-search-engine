"""OpenRouter client shared by every LLM-backed feature."""

import logging
import os
import time
from collections.abc import Callable
from functools import lru_cache

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI, OpenAIError

from rag_search import config
from rag_search.errors import LLMError

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    load_dotenv(find_dotenv(usecwd=True))
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMError("OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key.")
    return OpenAI(base_url=config.LLM_BASE_URL, api_key=api_key)


def ask[T](
    prompt: str | list,
    parse: Callable[[str], T] = str,
    retries: int = config.MAX_RETRIES,
) -> T:
    """Send a prompt (or a full message list) and return ``parse(reply_text)``.

    The request is retried when the API fails or ``parse`` raises ValueError.
    LLMError is raised once every attempt has failed.
    """
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    client = get_client()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(model=config.LLM_MODEL, messages=messages)
            text = (response.choices[0].message.content or "").strip()
            return parse(text)
        except (OpenAIError, ValueError) as exc:
            last_error = exc
            logger.warning("LLM request failed (attempt %d/%d): %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(config.RETRY_DELAY)
    raise LLMError(f"LLM request failed after {retries} attempts: {last_error}")
