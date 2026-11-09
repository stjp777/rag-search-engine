"""Splitting text into overlapping chunks."""

import re

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _check_window(size: int, overlap: int) -> None:
    if size < 1:
        raise ValueError("chunk size must be at least 1")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= size:
        raise ValueError("overlap must be smaller than the chunk size")


def _window(items: list[str], size: int, overlap: int) -> list[list[str]]:
    """Slide a window over items, stepping by ``size - overlap``."""
    windows = []
    step = size - overlap
    for start in range(0, len(items), step):
        piece = items[start : start + size]
        # A trailing piece made only of overlap adds nothing new.
        if windows and len(piece) <= overlap:
            break
        windows.append(piece)
    return windows


def semantic_chunking(text: str, max_sentences: int, overlap: int) -> list[str]:
    """Group sentences into chunks of up to ``max_sentences`` sentences."""
    _check_window(max_sentences, overlap)
    text = text.strip()
    if not text:
        return []
    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(text)]
    chunks = (" ".join(piece).strip() for piece in _window(sentences, max_sentences, overlap))
    return [chunk for chunk in chunks if chunk]


def fixed_size_chunking(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Group words into chunks of up to ``chunk_size`` words."""
    _check_window(chunk_size, overlap)
    return [" ".join(piece) for piece in _window(text.split(), chunk_size, overlap)]
