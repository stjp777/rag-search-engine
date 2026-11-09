"""Helpers shared by the CLI command modules."""

import argparse
from pathlib import Path

from rag_search.config import DISPLAY_CHARS
from rag_search.dataset import cache_dir_for, load_dataset, resolve_dataset_path
from rag_search.util import truncate


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be 0 or greater")
    return number


def unit_float(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def load_documents(args: argparse.Namespace) -> tuple[list[dict], Path]:
    """Load the selected dataset and return it with its cache directory."""
    path = resolve_dataset_path(args.dataset)
    return load_dataset(path), cache_dir_for(path)


def resolve_doc_id(documents: list[dict], raw: str):
    """Match a command-line id against the dataset's ids (integer or string)."""
    ids = {doc["id"] for doc in documents}
    if raw.lstrip("-").isdigit() and int(raw) in ids:
        return int(raw)
    if raw in ids:
        return raw
    raise ValueError(f"Unknown document id: {raw}")


def snippet(description: str) -> str:
    return truncate(description, DISPLAY_CHARS)
