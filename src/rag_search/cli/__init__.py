"""Command-line interface: ``rag-search <group> <command>``."""

import argparse
import logging
import sys
from collections.abc import Sequence

from rag_search import __version__
from rag_search.cli import evaluate, generate, hybrid, image, keyword, semantic
from rag_search.errors import RagSearchError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-search", description="Keyword, semantic and hybrid search over a JSON dataset."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--dataset",
        metavar="PATH",
        help="dataset JSON file (default: $RAG_DATASET or data/sample_movies.json)",
    )
    groups = parser.add_subparsers(dest="group", required=True, metavar="GROUP")
    for module in (keyword, semantic, hybrid, generate, evaluate, image):
        module.register(groups)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (RagSearchError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
