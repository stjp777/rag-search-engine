"""`rag-search image ...` commands."""

import argparse
from pathlib import Path

from rag_search.cli.common import load_documents, positive_int, snippet


def register(subparsers) -> None:
    parser = subparsers.add_parser("image", help="Search with an image")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    search = commands.add_parser("search", help="Find documents that match an image")
    search.add_argument("image", type=Path)
    search.add_argument("--limit", type=positive_int, default=5)
    search.set_defaults(func=search_command)

    embed = commands.add_parser("embed", help="Show the embedding size for an image")
    embed.add_argument("image", type=Path)
    embed.set_defaults(func=embed_command)

    rewrite = commands.add_parser(
        "rewrite", help="Rewrite a text query using an image (needs a vision-capable LLM)"
    )
    rewrite.add_argument("image", type=Path)
    rewrite.add_argument("query")
    rewrite.set_defaults(func=rewrite_command)


def search_command(args: argparse.Namespace) -> None:
    from rag_search.multimodal import MultiModalSearch

    documents, cache_dir = load_documents(args)
    results = MultiModalSearch(documents, cache_dir).search_with_image(args.image, args.limit)
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result['title']} (similarity: {result['similarity']:.3f})")
        print(f"   {snippet(result['description'])}\n")


def embed_command(args: argparse.Namespace) -> None:
    from rag_search.multimodal import MultiModalSearch

    embedding = MultiModalSearch().embed_image(args.image)
    print(f"Embedding dimensions: {embedding.shape[0]}")


def rewrite_command(args: argparse.Namespace) -> None:
    from rag_search.query_enhancement import rewrite_with_image

    print(f"Rewritten query: {rewrite_with_image(args.image, args.query)}")
