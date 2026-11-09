"""`rag-search semantic ...` commands."""

import argparse

from rag_search.chunking import fixed_size_chunking, semantic_chunking
from rag_search.cli.common import (
    load_documents,
    non_negative_int,
    positive_int,
    snippet,
)


def register(subparsers) -> None:
    parser = subparsers.add_parser("semantic", help="Embedding-based search")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    commands.add_parser(
        "info", help="Show the embedding model and its sequence length"
    ).set_defaults(func=info)

    embed = commands.add_parser("embed", help="Show the embedding of a piece of text")
    embed.add_argument("text")
    embed.set_defaults(func=embed_command)

    build = commands.add_parser("build", help="Build and cache embeddings")
    build.add_argument(
        "--chunks", action="store_true", help="Embed sentence chunks instead of whole documents"
    )
    build.set_defaults(func=build_command)

    search = commands.add_parser("search", help="Search whole-document embeddings")
    search.add_argument("query")
    search.add_argument("--limit", type=positive_int, default=5)
    search.set_defaults(func=search_command)

    search_chunks = commands.add_parser("search-chunks", help="Search sentence-chunk embeddings")
    search_chunks.add_argument("query")
    search_chunks.add_argument("--limit", type=positive_int, default=5)
    search_chunks.set_defaults(func=search_chunks_command)

    chunk = commands.add_parser("chunk", help="Split text into overlapping chunks")
    chunk.add_argument("text")
    chunk.add_argument("--by", choices=["words", "sentences"], default="words")
    chunk.add_argument(
        "--size", type=positive_int, default=200, help="words or sentences per chunk"
    )
    chunk.add_argument("--overlap", type=non_negative_int, default=0)
    chunk.set_defaults(func=chunk_command)


def info(args: argparse.Namespace) -> None:
    from rag_search.semantic import SemanticSearch

    search = SemanticSearch()
    print(f"Model: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")


def embed_command(args: argparse.Namespace) -> None:
    from rag_search.semantic import SemanticSearch

    embedding = SemanticSearch().generate_embedding(args.text)
    print(f"Dimensions: {embedding.shape[0]}")
    print(f"First 3 values: {embedding[:3]}")


def build_command(args: argparse.Namespace) -> None:
    from rag_search.semantic import ChunkedSemanticSearch, SemanticSearch

    documents, cache_dir = load_documents(args)
    if args.chunks:
        embeddings = ChunkedSemanticSearch(documents, cache_dir).load_or_create_chunk_embeddings()
        print(f"{len(embeddings)} chunk embeddings for {len(documents)} documents")
    else:
        embeddings = SemanticSearch(documents, cache_dir).load_or_create_embeddings()
        print(f"{embeddings.shape[0]} document embeddings, {embeddings.shape[1]} dimensions each")


def search_command(args: argparse.Namespace) -> None:
    from rag_search.semantic import SemanticSearch

    documents, cache_dir = load_documents(args)
    search = SemanticSearch(documents, cache_dir)
    search.load_or_create_embeddings()
    _print_results(search.search(args.query, args.limit))


def search_chunks_command(args: argparse.Namespace) -> None:
    from rag_search.semantic import ChunkedSemanticSearch

    documents, cache_dir = load_documents(args)
    search = ChunkedSemanticSearch(documents, cache_dir)
    search.load_or_create_chunk_embeddings()
    _print_results(search.search_chunks(args.query, args.limit))


def chunk_command(args: argparse.Namespace) -> None:
    split = fixed_size_chunking if args.by == "words" else semantic_chunking
    chunks = split(args.text, args.size, args.overlap)
    print(f"Split {len(args.text)} characters into {len(chunks)} chunks")
    for number, chunk in enumerate(chunks, start=1):
        print(f"{number}. {chunk}")


def _print_results(results: list[dict]) -> None:
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {snippet(result['description'])}")
