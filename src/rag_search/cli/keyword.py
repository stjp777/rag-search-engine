"""`rag-search keyword ...` commands."""

import argparse

from rag_search.cli.common import load_documents, positive_int, resolve_doc_id
from rag_search.config import BM25_B, BM25_K1
from rag_search.keyword import InvertedIndex, tokenize_term


def register(subparsers) -> None:
    parser = subparsers.add_parser("keyword", help="BM25 keyword search and index inspection")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    commands.add_parser("build", help="Build and cache the inverted index").set_defaults(func=build)

    search = commands.add_parser("search", help="Rank documents with BM25")
    search.add_argument("query")
    search.add_argument("--limit", type=positive_int, default=5)
    search.set_defaults(func=search_command)

    tf = commands.add_parser("tf", help="Term frequency of a word in a document")
    tf.add_argument("doc_id")
    tf.add_argument("term")
    tf.set_defaults(func=tf_command)

    idf = commands.add_parser("idf", help="Inverse document frequency of a word")
    idf.add_argument("term")
    idf.set_defaults(func=idf_command)

    tfidf = commands.add_parser("tfidf", help="TF-IDF score of a word in a document")
    tfidf.add_argument("doc_id")
    tfidf.add_argument("term")
    tfidf.set_defaults(func=tfidf_command)

    bm25_idf = commands.add_parser("bm25-idf", help="BM25 IDF of a word")
    bm25_idf.add_argument("term")
    bm25_idf.set_defaults(func=bm25_idf_command)

    bm25_tf = commands.add_parser(
        "bm25-tf", help="BM25 term frequency score of a word in a document"
    )
    bm25_tf.add_argument("doc_id")
    bm25_tf.add_argument("term")
    bm25_tf.add_argument("--k1", type=float, default=BM25_K1)
    bm25_tf.add_argument("--b", type=float, default=BM25_B)
    bm25_tf.set_defaults(func=bm25_tf_command)


def _loaded_index(args: argparse.Namespace) -> tuple[InvertedIndex, list[dict]]:
    documents, cache_dir = load_documents(args)
    index = InvertedIndex(documents)
    index.load(cache_dir)
    return index, documents


def build(args: argparse.Namespace) -> None:
    documents, cache_dir = load_documents(args)
    index = InvertedIndex(documents)
    index.build()
    index.save(cache_dir)
    print(f"Indexed {len(documents)} documents ({len(index.index)} terms) in {cache_dir}")


def search_command(args: argparse.Namespace) -> None:
    index, _ = _loaded_index(args)
    results = index.bm25_search(args.query, args.limit)
    if not results:
        print("No matching documents.")
        return
    for rank, (doc_id, score) in enumerate(results, start=1):
        print(f"{rank}. ({doc_id}) {index.documents[doc_id]['title']} - score {score:.2f}")


def tf_command(args: argparse.Namespace) -> None:
    index, documents = _loaded_index(args)
    doc_id = resolve_doc_id(documents, args.doc_id)
    print(index.term_frequency(doc_id, tokenize_term(args.term)))


def idf_command(args: argparse.Namespace) -> None:
    index, _ = _loaded_index(args)
    print(f"IDF of '{args.term}': {index.idf(tokenize_term(args.term)):.2f}")


def tfidf_command(args: argparse.Namespace) -> None:
    index, documents = _loaded_index(args)
    doc_id = resolve_doc_id(documents, args.doc_id)
    score = index.tf_idf(doc_id, tokenize_term(args.term))
    print(f"TF-IDF of '{args.term}' in document {doc_id}: {score:.2f}")


def bm25_idf_command(args: argparse.Namespace) -> None:
    index, _ = _loaded_index(args)
    print(f"BM25 IDF of '{args.term}': {index.bm25_idf(tokenize_term(args.term)):.2f}")


def bm25_tf_command(args: argparse.Namespace) -> None:
    index, documents = _loaded_index(args)
    doc_id = resolve_doc_id(documents, args.doc_id)
    score = index.bm25_tf(doc_id, tokenize_term(args.term), args.k1, args.b)
    print(f"BM25 TF of '{args.term}' in document {doc_id}: {score:.2f}")
