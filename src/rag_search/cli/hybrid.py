"""`rag-search hybrid ...` commands."""

import argparse

from rag_search import config
from rag_search.cli.common import load_documents, positive_int, snippet, unit_float
from rag_search.fusion import normalize
from rag_search.query_enhancement import METHODS as ENHANCE_METHODS
from rag_search.query_enhancement import enhance_query
from rag_search.rerank import METHODS as RERANK_METHODS


def register(subparsers) -> None:
    parser = subparsers.add_parser("hybrid", help="Combined keyword and semantic search")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    norm = commands.add_parser("normalize", help="Min-max normalise a list of scores")
    norm.add_argument("scores", nargs="+", type=float)
    norm.set_defaults(func=normalize_command)

    weighted = commands.add_parser("weighted", help="Weighted blend of keyword and semantic scores")
    weighted.add_argument("query")
    weighted.add_argument("--alpha", type=unit_float, default=0.5, help="keyword weight, 0-1")
    weighted.add_argument("--limit", type=positive_int, default=5)
    weighted.set_defaults(func=weighted_command)

    rrf = commands.add_parser("rrf", help="Reciprocal Rank Fusion")
    rrf.add_argument("query")
    rrf.add_argument("-k", type=positive_int, default=config.RRF_K)
    rrf.add_argument("--limit", type=positive_int, default=5)
    rrf.add_argument(
        "--enhance", choices=ENHANCE_METHODS, help="improve the query with an LLM first"
    )
    rrf.add_argument("--rerank", choices=RERANK_METHODS, help="rerank the top results")
    rrf.add_argument("--evaluate", action="store_true", help="grade the results with an LLM")
    rrf.set_defaults(func=rrf_command)


def normalize_command(args: argparse.Namespace) -> None:
    for score in normalize(args.scores):
        print(f"* {score:.4f}")


def weighted_command(args: argparse.Namespace) -> None:
    from rag_search.hybrid import HybridSearch

    documents, cache_dir = load_documents(args)
    results = HybridSearch(documents, cache_dir).weighted_search(args.query, args.alpha, args.limit)
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result['title']}")
        print(f"   Hybrid score: {result['hybrid_score']:.3f}")
        print(f"   BM25: {result['keyword_score']:.3f}, semantic: {result['semantic_score']:.3f}")
        print(f"   {snippet(result['description'])}")
        print()


def rrf_command(args: argparse.Namespace) -> None:
    from rag_search.evaluation import grade_results
    from rag_search.hybrid import HybridSearch
    from rag_search.rerank import rerank

    query = args.query
    if args.enhance:
        enhanced = enhance_query(query, args.enhance)
        if enhanced != query:
            print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced}'\n")
            query = enhanced

    fetch = args.limit * config.RERANK_OVERSAMPLE if args.rerank else args.limit
    documents, cache_dir = load_documents(args)
    results = HybridSearch(documents, cache_dir).rrf_search(query, args.k, fetch)

    if args.rerank:
        print(f"Reranking the top {len(results)} results ({args.rerank})...\n")
        results = rerank(query, results, args.rerank)
    results = results[: args.limit]

    print(f"Reciprocal Rank Fusion results for '{query}' (k={args.k}):\n")
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result['title']}")
        _print_rerank_score(result)
        print(f"   RRF score: {result['rrf_score']:.3f}")
        print(f"   BM25 rank: {result['keyword_rank']}, semantic rank: {result['semantic_rank']}")
        print(f"   {snippet(result['description'])}")
        print()

    if args.evaluate:
        grades = grade_results(query, results)
        if grades is None:
            print("Evaluation failed: the LLM did not return valid grades.")
        else:
            for rank, (result, grade) in enumerate(zip(results, grades, strict=True), start=1):
                print(f"{rank}. {result['title']}: {grade}/3")


def _print_rerank_score(result: dict) -> None:
    if "rerank_score" in result:
        score = result["rerank_score"]
        print(f"   Rerank score: {'n/a' if score is None else f'{score:.1f}/10'}")
    elif "rerank_rank" in result:
        print(f"   Rerank position: {result['rerank_rank']}")
    elif "cross_encoder_score" in result:
        print(f"   Cross-encoder score: {result['cross_encoder_score']:.3f}")
