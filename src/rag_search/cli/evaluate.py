"""`rag-search evaluate`: precision, recall and F1 against a golden dataset."""

import argparse
from pathlib import Path

from rag_search import config
from rag_search.cli.common import load_documents, positive_int


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "evaluate", help="Measure retrieval quality against a golden dataset"
    )
    parser.add_argument("--golden", type=Path, default=config.DEFAULT_GOLDEN_DATASET)
    parser.add_argument(
        "--limit", type=positive_int, default=5, help="k for precision@k and recall@k"
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    from rag_search.evaluation import load_golden_dataset, precision_recall_f1
    from rag_search.hybrid import HybridSearch

    documents, cache_dir = load_documents(args)
    test_cases = load_golden_dataset(args.golden)
    search = HybridSearch(documents, cache_dir)
    titles = {doc["title"] for doc in documents}

    print(f"k={args.limit}\n")
    evaluated = 0
    for case in test_cases:
        # Only titles present in the current dataset can be retrieved.
        relevant = [title for title in case["relevant_docs"] if title in titles]
        if not relevant:
            print(f"- Query: {case['query']}\n  - skipped: no relevant documents in this dataset\n")
            continue
        results = search.rrf_search(case["query"], config.RRF_K, args.limit)
        retrieved = [result["title"] for result in results]
        precision, recall, f1 = precision_recall_f1(retrieved, relevant)
        evaluated += 1

        print(f"- Query: {case['query']}")
        print(f"  - Precision@{args.limit}: {precision:.4f}")
        print(f"  - Recall@{args.limit}: {recall:.4f}")
        print(f"  - F1: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved)}")
        print(f"  - Relevant: {', '.join(relevant)}")
        print()

    if not evaluated:
        print("No test cases matched this dataset.")
