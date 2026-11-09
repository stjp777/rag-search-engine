"""`rag-search generate ...` commands: answers grounded in search results."""

import argparse
from collections.abc import Callable

from rag_search import config
from rag_search.cli.common import load_documents, positive_int

# command name -> (help text, name of the function in rag_search.generation, output heading)
_COMMANDS = {
    "answer": ("Answer a query from the top results", "generate_answer", "Answer"),
    "summarize": ("Summarise the top results", "summarize_results", "Summary"),
    "citations": ("Answer with numbered source citations", "answer_with_citations", "Answer"),
    "question": ("Answer a question conversationally", "answer_question", "Answer"),
}


def register(subparsers) -> None:
    parser = subparsers.add_parser("generate", help="LLM answers grounded in search results")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, (help_text, _, _) in _COMMANDS.items():
        command = commands.add_parser(name, help=help_text)
        command.add_argument("query")
        command.add_argument("--limit", type=positive_int, default=5)
        command.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    from rag_search import generation
    from rag_search.hybrid import HybridSearch

    _, function_name, heading = _COMMANDS[args.command]
    generate: Callable[[str, list[dict]], str] = getattr(generation, function_name)

    documents, cache_dir = load_documents(args)
    results = HybridSearch(documents, cache_dir).rrf_search(args.query, config.RRF_K, args.limit)

    print("Search results:")
    for result in results:
        print(f"- {result['title']}")
    print(f"\n{heading}:")
    print(generate(args.query, results))
