# RAG Search Engine

A command-line search engine that combines BM25 keyword search with semantic (embedding) search, and can rerank results or generate answers with an LLM. It works on any JSON dataset of titled text records; a 150-movie sample is included.

## Features

- **Keyword search**: inverted index with stemming, stopword removal, TF-IDF and BM25
- **Semantic search**: sentence-transformer embeddings, over whole documents or sentence chunks
- **Hybrid search**: weighted score blending and Reciprocal Rank Fusion (RRF)
- **Query enhancement**: spelling correction, rewriting and expansion with an LLM
- **Reranking**: LLM scoring, LLM batch ranking, or a local cross-encoder
- **Answer generation**: answers, summaries, cited answers and Q&A from the top results
- **Image search**: match an image to documents with CLIP
- **Evaluation**: precision, recall and F1 against a golden dataset

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/stjp777/rag-search-engine.git
cd rag-search-engine
uv sync
```

LLM features (query enhancement, LLM reranking, answer generation) use [OpenRouter](https://openrouter.ai). Copy `.env.example` to `.env` and add your key:

```bash
cp .env.example .env
```

Keyword, semantic and hybrid search, cross-encoder reranking, image search and evaluation work without a key.

## Usage

Commands are grouped as `rag-search <group> <command>`. Run `rag-search --help` or `rag-search <group> --help` for details. The first run of a semantic or hybrid command downloads the embedding model and builds a cache in `cache/`.

```bash
# Keyword search (BM25)
uv run rag-search keyword build
uv run rag-search keyword search "bear marmalade london" --limit 5

# Semantic search
uv run rag-search semantic search "a teddy bear that comes to life"
uv run rag-search semantic search-chunks "a teddy bear that comes to life"

# Hybrid search
uv run rag-search hybrid weighted "talking teddy bear comedy" --alpha 0.5
uv run rag-search hybrid rrf "cute british bear marmalade" --limit 5

# Hybrid search with query enhancement, reranking and LLM grading
uv run rag-search hybrid rrf "scary movie with bear" --enhance rewrite --rerank batch --evaluate

# Generate answers from the top results
uv run rag-search generate answer "Which movies feature a talking bear?"
uv run rag-search generate citations "Which movies feature a talking bear?"

# Search with an image
uv run rag-search image search data/paddington.jpeg

# Evaluate retrieval quality
uv run rag-search evaluate --limit 5
```

Hybrid search can be limited to documents with a matching field value. A list-valued field matches if any element does, and comparison ignores case:

```bash
uv run rag-search --dataset books.json hybrid rrf "a dragon guarding gold" --filter genre=fantasy --filter year=2001
```

Rerank methods: `individual` (LLM scores each result, several requests at a time), `batch` (LLM orders all results in one request), `cross-encoder` (local model).

## Using your own dataset

Point the tool at any JSON file with `--dataset` or the `RAG_DATASET` environment variable:

```bash
uv run rag-search --dataset path/to/books.json hybrid rrf "a dragon guarding gold"
```

The file must be a list of records, or an object with the list under a `"documents"` or `"movies"` key. Each record needs these fields. Extra fields are kept and can be used for filtering.

```json
[
  {"id": 1, "title": "Dragon Hoard", "description": "A dragon guards a mountain of gold."},
  {"id": 2, "title": "Ocean Crossing", "description": "A sailor crosses the Atlantic alone."}
]
```

`id` is an integer or string and must be unique. Indexes and embeddings are cached per dataset, keyed by file name and contents, so changing the file rebuilds them automatically.

The `evaluate` command reads test cases from `data/golden_dataset.json` (or `--golden PATH`):

```json
{"test_cases": [{"query": "talking teddy bear comedy", "relevant_docs": ["Ted", "Ted 2"]}]}
```

`relevant_docs` lists titles. Titles that are not in the current dataset are ignored, and cases with none are skipped. The bundled golden set was written for the full 5,000-movie collection; the sample includes most of its relevant titles.

## Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | API key for LLM features | none |
| `RAG_DATASET` | Dataset file | `data/sample_movies.json` |
| `RAG_CACHE_DIR` | Cache directory | `cache` |
| `RAG_LLM_MODEL` | Model name sent to OpenRouter | `nvidia/nemotron-3.5-super-vl:free` |
| `RAG_SERVICE_NAME` | Service name used in generation prompts | `Webflyx` |

The default model is a free OpenRouter model, which can be rate-limited or removed. Set `RAG_LLM_MODEL` to any OpenRouter model ID if it stops working. Image query rewriting needs a vision-capable model.

Models, BM25 parameters, chunk sizes and other constants are in `src/rag_search/config.py`.

## Project layout

```
src/rag_search/
  cli/              command-line interface, one module per command group
  keyword.py        tokenizer, inverted index, TF-IDF, BM25
  semantic.py       document and chunk embeddings
  chunking.py       sentence and word chunking
  hybrid.py         weighted and RRF search
  fusion.py         score normalisation and fusion helpers
  rerank.py         LLM and cross-encoder reranking
  query_enhancement.py
  generation.py     answer, summary, citation and Q&A prompts
  evaluation.py     metrics and LLM grading
  multimodal.py     CLIP image search
  llm.py            OpenRouter client with retries
  dataset.py        dataset loading and validation
  config.py
tests/
data/               sample dataset, golden dataset, example image
```

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```

The same checks run in GitHub Actions on every push and pull request.

## Notes

- Embeddings are compared with brute-force cosine similarity, which is fine for tens of thousands of chunks but not for millions.

## License

MIT
