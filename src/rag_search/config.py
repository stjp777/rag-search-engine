"""Default paths, model names and tuning constants.

A few values can be overridden with environment variables:

    RAG_DATASET       path to the dataset JSON file
    RAG_CACHE_DIR     directory for cached indexes and embeddings
    RAG_LLM_MODEL     model name sent to the OpenRouter API
    RAG_SERVICE_NAME  service name used in generation prompts
"""

import os
from pathlib import Path

# Paths (relative paths are resolved against the current working directory).
DEFAULT_DATASET = Path("data/sample_movies.json")
DEFAULT_GOLDEN_DATASET = Path("data/golden_dataset.json")
CACHE_ROOT = Path(os.environ.get("RAG_CACHE_DIR", "cache"))

# Models.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CLIP_MODEL = "clip-ViT-B-32"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "nvidia/nemotron-3.5-super-vl:free")
LLM_BASE_URL = "https://openrouter.ai/api/v1"
SERVICE_NAME = os.environ.get("RAG_SERVICE_NAME", "Webflyx")

# Keyword search (BM25).
BM25_K1 = 1.5
BM25_B = 0.75

# Semantic chunking: sentences per chunk and sentences shared between chunks.
CHUNK_SENTENCES = 4
CHUNK_OVERLAP = 1

# Hybrid search.
RRF_K = 60
CANDIDATE_MULTIPLIER = 50  # candidates fetched per retriever = limit * multiplier
RERANK_OVERSAMPLE = 5  # fetch limit * oversample results before reranking

# LLM calls.
RERANK_WORKERS = 4  # concurrent requests for individual reranking
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds
LLM_DOCUMENT_CHARS = 800  # description characters included in each LLM prompt
DISPLAY_CHARS = 100  # description characters shown in CLI output
