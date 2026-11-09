"""Loading and validating the document collection.

A dataset is a JSON file containing a list of records, or an object with the
list under a ``"documents"`` or ``"movies"`` key. Every record needs an
``id``, a ``title`` and a ``description``.
"""

import hashlib
import json
import os
from pathlib import Path

from rag_search import config
from rag_search.errors import DatasetError

REQUIRED_FIELDS = ("id", "title", "description")
_LIST_KEYS = ("documents", "movies")

Document = dict


def resolve_dataset_path(path: str | os.PathLike | None = None) -> Path:
    """Return the dataset path from an explicit value, RAG_DATASET, or the default."""
    if path:
        return Path(path)
    env_path = os.environ.get("RAG_DATASET")
    return Path(env_path) if env_path else config.DEFAULT_DATASET


def load_dataset(path: str | os.PathLike) -> list[Document]:
    """Read and validate a dataset file."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Dataset not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path} is not valid JSON: {exc}") from exc
    return validate_documents(_extract_records(raw, path))


def validate_documents(records: list) -> list[Document]:
    """Check that records have the required fields and unique ids."""
    seen_ids = set()
    documents = []
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise DatasetError(f"Record {position} is not a JSON object")
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            raise DatasetError(f"Record {position} is missing: {', '.join(missing)}")
        doc_id = record["id"]
        if not isinstance(doc_id, (int, str)) or isinstance(doc_id, bool):
            raise DatasetError(f"Record {position} has an id that is not an integer or string")
        if doc_id in seen_ids:
            raise DatasetError(f"Duplicate id: {doc_id!r}")
        seen_ids.add(doc_id)
        documents.append(
            {
                **record,
                "title": str(record["title"] or ""),
                "description": str(record["description"] or ""),
            }
        )
    if not documents:
        raise DatasetError("The dataset contains no records")
    return documents


def cache_dir_for(path: str | os.PathLike) -> Path:
    """Return a cache directory unique to the dataset's name and contents.

    Changing the file's contents changes the directory, so cached indexes
    and embeddings from an older version are never reused.
    """
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:10]
    return config.CACHE_ROOT / f"{path.stem}-{digest}"


def _extract_records(raw: object, path: Path) -> list:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in _LIST_KEYS:
            if isinstance(raw.get(key), list):
                return raw[key]
    raise DatasetError(
        f"{path} must contain a list of records or an object with a "
        f"{' or '.join(repr(k) for k in _LIST_KEYS)} list"
    )
