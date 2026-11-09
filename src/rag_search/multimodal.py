"""Search documents with an image, using CLIP embeddings."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from rag_search.config import CLIP_MODEL
from rag_search.semantic import cosine_scores

TEXT_EMBEDDINGS_FILENAME = "clip_text_embeddings.npy"


class MultiModalSearch:
    """Matches an image against document text in a shared CLIP embedding space."""

    def __init__(
        self,
        documents: Sequence[dict] = (),
        cache_dir: Path | None = None,
        model_name: str = CLIP_MODEL,
    ) -> None:
        self.documents = list(documents)
        self.cache_dir = cache_dir
        self.model = SentenceTransformer(model_name)
        self._text_embeddings: np.ndarray | None = None

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        with Image.open(path) as image:
            return self.model.encode([image.convert("RGB")])[0]

    def text_embeddings(self) -> np.ndarray:
        """Embed every document once and cache the result on disk."""
        if self._text_embeddings is not None:
            return self._text_embeddings
        cache_file = self.cache_dir / TEXT_EMBEDDINGS_FILENAME if self.cache_dir else None
        if cache_file and cache_file.is_file():
            cached = np.load(cache_file)
            if len(cached) == len(self.documents):
                self._text_embeddings = cached
                return cached
        texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self._text_embeddings = self.model.encode(texts, show_progress_bar=True)
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, self._text_embeddings)
        return self._text_embeddings

    def search_with_image(self, image_path: str | Path, limit: int = 5) -> list[dict]:
        scores = cosine_scores(self.text_embeddings(), self.embed_image(image_path))
        top = np.argsort(-scores)[:limit]
        return [
            {
                "id": self.documents[i]["id"],
                "title": self.documents[i]["title"],
                "description": self.documents[i]["description"],
                "similarity": float(scores[i]),
            }
            for i in top
        ]
