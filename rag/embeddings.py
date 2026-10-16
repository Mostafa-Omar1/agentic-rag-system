"""Embedding backends: sentence-transformers (default) with a TF-IDF fallback."""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    name: str = "base"

    def fit(self, texts: list[str]) -> None:  # only needed by TF-IDF
        pass

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.name = f"st:{model_name}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


class TfidfEmbedder(Embedder):
    """Dependency-light fallback that works fully offline."""

    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        matrix = self.vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


def get_embedder(backend: str = "auto") -> Embedder:
    if backend in ("auto", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder()
        except Exception as exc:  # model download / import failure
            if backend != "auto":
                raise
            log.warning("sentence-transformers unavailable (%s); using TF-IDF", exc)
    return TfidfEmbedder()
