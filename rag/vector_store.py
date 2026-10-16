"""Knowledge base: vector + keyword (BM25) hybrid retrieval, persisted to disk."""
from __future__ import annotations

import json
import math
import pickle
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .embeddings import Embedder, TfidfEmbedder, get_embedder
from .ingest import Chunk, Document, chunk_documents

_TOKEN = re.compile(r"[\w؀-ۿ]+", re.UNICODE)
STOPWORDS = set(
    "a an the of and or to in on for with is are was were be by as at from that this it its "
    "what which who how why when does do did can about into than then there their".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(t) for t in corpus]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = (sum(self.doc_len) / len(self.docs)) if self.docs else 0.0
        self.tf = [Counter(d) for d in self.docs]
        df = Counter(term for d in self.docs for term in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.docs), dtype=np.float32)
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
            for term in q:
                f = tf.get(term)
                if f:
                    out[i] += self.idf[term] * f * (self.k1 + 1) / (f + norm)
        return out


class KnowledgeBase:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or get_embedder()
        self.documents: dict[str, Document] = {}
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None
        self.bm25: BM25 | None = None

    # ------------------------------------------------------------ building
    def add_documents(self, docs: list[Document], chunk_size: int = 800, overlap: int = 150) -> int:
        new_docs = [d for d in docs if d.doc_id not in self.documents and d.text.strip()]
        for d in new_docs:
            self.documents[d.doc_id] = d
        self.chunks.extend(chunk_documents(new_docs, chunk_size, overlap))
        self._reindex()
        return len(new_docs)

    def remove_document(self, doc_id: str) -> None:
        self.documents.pop(doc_id, None)
        self.chunks = [c for c in self.chunks if c.doc_id != doc_id]
        self._reindex()

    def _reindex(self) -> None:
        texts = [c.text for c in self.chunks]
        if not texts:
            self.vectors, self.bm25 = None, None
            return
        if isinstance(self.embedder, TfidfEmbedder):
            self.embedder.fit(texts)
        self.vectors = self.embedder.embed(texts)
        self.bm25 = BM25(texts)

    # ------------------------------------------------------------ querying
    def search(self, query: str, k: int = 5, mode: str = "hybrid", sources: list[str] | None = None,
               alpha: float = 0.6) -> list[dict]:
        """mode: 'semantic' | 'keyword' | 'hybrid'. Optional filter by source filename."""
        if not self.chunks:
            return []
        sem = (self.vectors @ self.embedder.embed([query])[0]) if mode != "keyword" else None
        kw = self.bm25.scores(query) if mode != "semantic" else None

        def _norm(x: np.ndarray) -> np.ndarray:
            lo, hi = float(x.min()), float(x.max())
            return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

        if mode == "semantic":
            scores = sem
        elif mode == "keyword":
            scores = kw
        else:
            scores = alpha * _norm(sem) + (1 - alpha) * _norm(kw)

        mask = np.ones(len(self.chunks), dtype=bool)
        if sources:
            wanted = {s.lower() for s in sources}
            mask = np.array([c.source.lower() in wanted or c.doc_id in wanted for c in self.chunks])
        order = [i for i in np.argsort(-scores) if mask[i]][:k]
        results = []
        for i in order:
            c = self.chunks[i]
            if mode == "keyword" and scores[i] <= 0:
                continue
            results.append({
                "chunk_id": c.chunk_id,
                "source": c.source,
                "position": c.position,
                "score": round(float(scores[i]), 4),
                "text": c.text,
            })
        return results

    def list_sources(self) -> list[dict]:
        counts = Counter(c.doc_id for c in self.chunks)
        return [
            {"doc_id": d.doc_id, "source": d.source, "title": d.metadata.get("title"),
             "chars": len(d.text), "chunks": counts[d.doc_id]}
            for d in self.documents.values()
        ]

    def get_document_text(self, source: str) -> str | None:
        for d in self.documents.values():
            if d.source.lower() == source.lower() or d.doc_id == source:
                return d.text
        return None

    # ------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False, indent=1), encoding="utf-8")
        with open(directory / "store.pkl", "wb") as fh:
            pickle.dump({"documents": self.documents, "chunks": self.chunks,
                         "vectors": self.vectors, "embedder": self.embedder.name,
                         "tfidf": self.embedder if isinstance(self.embedder, TfidfEmbedder) else None}, fh)

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder | None = None) -> "KnowledgeBase":
        with open(Path(directory) / "store.pkl", "rb") as fh:
            state = pickle.load(fh)
        kb = cls(embedder=state["tfidf"] or embedder)
        if kb.embedder.name != state["embedder"]:  # backend changed -> re-embed
            kb.documents, kb.chunks = state["documents"], state["chunks"]
            kb._reindex()
            return kb
        kb.documents, kb.chunks, kb.vectors = state["documents"], state["chunks"], state["vectors"]
        kb.bm25 = BM25([c.text for c in kb.chunks]) if kb.chunks else None
        return kb
