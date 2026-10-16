"""Build (or rebuild) the knowledge base from data/documents into storage/.

Usage:  python scripts/build_index.py [--docs data/documents] [--chunk-size 800] [--overlap 150]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag import KnowledgeBase, load_directory  # noqa: E402
from rag.embeddings import get_embedder  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--docs", default=str(ROOT / "data" / "documents"))
    p.add_argument("--out", default=str(ROOT / "storage"))
    p.add_argument("--chunk-size", type=int, default=800)
    p.add_argument("--overlap", type=int, default=150)
    p.add_argument("--embedder", default="auto", choices=["auto", "sentence-transformers", "tfidf"])
    args = p.parse_args()

    docs = load_directory(args.docs)
    print(f"Extracted + cleaned {len(docs)} documents")
    kb = KnowledgeBase(embedder=get_embedder(args.embedder))
    kb.add_documents(docs, chunk_size=args.chunk_size, overlap=args.overlap)
    kb.save(args.out)
    print(f"Embedded {len(kb.chunks)} chunks with {kb.embedder.name} -> {args.out}")
    for d in kb.list_sources():
        print(f"  - {d['source']}: {d['chunks']} chunks, {d['chars']} chars")


if __name__ == "__main__":
    main()
