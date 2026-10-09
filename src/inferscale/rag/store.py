"""A searchable document store: chunks, their embeddings and the vector index."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inferscale.rag.corpus import Chunk, default_corpus_paths, load_chunks
from inferscale.rag.embed import Embedder, HashingEmbedder, make_embedder
from inferscale.rag.index import make_index


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class DocumentStore:
    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder, index_kind: str = "auto"):
        if not chunks:
            raise ValueError("cannot build a document store with no chunks")
        self.chunks = list(chunks)
        self.embedder = embedder
        texts = [c.text for c in self.chunks]
        if isinstance(embedder, HashingEmbedder):
            embedder.fit(texts)
        self.index = make_index(embedder.dim, index_kind)
        self.index.add(embedder.embed(texts))

    @classmethod
    def from_paths(
        cls,
        paths: Iterable[str | Path] | None = None,
        embedder: str = "hashing",
        index_kind: str = "auto",
        max_words: int = 160,
    ) -> DocumentStore:
        chunks = load_chunks(paths or default_corpus_paths(), max_words=max_words)
        return cls(chunks, make_embedder(embedder), index_kind)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        scores, ids = self.index.search(self.embedder.embed([query]), k)
        return [
            Hit(self.chunks[int(i)], float(s))
            for s, i in zip(scores[0], ids[0], strict=True)
            if i >= 0 and s > 0
        ]

    def info(self) -> dict[str, Any]:
        return {
            "chunks": len(self.chunks),
            "sources": len({c.source for c in self.chunks}),
            "embedder": self.embedder.name,
            "index": self.index.kind,
        }
