"""Exact inner-product vector search, backed by FAISS when installed.

The corpus is small (hundreds to low thousands of chunks), so exact search
(``IndexFlatIP``) is both fast enough and free of approximate-recall loss. For
millions of vectors, swap in ``IndexHNSWFlat`` or ``IndexIVFFlat``.
"""

from __future__ import annotations

import numpy as np


class NumpyIndex:
    kind = "numpy"

    def __init__(self, dim: int):
        self.dim = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, len(self._vectors))
        scores = queries.astype(np.float32) @ self._vectors.T
        top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(scores, top, axis=1), top

    def __len__(self) -> int:
        return len(self._vectors)


class FaissIndex:
    kind = "faiss"

    def __init__(self, dim: int):
        import faiss

        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray) -> None:
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, self._index.ntotal)
        return self._index.search(np.ascontiguousarray(queries, dtype=np.float32), k)

    def __len__(self) -> int:
        return int(self._index.ntotal)


def faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
    except ImportError:
        return False
    return True


def make_index(dim: int, kind: str = "auto") -> NumpyIndex | FaissIndex:
    if kind == "faiss" or (kind == "auto" and faiss_available()):
        return FaissIndex(dim)
    if kind in {"numpy", "auto"}:
        return NumpyIndex(dim)
    raise ValueError(f"unknown index {kind!r}; use 'auto', 'faiss' or 'numpy'")
