"""Text embedders. All return float32 rows with unit L2 norm, so inner product is
cosine similarity.

* ``HashingEmbedder`` (default): sparse lexical features (unigrams + bigrams)
  hashed into a fixed-size vector and weighted by IDF fitted on the corpus. No
  model download, deterministic, milliseconds per query: the gateway image stays
  small and CI needs no network.
* ``SentenceTransformerEmbedder``: dense semantic embeddings (``pip install
  inferscale[rag]``). Better on paraphrased questions, at the cost of a model
  download and a heavier image.
"""

from __future__ import annotations

import math
import re
import zlib
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9][a-z0-9+.\-]*[a-z0-9]|[a-z0-9]")
_STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from how i in is it its of on or so "
    "that the this to was what when where which who why will with you your".split()
)


_SUFFIXES = ("ing", "ed", "es", "s")


def stem(token: str) -> str:
    """Light suffix stripping so "scaling", "scaled" and "scales" share a feature.

    Measurably improves recall on the labeled question set (docs/agentic-rag.md).
    Deliberately crude: no dictionary, and it never touches short words or tokens
    ending in digits such as "p95".
    """
    for suffix in _SUFFIXES:
        if (
            len(token) > len(suffix) + 3
            and token.endswith(suffix)
            and token[-len(suffix) - 1].isalpha()
        ):
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = 2048):
        self.dim = dim
        self._idf: np.ndarray | None = None

    def _features(self, text: str) -> dict[int, float]:
        tokens = tokenize(text)
        grams = tokens + [f"{a} {b}" for a, b in pairwise(tokens)]
        counts: dict[int, float] = {}
        for gram in grams:
            h = zlib.crc32(gram.encode())
            # The sign bit spreads hash collisions around zero instead of piling them up.
            bucket, sign = h % self.dim, 1.0 if (h >> 31) & 1 else -1.0
            counts[bucket] = counts.get(bucket, 0.0) + sign
        return counts

    def fit(self, texts: Sequence[str]) -> HashingEmbedder:
        df = np.zeros(self.dim, dtype=np.float64)
        for text in texts:
            for bucket in self._features(text):
                df[bucket] += 1
        self._idf = np.log((len(texts) + 1) / (df + 1)) + 1.0
        return self

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            for bucket, count in self._features(text).items():
                # Sublinear term frequency: the 10th mention adds less than the 1st.
                magnitude = 1.0 + math.log(abs(count)) if count else 0.0
                matrix[row, bucket] = math.copysign(magnitude, count)
        if self._idf is not None:
            matrix *= self._idf
        return _normalize(matrix)


class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "sentence-transformers is not installed; pip install 'inferscale[rag]'"
            ) from exc
        self._model = SentenceTransformer(model)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"sentence-transformers:{model.rsplit('/', 1)[-1]}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(list(texts), normalize_embeddings=True, batch_size=32)
        return np.asarray(vectors, dtype=np.float32)


def make_embedder(kind: str) -> HashingEmbedder | SentenceTransformerEmbedder:
    if kind == "hashing":
        return HashingEmbedder()
    if kind.startswith("sentence-transformers"):
        _, _, model = kind.partition(":")
        return SentenceTransformerEmbedder(model or "sentence-transformers/all-MiniLM-L6-v2")
    raise ValueError(f"unknown embedder {kind!r}; use 'hashing' or 'sentence-transformers[:model]'")
