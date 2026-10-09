"""Retrieval quality on a labeled question set: recall@k and mean reciprocal rank.

Each labeled question names the source file and a fragment of the section heading
that answers it. A retrieved chunk counts as relevant when both match. Measuring
retrieval separately from generation tells you whether a bad answer came from
the search or from the model.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inferscale.rag.corpus import Chunk
from inferscale.rag.store import DocumentStore


@dataclass(frozen=True)
class LabeledQuestion:
    question: str
    source: str
    section: str

    def matches(self, chunk: Chunk) -> bool:
        return chunk.source == self.source and self.section.lower() in chunk.heading.lower()


def load_questions(path: str | Path) -> list[LabeledQuestion]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [LabeledQuestion(**json.loads(line)) for line in lines if line.strip()]


def evaluate(
    store: DocumentStore, questions: Sequence[LabeledQuestion], ks: Sequence[int] = (1, 3, 5)
) -> dict[str, Any]:
    depth = max(ks)
    ranks: list[int | None] = []
    latencies: list[float] = []
    misses: list[str] = []
    for q in questions:
        start = time.perf_counter()
        hits = store.search(q.question, depth)
        latencies.append((time.perf_counter() - start) * 1000)
        rank = next((i for i, h in enumerate(hits, 1) if q.matches(h.chunk)), None)
        ranks.append(rank)
        if rank is None:
            misses.append(q.question)

    n = len(questions)
    latencies.sort()
    return {
        "questions": n,
        "corpus": store.info(),
        **{f"recall@{k}": round(sum(1 for r in ranks if r and r <= k) / n, 3) for k in ks},
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
        "search_ms_p50": round(latencies[n // 2], 3),
        "search_ms_max": round(latencies[-1], 3),
        "misses": misses,
    }
