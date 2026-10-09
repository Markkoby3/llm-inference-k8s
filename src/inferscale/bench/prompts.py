"""Prompt sets for benchmarking.

Synthetic prompts are deterministic and all distinct. Distinct prompts matter:
repeating one prompt lets prefix caching skip prefill and inflates throughput
numbers that real traffic would never see.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

_TOPICS = (
    "a distributed key-value store",
    "GPU memory management for transformer inference",
    "a city's public transit network",
    "photosynthesis in desert plants",
    "the history of the printing press",
    "container orchestration on Kubernetes",
    "a beginner's guide to compound interest",
    "how vaccines train the immune system",
)

_FILLER = (
    "system latency throughput request batch cache memory network schedule queue "
    "replica token model kernel tensor bandwidth cluster node service region load "
    "policy budget signal design trade-off failure recovery capacity forecast"
).split()


def synthetic(count: int, input_words: int, seed: int = 0) -> list[list[dict[str, str]]]:
    rng = random.Random(seed)
    prompts = []
    for i in range(count):
        topic = _TOPICS[i % len(_TOPICS)]
        context = " ".join(rng.choice(_FILLER) for _ in range(max(input_words - 20, 0)))
        prompts.append(
            [
                {"role": "system", "content": "You are a concise technical writer."},
                {
                    "role": "user",
                    "content": f"Request {i}. Using these notes as context: {context}. "
                    f"Write a detailed explanation of {topic}.",
                },
            ]
        )
    return prompts


def from_jsonl(path: str | Path) -> list[list[dict[str, str]]]:
    """Load prompts from JSONL: each line has either ``messages`` or ``prompt``."""
    prompts: list[list[dict[str, str]]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record: dict[str, Any] = json.loads(line)
            if "messages" in record:
                prompts.append(record["messages"])
            elif "prompt" in record:
                prompts.append([{"role": "user", "content": record["prompt"]}])
            else:
                raise ValueError(f"{path}:{lineno}: expected a 'messages' or 'prompt' field")
    if not prompts:
        raise ValueError(f"{path}: no prompts found")
    return prompts
