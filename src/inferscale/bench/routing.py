"""Compare replica routing policies in a prefix-cache simulation.

Runs the gateway's real ``ReplicaPool`` over mock replicas that model an
engine's prefix cache: an LRU of prompt prefixes per replica, where a miss pays
prefill time for the whole prompt and a hit pays only for the uncached suffix.
The workload is agent-style traffic: many requests share one of a set of long
system prompts, chosen with Zipf popularity, followed by a short unique question.

This isolates the routing decision. It is a model, not a GPU measurement: real
hit rates depend on the engine's KV-cache size, block size and eviction policy.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any

from inferscale.backends.base import GenerationParams
from inferscale.backends.mock import MockBackend
from inferscale.backends.pool import POLICIES, Replica, ReplicaPool
from inferscale.bench.stats import percentile
from inferscale.schemas import ChatMessage


@dataclass(frozen=True)
class Workload:
    replicas: int = 4
    prefixes: int = 64
    zipf_s: float = 1.1
    prefix_chars: int = 3000
    cache_per_replica: int = 8
    prefill_ms_per_kchar: float = 25.0
    base_ttft_ms: float = 10.0
    # Batch slots per replica: offered load above this queues on the replica.
    slots_per_replica: int = 8
    requests: int = 2000
    concurrency: int = 28
    seed: int = 0


def _system_prompts(w: Workload) -> list[str]:
    rng = random.Random(w.seed)
    words = "tool search answer cite passage plan step json schema retry budget".split()
    return [
        f"Agent profile {i}. " + " ".join(rng.choice(words) for _ in range(w.prefix_chars // 6))
        for i in range(w.prefixes)
    ][: w.prefixes]


def _requests(w: Workload) -> list[list[ChatMessage]]:
    rng = random.Random(w.seed + 1)
    systems = _system_prompts(w)
    weights = [1 / (rank + 1) ** w.zipf_s for rank in range(w.prefixes)]
    chosen = rng.choices(range(w.prefixes), weights=weights, k=w.requests)
    return [
        [
            ChatMessage(role="system", content=systems[k]),
            ChatMessage(role="user", content=f"Question {i}: summarize step {i % 17}."),
        ]
        for i, k in enumerate(chosen)
    ]


async def run_policy(policy: str, w: Workload) -> dict[str, Any]:
    mocks = [
        MockBackend(
            ttft_ms=w.base_ttft_ms,
            inter_token_ms=0,
            prefix_cache_size=w.cache_per_replica,
            prefill_ms_per_kchar=w.prefill_ms_per_kchar,
            prefix_chars=1024,
            max_concurrency=w.slots_per_replica,
        )
        for _ in range(w.replicas)
    ]
    load_factor = 1.25
    if policy == "prefix_affinity_unbounded":
        policy, load_factor = "prefix_affinity", 1e9
    pool = ReplicaPool(
        "sim",
        [Replica(f"r{i}", m) for i, m in enumerate(mocks)],
        policy=policy,
        load_factor=load_factor,
    )
    requests = _requests(w)
    params = GenerationParams(max_tokens=1, temperature=0.0, top_p=1.0)
    latencies: list[float] = []
    queue = iter(requests)

    async def worker() -> None:
        for messages in queue:
            start = time.perf_counter()
            await pool.complete(messages, params)
            latencies.append((time.perf_counter() - start) * 1000)

    start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(w.concurrency)))
    wall = time.perf_counter() - start

    hits = sum(m.cache_hits for m in mocks)
    misses = sum(m.cache_misses for m in mocks)
    served = [r.requests for r in pool.replicas]
    return {
        "policy": policy if load_factor < 1e9 else "prefix_affinity_unbounded",
        "cache_hit_rate": round(hits / (hits + misses), 3),
        "ttft_ms_p50": round(percentile(latencies, 50), 1),
        "ttft_ms_p95": round(percentile(latencies, 95), 1),
        "requests_per_s": round(len(latencies) / wall, 1),
        "load_imbalance": round(max(served) / (sum(served) / len(served)), 2),
    }


async def compare(w: Workload) -> list[dict[str, Any]]:
    policies = [*POLICIES, "prefix_affinity_unbounded"]
    return [await run_policy(policy, w) for policy in policies]


def markdown(rows: list[dict[str, Any]], w: Workload) -> str:
    lines = [
        f"Simulation: {w.replicas} replicas, {w.prefixes} system prompts "
        f"(~{w.prefix_chars} chars, Zipf s={w.zipf_s}), prefix cache of {w.cache_per_replica} "
        f"per replica, {w.slots_per_replica} batch slots per replica, "
        f"{w.requests} requests at concurrency {w.concurrency}.",
        "",
        "| Policy | Cache hit rate | TTFT p50 (ms) | TTFT p95 (ms) | Req/s | Load imbalance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['policy']} | {r['cache_hit_rate']:.0%} | {r['ttft_ms_p50']} | "
            f"{r['ttft_ms_p95']} | {r['requests_per_s']} | {r['load_imbalance']}x |"
        )
    return "\n".join(lines) + "\n"
