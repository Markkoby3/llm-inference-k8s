"""Drive load at each concurrency level and collect per-request results.

Two load models:

* Closed loop (default): exactly ``concurrency`` requests are in flight; each
  worker sends its next request as soon as the previous one finishes. This finds
  peak throughput at a given level of parallelism.
* Open loop (``request_rate`` set): requests arrive as a Poisson process at the
  given rate, capped at ``concurrency`` in flight. This models real traffic, where
  arrivals do not wait for the server, and exposes queueing in tail latency.
"""

from __future__ import annotations

import asyncio
import itertools
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from inferscale.bench.client import RequestResult, send_streaming
from inferscale.bench.stats import summarize


@dataclass
class RunConfig:
    max_tokens: int = 256
    temperature: float = 0.0
    ignore_eos: bool = True
    backend: str | None = None
    request_rate: float | None = None
    seed: int = 0
    extra_headers: dict[str, str] = field(default_factory=dict)

    def headers(self) -> dict[str, str]:
        headers = dict(self.extra_headers)
        if self.backend:
            headers["X-InferScale-Backend"] = self.backend
        return headers

    def payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            # Fixed output length so every backend does the same amount of decode work.
            "ignore_eos": self.ignore_eos,
        }


async def _closed_loop(
    client: httpx.AsyncClient, prompts: list, n: int, concurrency: int, cfg: RunConfig
) -> list[RequestResult]:
    counter = itertools.count()
    results: list[RequestResult] = []

    async def worker() -> None:
        while (i := next(counter)) < n:
            results.append(
                await send_streaming(client, cfg.payload(prompts[i % len(prompts)]), cfg.headers())
            )

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return results


async def _open_loop(
    client: httpx.AsyncClient, prompts: list, n: int, concurrency: int, cfg: RunConfig
) -> list[RequestResult]:
    assert cfg.request_rate
    rng = random.Random(cfg.seed)
    slots = asyncio.Semaphore(concurrency)

    async def one(i: int) -> RequestResult:
        async with slots:
            return await send_streaming(
                client, cfg.payload(prompts[i % len(prompts)]), cfg.headers()
            )

    tasks = []
    for i in range(n):
        tasks.append(asyncio.create_task(one(i)))
        await asyncio.sleep(rng.expovariate(cfg.request_rate))
    return list(await asyncio.gather(*tasks))


async def run_level(
    client: httpx.AsyncClient,
    prompts: list,
    concurrency: int,
    num_requests: int,
    cfg: RunConfig,
    warmup: int = 0,
) -> dict[str, Any]:
    if warmup:
        # Warm CUDA graphs, the KV cache allocator and connection pools; discard timings.
        await _closed_loop(client, prompts, warmup, min(concurrency, warmup), cfg)

    start = time.perf_counter()
    if cfg.request_rate:
        results = await _open_loop(client, prompts, num_requests, concurrency, cfg)
    else:
        results = await _closed_loop(client, prompts, num_requests, concurrency, cfg)
    wall = time.perf_counter() - start

    summary = summarize(results, wall, concurrency)
    if cfg.request_rate:
        summary["request_rate"] = cfg.request_rate
    return summary


def make_client(base_url: str, concurrency: int, timeout_s: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(timeout_s, connect=10.0),
        limits=httpx.Limits(
            max_connections=concurrency + 8, max_keepalive_connections=concurrency + 8
        ),
    )
