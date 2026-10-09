"""Load test for the multi-step agent endpoint (/v1/agent/chat).

Reports end-to-end latency and its breakdown by stage (LLM planning, vector
retrieval, LLM generation), so you can see which stage dominates and how each
one degrades as concurrency rises.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from typing import Any

import httpx

from inferscale.bench.stats import PERCENTILES, percentile

STAGES = ("plan", "retrieve", "generate", "total")


async def _one(
    client: httpx.AsyncClient, question: str, headers: dict[str, str], max_tokens: int
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        response = await client.post(
            "/v1/agent/chat",
            json={"messages": [{"role": "user", "content": question}], "max_tokens": max_tokens},
            headers=headers,
        )
    except httpx.HTTPError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    client_ms = (time.perf_counter() - start) * 1000
    if response.status_code != 200:
        return {"ok": False, "error": f"HTTP {response.status_code}: {response.text[:200]}"}
    body = response.json()
    return {
        "ok": True,
        "client_ms": client_ms,
        "timings": body["timings_ms"],
        "steps": len(body["steps"]),
        "searches": sum(1 for s in body["steps"] if "search" in s["action"]),
        "cited": any(c["cited"] for c in body["citations"]),
    }


def _dist(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {f"p{p}": None for p in PERCENTILES}
    return {f"p{p}": round(percentile(values, p), 2) for p in PERCENTILES}


async def run_agent_level(
    client: httpx.AsyncClient,
    questions: list[str],
    concurrency: int,
    num_requests: int,
    backend: str | None = None,
    max_tokens: int = 256,
) -> dict[str, Any]:
    headers = {"X-InferScale-Backend": backend} if backend else {}
    counter = itertools.count()
    results: list[dict[str, Any]] = []

    async def worker() -> None:
        while (i := next(counter)) < num_requests:
            results.append(await _one(client, questions[i % len(questions)], headers, max_tokens))

    start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall = time.perf_counter() - start

    ok = [r for r in results if r["ok"]]
    summary: dict[str, Any] = {
        "concurrency": concurrency,
        "requests": len(results),
        "errors": len(results) - len(ok),
        "requests_per_s": round(len(ok) / wall, 3) if wall else 0.0,
        "mean_steps": round(sum(r["steps"] for r in ok) / len(ok), 2) if ok else None,
        "mean_searches": round(sum(r["searches"] for r in ok) / len(ok), 2) if ok else None,
        "answers_with_citation": round(sum(r["cited"] for r in ok) / len(ok), 3) if ok else None,
        "client_latency_ms": _dist([r["client_ms"] for r in ok]),
        "error_samples": sorted({r["error"] for r in results if not r["ok"]})[:5],
    }
    for stage in STAGES:
        summary[f"{stage}_ms"] = _dist([r["timings"][stage] for r in ok])
    return summary


def agent_markdown(run: dict[str, Any]) -> str:
    meta = run["meta"]
    lines = [
        f"### Agent: {meta['label']}",
        "",
        f"Backend `{meta.get('backend') or 'default'}` · model `{meta.get('model', '?')}` · "
        f"GPU {meta.get('gpu') or 'not recorded'} · {meta['timestamp']}",
        "",
        "| Concurrency | Req/s | Steps | Plan p50 (ms) | Retrieve p50 (ms) | Generate p50 (ms) "
        "| Total p50 (ms) | Total p95 (ms) | Cited | Errors |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lv in run["levels"]:
        cited = lv["answers_with_citation"]
        lines.append(
            f"| {lv['concurrency']} | {lv['requests_per_s']} | {lv['mean_steps']} "
            f"| {lv['plan_ms']['p50']} | {lv['retrieve_ms']['p50']} | {lv['generate_ms']['p50']} "
            f"| {lv['total_ms']['p50']} | {lv['total_ms']['p95']} "
            f"| {'-' if cited is None else f'{cited:.0%}'} | {lv['errors']}/{lv['requests']} |"
        )
    return "\n".join(lines) + "\n"
