"""A single streamed request, timed from send to the last byte."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class RequestResult:
    ok: bool
    latency_s: float
    ttft_s: float | None = None
    output_tokens: int = 0
    status: int | None = None
    error: str | None = None

    @property
    def tpot_s(self) -> float | None:
        """Mean time per output token after the first (inter-token latency)."""
        if self.ttft_s is None or self.output_tokens < 2:
            return None
        return (self.latency_s - self.ttft_s) / (self.output_tokens - 1)


async def send_streaming(
    client: httpx.AsyncClient,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> RequestResult:
    """Send one streaming chat completion and time it.

    Token count comes from the server's usage chunk when present (exact), and
    falls back to counting content chunks, which is one token per chunk on vLLM
    and Triton.
    """
    start = time.perf_counter()
    ttft: float | None = None
    chunks = 0
    usage_tokens: int | None = None

    def failed(status: int | None, error: str) -> RequestResult:
        return RequestResult(False, time.perf_counter() - start, ttft, chunks, status, error)

    try:
        async with client.stream(
            "POST", "/v1/chat/completions", json=payload, headers=headers
        ) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", errors="replace")[:200]
                return failed(resp.status_code, f"HTTP {resp.status_code}: {body}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if "error" in event:
                    return failed(200, f"stream error: {event['error']}")
                if event.get("usage"):
                    usage_tokens = event["usage"].get("completion_tokens")
                for choice in event.get("choices", []):
                    if (choice.get("delta") or {}).get("content"):
                        if ttft is None:
                            ttft = time.perf_counter() - start
                        chunks += 1
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        return failed(None, f"{type(exc).__name__}: {exc}")

    tokens = usage_tokens if usage_tokens is not None else chunks
    return RequestResult(True, time.perf_counter() - start, ttft, tokens, 200)
