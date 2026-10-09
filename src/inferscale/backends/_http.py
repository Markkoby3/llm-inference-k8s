"""HTTP plumbing shared by the network backends: client setup, SSE parsing, error mapping."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager

import httpx

from inferscale.backends.base import BackendError


def build_client(
    base_url: str, connect_timeout_s: float, request_timeout_s: float
) -> httpx.AsyncClient:
    # One pooled client per backend. The pool must be at least as large as the
    # gateway's admission limit or requests queue on connections instead.
    return httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(request_timeout_s, connect=connect_timeout_s),
        limits=httpx.Limits(max_connections=512, max_keepalive_connections=128),
    )


@contextmanager
def upstream_errors(backend: str) -> Iterator[None]:
    """Translate transport failures into gateway status codes.

    Connection refused means the engine is down or still loading weights (503,
    retryable). A timeout means it is up but overloaded or stuck (504).
    """
    try:
        yield
    except BackendError:
        raise
    except httpx.TimeoutException as exc:
        raise BackendError(f"{backend} timed out: {exc!r}", 504, "upstream_timeout") from exc
    except httpx.ConnectError as exc:
        raise BackendError(f"{backend} unreachable: {exc!r}", 503, "upstream_unavailable") from exc
    except httpx.HTTPError as exc:
        raise BackendError(f"{backend} transport error: {exc!r}", 502) from exc
    except json.JSONDecodeError as exc:
        raise BackendError(f"{backend} sent malformed JSON: {exc}", 502) from exc


async def raise_for_upstream_status(backend: str, response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    body = (await response.aread()).decode("utf-8", errors="replace")[:500]
    if response.status_code < 500:
        # The engine rejected the request itself (e.g. prompt longer than the
        # context window). That is the caller's error, so keep the 4xx.
        raise BackendError(
            f"{backend} rejected request: {body}", response.status_code, "invalid_request"
        )
    raise BackendError(f"{backend} returned {response.status_code}: {body}", 502)


async def iter_sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """Yield the payload of each ``data:`` line of a server-sent event stream."""
    async for line in response.aiter_lines():
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload:
                yield payload
