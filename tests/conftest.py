from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from inferscale.app import create_app
from inferscale.backends import MockBackend
from inferscale.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(backends=("mock",), default_backend="mock", max_inflight=8)


@pytest.fixture
def app(settings: Settings):
    return create_app(settings, backends={"mock": MockBackend(ttft_ms=1, inter_token_ms=0)})


@pytest.fixture
async def client(app) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as c:
        yield c


def chat(content: str = "hello", **overrides) -> dict:
    return {"messages": [{"role": "user", "content": content}], "max_tokens": 8, **overrides}
