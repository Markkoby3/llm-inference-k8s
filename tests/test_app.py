from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence

import httpx

from inferscale.app import create_app
from inferscale.backends import Backend, BackendError, Completion, Delta, GenerationParams
from inferscale.config import Settings
from inferscale.schemas import ChatMessage
from tests.conftest import chat


def sse_events(text: str) -> list:
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            data = line[len("data: ") :]
            events.append(data if data == "[DONE]" else json.loads(data))
    return events


async def test_non_streaming_returns_openai_shape(client: httpx.AsyncClient):
    response = await client.post("/v1/chat/completions", json=chat(max_tokens=5, ignore_eos=True))
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 5
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + 5
    assert response.headers["X-InferScale-Backend"] == "mock"
    assert response.headers["X-InferScale-Usage"] == "exact"


async def test_streaming_emits_role_content_finish_usage_done(client: httpx.AsyncClient):
    response = await client.post(
        "/v1/chat/completions",
        json=chat(
            max_tokens=4, ignore_eos=True, stream=True, stream_options={"include_usage": True}
        ),
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response.text)

    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    content = [e["choices"][0]["delta"]["content"] for e in events[1:5]]
    assert len(content) == 4 and all(content)
    assert events[5]["choices"][0]["finish_reason"] == "length"
    assert events[6]["choices"] == [] and events[6]["usage"]["completion_tokens"] == 4
    assert events[-1] == "[DONE]"
    # One completion id across the whole stream.
    assert len({e["id"] for e in events if isinstance(e, dict)}) == 1


async def test_streaming_without_usage_option_omits_usage_chunk(client: httpx.AsyncClient):
    response = await client.post("/v1/chat/completions", json=chat(stream=True))
    events = sse_events(response.text)
    assert not any(isinstance(e, dict) and "usage" in e for e in events)
    assert events[-1] == "[DONE]"


async def test_stop_sequence_truncates_output(client: httpx.AsyncClient):
    full = (await client.post("/v1/chat/completions", json=chat(max_tokens=20, seed=7))).json()
    text = full["choices"][0]["message"]["content"]
    stop_word = text.split()[3]
    cut = (
        await client.post(
            "/v1/chat/completions", json=chat(max_tokens=20, seed=7, stop=[stop_word])
        )
    ).json()
    assert cut["choices"][0]["message"]["content"] == text[: text.index(stop_word)]
    assert cut["choices"][0]["finish_reason"] == "stop"


async def test_validation_errors_are_openai_style_400(client: httpx.AsyncClient):
    response = await client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"

    response = await client.post("/v1/chat/completions", json=chat(max_tokens=0))
    assert response.status_code == 400
    assert "max_tokens" in response.json()["error"]["message"]


async def test_unknown_backend_header_is_rejected(client: httpx.AsyncClient):
    response = await client.post(
        "/v1/chat/completions", json=chat(), headers={"X-InferScale-Backend": "tensorrt"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unknown_backend"


async def test_admission_control_sheds_load_with_429(app, client: httpx.AsyncClient):
    app.state.admission.inflight = app.state.settings.max_inflight
    response = await client.post("/v1/chat/completions", json=chat())
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "1"
    app.state.admission.inflight = 0
    assert (await client.post("/v1/chat/completions", json=chat())).status_code == 200


async def test_inflight_counter_returns_to_zero(app, client: httpx.AsyncClient):
    await client.post("/v1/chat/completions", json=chat())
    await client.post("/v1/chat/completions", json=chat(stream=True))
    assert app.state.admission.inflight == 0


async def test_health_readiness_models_and_metrics(client: httpx.AsyncClient):
    assert (await client.get("/healthz")).json() == {"status": "ok"}
    ready = await client.get("/readyz")
    assert ready.status_code == 200 and ready.json()["backends"] == {"mock": True}
    models = (await client.get("/v1/models")).json()
    assert models["data"][0]["id"] == "qwen2.5-1.5b-instruct"

    await client.post("/v1/chat/completions", json=chat(stream=True))
    metrics = (await client.get("/metrics")).text
    assert 'inferscale_requests_total{backend="mock",status="200",stream="true"} 1.0' in metrics
    assert "inferscale_time_to_first_token_seconds_bucket" in metrics


class _FailingBackend(Backend):
    """Fails before the first token, or after some tokens if ``after`` is set."""

    name = "broken"

    def __init__(self, status: int = 503, after: int | None = None):
        self.status = status
        self.after = after

    async def complete(
        self, messages: Sequence[ChatMessage], params: GenerationParams
    ) -> Completion:
        raise BackendError("engine down", self.status, "upstream_unavailable")

    async def stream(
        self, messages: Sequence[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[Delta]:
        for i in range(self.after or 0):
            yield Delta(f"t{i} ")
        raise BackendError("engine down", self.status, "upstream_unavailable")

    async def ready(self) -> bool:
        return False


def _client_for(backend: Backend) -> httpx.AsyncClient:
    settings = Settings(backends=("broken",), default_backend="broken")
    app = create_app(settings, backends={"broken": backend})
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway")


async def test_upstream_failure_maps_to_status_code():
    async with _client_for(_FailingBackend(503)) as client:
        response = await client.post("/v1/chat/completions", json=chat())
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "upstream_unavailable"
        assert (await client.get("/readyz")).status_code == 503


async def test_stream_failure_before_first_token_is_a_real_http_error():
    async with _client_for(_FailingBackend(504)) as client:
        response = await client.post("/v1/chat/completions", json=chat(stream=True))
        assert response.status_code == 504
        assert response.json()["error"]["type"] == "server_error"


async def test_stream_failure_mid_stream_is_reported_in_band():
    async with _client_for(_FailingBackend(502, after=2)) as client:
        response = await client.post("/v1/chat/completions", json=chat(stream=True))
        assert response.status_code == 200
        events = sse_events(response.text)
        assert [e["choices"][0]["delta"].get("content") for e in events[1:3]] == ["t0 ", "t1 "]
        assert "error" in events[-2]
        assert events[-1] == "[DONE]"
