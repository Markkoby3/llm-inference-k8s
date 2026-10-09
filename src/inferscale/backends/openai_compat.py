"""Adapters for engines that speak the OpenAI chat completions API.

vLLM's server (``vllm serve``) and NVIDIA NIM both expose ``/v1/chat/completions``
with SSE streaming, so they share one implementation and differ only in health
checks, authentication and naming.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any, ClassVar

import httpx

from inferscale.backends._http import (
    build_client,
    iter_sse_data,
    raise_for_upstream_status,
    upstream_errors,
)
from inferscale.backends.base import Backend, BackendError, Completion, Delta, GenerationParams
from inferscale.schemas import ChatMessage


class OpenAICompatBackend(Backend):
    name: ClassVar[str] = "openai"
    health_path: ClassVar[str] = "/health"

    def __init__(self, client: httpx.AsyncClient, model: str, health_path: str | None = None):
        self._client = client
        self._model = model
        self._health_path = health_path or self.health_path

    def _payload(
        self, messages: Sequence[ChatMessage], params: GenerationParams, stream: bool
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [m.model_dump() for m in messages],
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "stream": stream,
        }
        if params.stop:
            payload["stop"] = list(params.stop)
        if params.seed is not None:
            payload["seed"] = params.seed
        if params.ignore_eos:
            payload["ignore_eos"] = True
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def complete(
        self, messages: Sequence[ChatMessage], params: GenerationParams
    ) -> Completion:
        with upstream_errors(self.name):
            response = await self._client.post(
                "/v1/chat/completions", json=self._payload(messages, params, stream=False)
            )
            await raise_for_upstream_status(self.name, response)
            body = response.json()
        try:
            choice = body["choices"][0]
            usage = body.get("usage") or {}
            return Completion(
                text=choice["message"]["content"] or "",
                finish_reason=choice.get("finish_reason") or "stop",
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(
                f"{self.name} returned an unexpected body: {body!r:.300}", 502
            ) from exc

    async def stream(
        self, messages: Sequence[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[Delta]:
        payload = self._payload(messages, params, stream=True)
        finish_reason: str | None = None
        with upstream_errors(self.name):
            async with self._client.stream(
                "POST", "/v1/chat/completions", json=payload
            ) as response:
                await raise_for_upstream_status(self.name, response)
                async for data in iter_sse_data(response):
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise BackendError(f"{self.name} stream error: {chunk['error']}", 502)
                    usage = chunk.get("usage")
                    if usage and not chunk.get("choices"):
                        # Final usage-only chunk requested via stream_options.
                        yield Delta(
                            "",
                            finish_reason=finish_reason or "stop",
                            prompt_tokens=usage.get("prompt_tokens"),
                            completion_tokens=usage.get("completion_tokens"),
                        )
                        return
                    for choice in chunk.get("choices", []):
                        text = (choice.get("delta") or {}).get("content") or ""
                        finish_reason = choice.get("finish_reason") or finish_reason
                        if text:
                            yield Delta(text)
        # Some servers ignore stream_options and never send a usage chunk.
        yield Delta("", finish_reason=finish_reason or "stop")

    async def ready(self) -> bool:
        try:
            response = await self._client.get(self._health_path, timeout=2.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def aclose(self) -> None:
        await self._client.aclose()


class VLLMBackend(OpenAICompatBackend):
    """vLLM's OpenAI-compatible server (``vllm serve``)."""

    name = "vllm"
    health_path = "/health"

    @classmethod
    def for_url(cls, settings: Any, url: str) -> VLLMBackend:
        client = build_client(url, settings.connect_timeout_s, settings.request_timeout_s)
        return cls(client, settings.vllm_model)


class NIMBackend(OpenAICompatBackend):
    """NVIDIA NIM: a self-hosted NIM container, or the hosted API catalog.

    Self-hosted NIM serves ``/v1/health/ready``. The hosted endpoint
    (``https://integrate.api.nvidia.com``) needs an API key and has no health
    route, so set ``INFERSCALE_NIM_HEALTH_PATH=/v1/models`` there.
    """

    name = "nim"
    health_path = "/v1/health/ready"

    @classmethod
    def for_url(cls, settings: Any, url: str) -> NIMBackend:
        client = build_client(url, settings.connect_timeout_s, settings.request_timeout_s)
        if settings.nim_api_key:
            client.headers["Authorization"] = f"Bearer {settings.nim_api_key}"
        return cls(client, settings.nim_model, settings.nim_health_path or None)
