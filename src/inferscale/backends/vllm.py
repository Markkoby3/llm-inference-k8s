"""Adapter for vLLM's OpenAI-compatible server (``vllm serve``)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from inferscale.backends._http import (
    build_client,
    iter_sse_data,
    raise_for_upstream_status,
    upstream_errors,
)
from inferscale.backends.base import Backend, BackendError, Completion, Delta, GenerationParams
from inferscale.schemas import ChatMessage


class VLLMBackend(Backend):
    name = "vllm"

    def __init__(self, client: httpx.AsyncClient, model: str):
        self._client = client
        self._model = model

    @classmethod
    def from_settings(cls, settings: Any) -> VLLMBackend:
        client = build_client(
            settings.vllm_url, settings.connect_timeout_s, settings.request_timeout_s
        )
        return cls(client, settings.vllm_model)

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
            raise BackendError(f"vllm returned an unexpected body: {body!r:.300}", 502) from exc

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
                        raise BackendError(f"vllm stream error: {chunk['error']}", 502)
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
        # Older vLLM builds ignore stream_options and never send a usage chunk.
        yield Delta("", finish_reason=finish_reason or "stop")

    async def ready(self) -> bool:
        try:
            response = await self._client.get("/health", timeout=2.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def aclose(self) -> None:
        await self._client.aclose()
