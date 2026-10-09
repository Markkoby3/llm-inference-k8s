"""Adapter for NVIDIA Triton Inference Server running the vLLM backend.

Uses Triton's generate extension (``/v2/models/<model>/generate`` and
``generate_stream``), which takes a pre-rendered prompt and scalar parameters.
That shapes two choices here:

* the chat template is rendered gateway-side (see ``inferscale.prompt``), and
* stop sequences are applied gateway-side (see ``inferscale.stops``), because a
  list of strings cannot be sent as a scalar parameter.
"""

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
from inferscale.prompt import TEMPLATE_STOPS, render
from inferscale.schemas import ChatMessage
from inferscale.stops import StopSequenceFilter


class _IncrementalText:
    """Normalize Triton stream chunks to deltas.

    Depending on the vLLM-backend version, each streamed ``text_output`` is either
    the new text only or everything generated so far. Detect cumulative chunks by
    prefix and emit only what is new, so the gateway works with either.
    """

    def __init__(self) -> None:
        self._seen = ""

    def delta(self, text_output: str) -> str:
        if self._seen and text_output.startswith(self._seen):
            new = text_output[len(self._seen) :]
            self._seen = text_output
            return new
        self._seen += text_output
        return text_output


class TritonBackend(Backend):
    name = "triton"

    def __init__(self, client: httpx.AsyncClient, model: str, chat_template: str = "chatml"):
        self._client = client
        self._model = model
        self._template = chat_template
        render([ChatMessage(role="user", content="")], chat_template)  # fail fast on bad config

    @classmethod
    def for_url(cls, settings: Any, url: str) -> TritonBackend:
        client = build_client(url, settings.connect_timeout_s, settings.request_timeout_s)
        return cls(client, settings.triton_model, settings.triton_chat_template)

    def _stops(self, params: GenerationParams) -> tuple[str, ...]:
        return tuple(params.stop) + TEMPLATE_STOPS.get(self._template, ())

    def _payload(
        self, messages: Sequence[ChatMessage], params: GenerationParams, stream: bool
    ) -> dict[str, Any]:
        parameters: dict[str, Any] = {
            "stream": stream,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
        }
        if params.seed is not None:
            parameters["seed"] = params.seed
        if params.ignore_eos:
            parameters["ignore_eos"] = True
        return {
            "text_input": render(messages, self._template),
            "exclude_input_in_output": True,
            "parameters": parameters,
        }

    async def complete(
        self, messages: Sequence[ChatMessage], params: GenerationParams
    ) -> Completion:
        with upstream_errors(self.name):
            response = await self._client.post(
                f"/v2/models/{self._model}/generate", json=self._payload(messages, params, False)
            )
            await raise_for_upstream_status(self.name, response)
            body = response.json()
        if "text_output" not in body:
            raise BackendError(f"triton returned an unexpected body: {body!r:.300}", 502)

        text, matched = StopSequenceFilter(self._stops(params)).apply(body["text_output"])
        return Completion(
            text=text,
            finish_reason="stop" if matched else str(body.get("finish_reason", "stop")),
            # The generate endpoint does not report token counts by default; the
            # gateway marks usage as estimated when these are None.
            prompt_tokens=None,
            completion_tokens=None,
        )

    async def stream(
        self, messages: Sequence[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[Delta]:
        stops = StopSequenceFilter(self._stops(params))
        incremental = _IncrementalText()
        chunks = 0
        with upstream_errors(self.name):
            async with self._client.stream(
                "POST",
                f"/v2/models/{self._model}/generate_stream",
                json=self._payload(messages, params, True),
            ) as response:
                await raise_for_upstream_status(self.name, response)
                async for data in iter_sse_data(response):
                    event = json.loads(data)
                    if "error" in event:
                        raise BackendError(f"triton stream error: {event['error']}", 502)
                    new_text = incremental.delta(event.get("text_output", ""))
                    if not new_text:
                        continue
                    # The vLLM backend emits one token per streamed response.
                    chunks += 1
                    safe = stops.feed(new_text)
                    if safe:
                        yield Delta(safe)
                    if stops.stopped:
                        break

        tail = stops.flush()
        if tail:
            yield Delta(tail)
        reached_limit = not stops.stopped and chunks >= params.max_tokens
        yield Delta(
            "",
            finish_reason="length" if reached_limit else "stop",
            completion_tokens=chunks,
        )

    async def ready(self) -> bool:
        try:
            server = await self._client.get("/v2/health/ready", timeout=2.0)
            model = await self._client.get(f"/v2/models/{self._model}/ready", timeout=2.0)
        except httpx.HTTPError:
            return False
        return server.status_code == 200 and model.status_code == 200

    async def aclose(self) -> None:
        await self._client.aclose()
