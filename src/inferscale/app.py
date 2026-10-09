"""FastAPI application: the OpenAI-compatible surface in front of the backends."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from inferscale import __version__
from inferscale.backends import Backend, BackendError, Delta, GenerationParams, build_backends
from inferscale.config import Settings
from inferscale.metrics import Metrics
from inferscale.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ResponseMessage,
    Usage,
)

log = logging.getLogger("inferscale")

BACKEND_HEADER = "X-InferScale-Backend"


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 characters per token) for engines that report none."""
    return max(1, round(len(text) / 4)) if text else 0


def error_response(
    status: int,
    message: str,
    kind: str,
    code: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = {"error": {"message": message, "type": kind, "code": code}}
    return JSONResponse(body, status_code=status, headers=headers)


def _error_kind(status: int) -> str:
    if status == 429:
        return "rate_limit_error"
    return "invalid_request_error" if status < 500 else "server_error"


class Admission:
    """Bounded in-flight limit. Shedding load with 429 keeps tail latency stable
    and gives the autoscaler a clear signal, instead of letting queues grow inside
    the gateway until requests time out."""

    def __init__(self, limit: int, metrics: Metrics):
        self.limit = limit
        self.inflight = 0
        self._metrics = metrics

    def try_acquire(self) -> bool:
        if self.limit and self.inflight >= self.limit:
            self._metrics.rejected.inc()
            return False
        self.inflight += 1
        self._metrics.inflight.set(self.inflight)
        return True

    def release(self) -> None:
        self.inflight -= 1
        self._metrics.inflight.set(self.inflight)


def _sse(payload: dict[str, Any] | str) -> str:
    data = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    return f"data: {data}\n\n"


def create_app(
    settings: Settings | None = None, backends: dict[str, Backend] | None = None
) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.validate()
    metrics = Metrics()
    admission = Admission(settings.max_inflight, metrics)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = not hasattr(app.state, "backends")
        if owned:
            app.state.backends = build_backends(settings)
        log.info("serving %s via backends=%s", settings.model_name, list(app.state.backends))
        try:
            yield
        finally:
            if owned:
                await asyncio.gather(*(b.aclose() for b in app.state.backends.values()))

    app = FastAPI(title="InferScale", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.metrics = metrics
    app.state.admission = admission
    if backends is not None:
        app.state.backends = backends

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # OpenAI clients expect 400 with an error object, not FastAPI's 422 shape.
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        return error_response(
            400, f"{where}: {first.get('msg', 'invalid request')}", "invalid_request_error"
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        names = list(app.state.backends)
        results = await asyncio.gather(*(app.state.backends[n].ready() for n in names))
        status = dict(zip(names, results, strict=True))
        ready = status[settings.default_backend]
        return JSONResponse({"ready": ready, "backends": status}, status_code=200 if ready else 503)

    @app.get("/metrics")
    async def prometheus() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": settings.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "inferscale",
                }
            ],
        }

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        body: ChatCompletionRequest,
        x_inferscale_backend: str | None = Header(default=None),
    ) -> JSONResponse | StreamingResponse:
        name = x_inferscale_backend or settings.default_backend
        backend = app.state.backends.get(name)
        if backend is None:
            return error_response(
                400,
                f"unknown backend {name!r}; configured: {sorted(app.state.backends)}",
                "invalid_request_error",
                "unknown_backend",
            )
        if not admission.try_acquire():
            return error_response(
                429,
                "gateway is at capacity, retry shortly",
                "rate_limit_error",
                "overloaded",
                headers={"Retry-After": "1"},
            )

        params = GenerationParams(
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            stop=tuple(body.stop or ()),
            seed=body.seed,
            ignore_eos=body.ignore_eos,
        )
        if body.stream:
            return await _stream(name, backend, body, params)
        try:
            return await _complete(name, backend, body, params)
        finally:
            admission.release()

    async def _complete(
        name: str, backend: Backend, body: ChatCompletionRequest, params: GenerationParams
    ) -> JSONResponse:
        start = time.perf_counter()
        try:
            completion = await backend.complete(body.messages, params)
        except BackendError as exc:
            metrics.requests.labels(name, "false", str(exc.status_code)).inc()
            log.warning("backend=%s error=%s", name, exc.message)
            return error_response(
                exc.status_code, exc.message, _error_kind(exc.status_code), exc.code
            )

        exact = completion.completion_tokens is not None and completion.prompt_tokens is not None
        prompt_tokens = completion.prompt_tokens
        if prompt_tokens is None:
            prompt_tokens = estimate_tokens("".join(m.content for m in body.messages))
        completion_tokens = completion.completion_tokens
        if completion_tokens is None:
            completion_tokens = estimate_tokens(completion.text)

        metrics.requests.labels(name, "false", "200").inc()
        metrics.latency.labels(name, "false").observe(time.perf_counter() - start)
        metrics.completion_tokens.labels(name).inc(completion_tokens)

        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            created=int(time.time()),
            model=settings.model_name,
            choices=[
                Choice(
                    message=ResponseMessage(content=completion.text),
                    finish_reason=completion.finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
        return JSONResponse(
            response.model_dump(),
            headers={BACKEND_HEADER: name, "X-InferScale-Usage": "exact" if exact else "estimated"},
        )

    async def _stream(
        name: str, backend: Backend, body: ChatCompletionRequest, params: GenerationParams
    ) -> JSONResponse | StreamingResponse:
        start = time.perf_counter()
        deltas = backend.stream(body.messages, params)

        # Pull the first delta before committing to a 200. If the engine is down or
        # rejects the prompt, the client gets a real HTTP error instead of a stream
        # that opens successfully and then dies.
        try:
            first: Delta | None = await anext(deltas)
        except StopAsyncIteration:
            first = None
        except BackendError as exc:
            await deltas.aclose()
            admission.release()
            metrics.requests.labels(name, "true", str(exc.status_code)).inc()
            log.warning("backend=%s error=%s", name, exc.message)
            return error_response(
                exc.status_code, exc.message, _error_kind(exc.status_code), exc.code
            )

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        include_usage = bool(body.stream_options and body.stream_options.include_usage)

        def chunk(delta: dict[str, str], finish_reason: str | None = None) -> str:
            return _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": settings.model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                }
            )

        async def all_deltas() -> AsyncIterator[Delta]:
            if first is not None:
                yield first
            async for delta in deltas:
                yield delta

        async def events() -> AsyncIterator[str]:
            status = "200"
            text_chunks = 0
            final: Delta | None = None
            try:
                yield chunk({"role": "assistant", "content": ""})
                async for delta in all_deltas():
                    if delta.text:
                        if text_chunks == 0:
                            metrics.ttft.labels(name).observe(time.perf_counter() - start)
                        text_chunks += 1
                        yield chunk({"content": delta.text})
                    if delta.finish_reason is not None:
                        final = delta
                yield chunk({}, finish_reason=final.finish_reason if final else "stop")

                completion_tokens = (
                    final.completion_tokens
                    if final and final.completion_tokens is not None
                    else text_chunks
                )
                metrics.completion_tokens.labels(name).inc(completion_tokens)
                if include_usage:
                    prompt_tokens = (
                        final.prompt_tokens
                        if final and final.prompt_tokens is not None
                        else estimate_tokens("".join(m.content for m in body.messages))
                    )
                    yield _sse(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": settings.model_name,
                            "choices": [],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": prompt_tokens + completion_tokens,
                            },
                        }
                    )
                yield _sse("[DONE]")
            except BackendError as exc:
                # Headers are already sent, so report the failure in-band.
                status = str(exc.status_code)
                log.warning("backend=%s stream error=%s", name, exc.message)
                yield _sse(
                    {"error": {"message": exc.message, "type": "server_error", "code": exc.code}}
                )
                yield _sse("[DONE]")
            except asyncio.CancelledError:
                status = "499"  # client went away mid-stream
                raise
            finally:
                await deltas.aclose()
                admission.release()
                metrics.requests.labels(name, "true", status).inc()
                metrics.latency.labels(name, "true").observe(time.perf_counter() - start)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={BACKEND_HEADER: name, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
