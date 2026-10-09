"""Gateway configuration, read from environment variables.

Every setting has a safe default so the gateway boots with the mock backend and
no external dependencies, which is what CI and the CPU-only kind cluster use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_PREFIX = "INFERSCALE_"


def _env(name: str, default: str) -> str:
    return os.environ.get(_PREFIX + name, default)


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in _env(name, default).split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    # Which backends to construct, and which one serves requests with no routing header.
    backends: tuple[str, ...] = ("mock",)
    default_backend: str = "mock"

    # Model name reported to clients. Both GPU backends serve the same weights.
    model_name: str = "qwen2.5-1.5b-instruct"

    # vLLM OpenAI-compatible server.
    vllm_url: str = "http://vllm:8000"
    vllm_model: str = "qwen2.5-1.5b-instruct"

    # Triton Inference Server running the vLLM backend.
    triton_url: str = "http://triton:8000"
    triton_model: str = "llm"
    # Chat template applied gateway-side for Triton, which accepts raw text only.
    # "chatml" matches Qwen2.5; "llama3" and "plain" are also supported.
    triton_chat_template: str = "chatml"

    # Mock backend timing, used to exercise the gateway and harness without a GPU.
    mock_ttft_ms: float = 40.0
    mock_inter_token_ms: float = 8.0

    # Upstream request timeouts.
    connect_timeout_s: float = 5.0
    request_timeout_s: float = 300.0

    # Admission control: requests beyond this many in flight get 429 + Retry-After
    # instead of queueing unboundedly inside the gateway. 0 disables the limit.
    max_inflight: int = 256

    # Agentic RAG endpoint (/v1/agent/chat). An empty corpus path means the
    # project's own documentation, which ships inside the package.
    rag_enabled: bool = True
    rag_corpus: str = ""
    rag_embedder: str = "hashing"
    rag_index: str = "auto"
    rag_max_steps: int = 3
    rag_top_k: int = 4

    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "info"

    @classmethod
    def from_env(cls) -> Settings:
        backends = _env_list("BACKENDS", "mock")
        settings = cls(
            backends=backends,
            default_backend=_env("DEFAULT_BACKEND", backends[0] if backends else "mock"),
            model_name=_env("MODEL_NAME", cls.model_name),
            vllm_url=_env("VLLM_URL", cls.vllm_url),
            vllm_model=_env("VLLM_MODEL", cls.vllm_model),
            triton_url=_env("TRITON_URL", cls.triton_url),
            triton_model=_env("TRITON_MODEL", cls.triton_model),
            triton_chat_template=_env("TRITON_CHAT_TEMPLATE", cls.triton_chat_template),
            mock_ttft_ms=_env_float("MOCK_TTFT_MS", cls.mock_ttft_ms),
            mock_inter_token_ms=_env_float("MOCK_INTER_TOKEN_MS", cls.mock_inter_token_ms),
            connect_timeout_s=_env_float("CONNECT_TIMEOUT_S", cls.connect_timeout_s),
            request_timeout_s=_env_float("REQUEST_TIMEOUT_S", cls.request_timeout_s),
            max_inflight=_env_int("MAX_INFLIGHT", cls.max_inflight),
            rag_enabled=_env_bool("RAG_ENABLED", cls.rag_enabled),
            rag_corpus=_env("RAG_CORPUS", cls.rag_corpus),
            rag_embedder=_env("RAG_EMBEDDER", cls.rag_embedder),
            rag_index=_env("RAG_INDEX", cls.rag_index),
            rag_max_steps=_env_int("RAG_MAX_STEPS", cls.rag_max_steps),
            rag_top_k=_env_int("RAG_TOP_K", cls.rag_top_k),
            host=_env("HOST", cls.host),
            port=_env_int("PORT", cls.port),
            log_level=_env("LOG_LEVEL", cls.log_level),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.backends:
            raise ValueError("INFERSCALE_BACKENDS must list at least one backend")
        if self.default_backend not in self.backends:
            raise ValueError(
                f"default backend {self.default_backend!r} is not in configured "
                f"backends {list(self.backends)}"
            )
        if self.max_inflight < 0:
            raise ValueError("INFERSCALE_MAX_INFLIGHT must be >= 0")
