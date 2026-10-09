"""Gateway configuration, read from environment variables.

Every setting has a safe default so the gateway boots with the mock backend and
no external dependencies, which is what CI and the CPU-only kind cluster use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

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

    # Engine endpoints. Each accepts one URL, a comma-separated list of replica
    # URLs, or dns://host:port to discover replicas from a headless Service.
    # vLLM OpenAI-compatible server.
    vllm_url: str = "http://vllm:8000"
    vllm_model: str = "qwen2.5-1.5b-instruct"

    # Triton Inference Server running the vLLM backend.
    triton_url: str = "http://triton:8000"
    triton_model: str = "llm"
    # Chat template applied gateway-side for Triton, which accepts raw text only.
    # "chatml" matches Qwen2.5; "llama3" and "plain" are also supported.
    triton_chat_template: str = "chatml"

    # NVIDIA NIM: a self-hosted NIM container, or the hosted API catalog
    # (https://integrate.api.nvidia.com with an API key and health path /v1/models).
    nim_url: str = "http://nim:8000"
    nim_model: str = "meta/llama-3.1-8b-instruct"
    nim_api_key: str = field(default="", repr=False)
    nim_health_path: str = ""

    # Replica routing when an engine has several replicas:
    # prefix_affinity | least_inflight | round_robin.
    routing_policy: str = "prefix_affinity"
    routing_prefix_chars: int = 1024
    # A replica may exceed the pool's average in-flight load by at most this factor
    # before affinity spills a request to the next replica.
    routing_load_factor: float = 1.25
    discovery_refresh_s: float = 10.0

    # Mock backend timing, used to exercise the gateway and harness without a GPU.
    mock_ttft_ms: float = 40.0
    mock_inter_token_ms: float = 8.0
    # Simulated replicas and prefix cache, for exercising replica routing.
    mock_replicas: int = 1
    mock_prefix_cache_size: int = 0
    mock_prefill_ms_per_kchar: float = 0.0

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
            nim_url=_env("NIM_URL", cls.nim_url),
            nim_model=_env("NIM_MODEL", cls.nim_model),
            nim_api_key=_env("NIM_API_KEY", os.environ.get("NVIDIA_API_KEY", "")),
            nim_health_path=_env("NIM_HEALTH_PATH", cls.nim_health_path),
            routing_policy=_env("ROUTING_POLICY", cls.routing_policy),
            routing_prefix_chars=_env_int("ROUTING_PREFIX_CHARS", cls.routing_prefix_chars),
            routing_load_factor=_env_float("ROUTING_LOAD_FACTOR", cls.routing_load_factor),
            discovery_refresh_s=_env_float("DISCOVERY_REFRESH_S", cls.discovery_refresh_s),
            mock_replicas=_env_int("MOCK_REPLICAS", cls.mock_replicas),
            mock_prefix_cache_size=_env_int("MOCK_PREFIX_CACHE_SIZE", cls.mock_prefix_cache_size),
            mock_prefill_ms_per_kchar=_env_float(
                "MOCK_PREFILL_MS_PER_KCHAR", cls.mock_prefill_ms_per_kchar
            ),
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
        if self.routing_policy not in ("prefix_affinity", "least_inflight", "round_robin"):
            raise ValueError(f"unknown INFERSCALE_ROUTING_POLICY {self.routing_policy!r}")
        if self.routing_load_factor < 1.0:
            raise ValueError("INFERSCALE_ROUTING_LOAD_FACTOR must be >= 1.0")
        if self.max_inflight < 0:
            raise ValueError("INFERSCALE_MAX_INFLIGHT must be >= 0")
