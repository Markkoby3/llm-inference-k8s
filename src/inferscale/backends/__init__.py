"""Backend registry: maps configuration to adapters, single or pooled."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from inferscale.backends.base import (
    Backend,
    BackendError,
    Completion,
    Delta,
    GenerationParams,
)
from inferscale.backends.mock import MockBackend
from inferscale.backends.openai_compat import NIMBackend, VLLMBackend
from inferscale.backends.pool import Replica, ReplicaPool, dns_discovery, split_urls
from inferscale.backends.triton import TritonBackend
from inferscale.config import Settings

AVAILABLE = ("mock", "vllm", "triton", "nim")


def _pool_options(settings: Settings, metrics: Any) -> dict[str, Any]:
    return {
        "policy": settings.routing_policy,
        "prefix_chars": settings.routing_prefix_chars,
        "load_factor": settings.routing_load_factor,
        "metrics": metrics,
        "refresh_s": settings.discovery_refresh_s,
    }


def _networked(
    name: str, urls: str, make: Callable[[str], Backend], settings: Settings, metrics: Any
) -> Backend:
    """One URL -> a plain adapter; several URLs or dns:// -> a routed replica pool."""
    if urls.startswith("dns://"):
        return ReplicaPool(
            name, [], discover=dns_discovery(urls), factory=make, **_pool_options(settings, metrics)
        )
    targets = split_urls(urls)
    if len(targets) == 1:
        return make(targets[0])
    return ReplicaPool(
        name,
        [Replica(u, make(u)) for u in targets],
        factory=make,
        **_pool_options(settings, metrics),
    )


def build_backend(name: str, settings: Settings, metrics: Any = None) -> Backend:
    if name == "mock":
        if settings.mock_replicas <= 1:
            return MockBackend.from_settings(settings)
        replicas = [
            Replica(f"mock-{i}", MockBackend.from_settings(settings))
            for i in range(settings.mock_replicas)
        ]
        return ReplicaPool("mock", replicas, **_pool_options(settings, metrics))
    if name == "vllm":
        return _networked(
            name, settings.vllm_url, lambda u: VLLMBackend.for_url(settings, u), settings, metrics
        )
    if name == "triton":
        return _networked(
            name,
            settings.triton_url,
            lambda u: TritonBackend.for_url(settings, u),
            settings,
            metrics,
        )
    if name == "nim":
        return _networked(
            name, settings.nim_url, lambda u: NIMBackend.for_url(settings, u), settings, metrics
        )
    raise ValueError(f"unknown backend {name!r}; available: {list(AVAILABLE)}")


def build_backends(settings: Settings, metrics: Any = None) -> dict[str, Backend]:
    return {name: build_backend(name, settings, metrics) for name in settings.backends}


__all__ = [
    "AVAILABLE",
    "Backend",
    "BackendError",
    "Completion",
    "Delta",
    "GenerationParams",
    "MockBackend",
    "NIMBackend",
    "ReplicaPool",
    "TritonBackend",
    "VLLMBackend",
    "build_backend",
    "build_backends",
]
