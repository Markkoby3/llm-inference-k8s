"""Backend registry: maps the names used in configuration to adapter classes."""

from __future__ import annotations

from inferscale.backends.base import (
    Backend,
    BackendError,
    Completion,
    Delta,
    GenerationParams,
)
from inferscale.backends.mock import MockBackend
from inferscale.backends.triton import TritonBackend
from inferscale.backends.vllm import VLLMBackend
from inferscale.config import Settings

REGISTRY: dict[str, type[MockBackend] | type[VLLMBackend] | type[TritonBackend]] = {
    "mock": MockBackend,
    "vllm": VLLMBackend,
    "triton": TritonBackend,
}


def build_backends(settings: Settings) -> dict[str, Backend]:
    unknown = set(settings.backends) - set(REGISTRY)
    if unknown:
        raise ValueError(f"unknown backends {sorted(unknown)}; available: {sorted(REGISTRY)}")
    return {name: REGISTRY[name].from_settings(settings) for name in settings.backends}


__all__ = [
    "REGISTRY",
    "Backend",
    "BackendError",
    "Completion",
    "Delta",
    "GenerationParams",
    "build_backends",
]
