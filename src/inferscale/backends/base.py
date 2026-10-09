"""The backend contract every serving engine adapter implements.

The gateway only talks to this interface, so adding a backend (TensorRT-LLM,
SGLang, a remote API) means writing one adapter, not touching request handling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from inferscale.schemas import ChatMessage


@dataclass(frozen=True)
class GenerationParams:
    max_tokens: int
    temperature: float
    top_p: float
    stop: tuple[str, ...] = ()
    seed: int | None = None
    ignore_eos: bool = False


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class Delta:
    """One streamed increment.

    ``text`` is the new text only. The final delta carries ``finish_reason`` and,
    when the engine reports them, exact token counts.
    """

    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class BackendError(Exception):
    """An upstream failure, already mapped to the HTTP status the gateway returns."""

    def __init__(self, message: str, status_code: int = 502, code: str = "upstream_error"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


class Backend(ABC):
    name: str

    @abstractmethod
    async def complete(
        self, messages: Sequence[ChatMessage], params: GenerationParams
    ) -> Completion: ...

    @abstractmethod
    def stream(
        self, messages: Sequence[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[Delta]: ...

    @abstractmethod
    async def ready(self) -> bool:
        """True when the backend can serve traffic right now."""

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release connections. Override when the adapter holds resources."""
