"""Request and response models for the OpenAI-compatible chat completions API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=256, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: list[str] | str | None = None
    seed: int | None = None
    # vLLM extension: keep generating past EOS so every request produces exactly
    # max_tokens. Benchmarks set this so backends are compared on equal output lengths.
    ignore_eos: bool = False

    @field_validator("stop")
    @classmethod
    def _normalize_stop(cls, value: list[str] | str | None) -> list[str] | None:
        if value is None:
            return None
        stops = [value] if isinstance(value, str) else value
        stops = [s for s in stops if s]
        if len(stops) > 4:
            raise ValueError("at most 4 stop sequences are supported")
        return stops or None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str | None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


class ErrorBody(BaseModel):
    message: str
    type: str
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class AgentRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    max_steps: int | None = Field(default=None, ge=1, le=5)
    top_k: int | None = Field(default=None, ge=1, le=10)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    @field_validator("messages")
    @classmethod
    def _last_is_user(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        if value[-1].role != "user":
            raise ValueError("the last message must come from the user")
        return value
