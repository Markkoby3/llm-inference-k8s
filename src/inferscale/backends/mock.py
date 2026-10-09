"""A deterministic in-process backend with configurable latency.

It lets the gateway, the Helm chart and the benchmark harness be exercised end to
end on a laptop or in CI with no GPU. Timing follows the shape of a real engine:
a fixed time-to-first-token, then a steady inter-token gap.
"""

from __future__ import annotations

import asyncio
import json
import random
import zlib
from collections import OrderedDict
from collections.abc import AsyncIterator, Sequence
from typing import Any

from inferscale.backends.base import Backend, Completion, Delta, GenerationParams
from inferscale.prompt import prefix_key
from inferscale.schemas import ChatMessage
from inferscale.stops import StopSequenceFilter

_VOCAB = (
    "the model serves tokens from a batch while the scheduler keeps the gpu busy and "
    "the cache grows with every request so latency depends on load throughput and memory"
).split()

# Without ignore_eos the mock "stops naturally" after this many tokens.
_NATURAL_LENGTH = 64


class MockBackend(Backend):
    name = "mock"

    def __init__(
        self,
        ttft_ms: float = 40.0,
        inter_token_ms: float = 8.0,
        prefix_cache_size: int = 0,
        prefill_ms_per_kchar: float = 0.0,
        prefix_chars: int = 1024,
        max_concurrency: int = 0,
    ):
        """``prefix_cache_size`` > 0 simulates an engine's prefix (KV) cache: an LRU
        of recently seen prompt prefixes. A miss pays prefill time for the whole
        prompt; a hit pays only for the part after the cached prefix. Used to
        compare replica routing policies without GPUs (``inferscale-bench routing``).
        ``max_concurrency`` > 0 caps simultaneous requests like an engine's batch
        slots, so an overloaded replica queues instead of scaling for free."""
        self._slots = asyncio.Semaphore(max_concurrency) if max_concurrency else None
        self._prefix_cache: OrderedDict[str, None] = OrderedDict()
        self._prefix_cache_size = prefix_cache_size
        self._prefill_s_per_char = prefill_ms_per_kchar / 1000 / 1000
        self._prefix_chars = prefix_chars
        self.cache_hits = 0
        self.cache_misses = 0
        self._ttft_s = ttft_ms / 1000
        self._itl_s = inter_token_ms / 1000

    @classmethod
    def from_settings(cls, settings: Any) -> MockBackend:
        return cls(
            settings.mock_ttft_ms,
            settings.mock_inter_token_ms,
            settings.mock_prefix_cache_size,
            settings.mock_prefill_ms_per_kchar,
            settings.routing_prefix_chars,
        )

    def _prefill_s(self, messages: Sequence[ChatMessage]) -> float:
        """Simulated prefill time, given what the prefix cache already holds."""
        if not self._prefix_cache_size and not self._prefill_s_per_char:
            return 0.0
        prompt_chars = sum(len(m.content) for m in messages)
        key = prefix_key(messages, self._prefix_chars)
        if self._prefix_cache_size and key in self._prefix_cache:
            self._prefix_cache.move_to_end(key)
            self.cache_hits += 1
            # A hit reuses KV for the whole shared prefix (everything before the
            # newest message); only the new message is prefilled.
            uncached = len(messages[-1].content) if len(messages) > 1 else 0
        else:
            self.cache_misses += 1
            uncached = prompt_chars
            if self._prefix_cache_size:
                self._prefix_cache[key] = None
                while len(self._prefix_cache) > self._prefix_cache_size:
                    self._prefix_cache.popitem(last=False)
        return uncached * self._prefill_s_per_char

    @staticmethod
    def _prompt_tokens(messages: Sequence[ChatMessage]) -> int:
        return sum(len(m.content.split()) for m in messages) + 4 * len(messages)

    @staticmethod
    def _tokens(messages: Sequence[ChatMessage], params: GenerationParams) -> list[str]:
        # crc32, not hash(): str hashing is randomized per process.
        seed = params.seed if params.seed is not None else zlib.crc32(messages[-1].content.encode())
        rng = random.Random(seed)
        n = params.max_tokens if params.ignore_eos else min(params.max_tokens, _NATURAL_LENGTH)
        return [(" " if i else "") + rng.choice(_VOCAB) for i in range(n)]

    @staticmethod
    def _agent_reply(messages: Sequence[ChatMessage]) -> str | None:
        """Play the agent's planner so the multi-step RAG path runs end to end
        without a GPU: search once with the question's keywords, then answer."""
        system = messages[0].content if messages and messages[0].role == "system" else ""
        if '{"action": "answer"}' not in system:
            return None
        prompt = messages[-1].content
        if "(none yet)" not in prompt:
            return '{"action": "answer"}'
        question = prompt.partition("Question:")[2].partition("\n")[0]
        keywords = [w for w in question.split() if len(w) > 3][:6]
        return json.dumps({"action": "search", "query": " ".join(keywords) or question.strip()})

    async def complete(
        self, messages: Sequence[ChatMessage], params: GenerationParams
    ) -> Completion:
        scripted = self._agent_reply(messages)
        if scripted is not None:
            await asyncio.sleep(self._ttft_s)
            return Completion(
                scripted, "stop", self._prompt_tokens(messages), len(scripted.split())
            )
        tokens = self._tokens(messages, params)
        if self._slots is not None:
            async with self._slots:
                prefill = self._prefill_s(messages)
                await asyncio.sleep(prefill + self._ttft_s + self._itl_s * max(len(tokens) - 1, 0))
        else:
            prefill = self._prefill_s(messages)
            await asyncio.sleep(prefill + self._ttft_s + self._itl_s * max(len(tokens) - 1, 0))
        text, matched = StopSequenceFilter(params.stop).apply("".join(tokens))
        if "[1] (" in messages[-1].content and messages[0].content.startswith(
            "Answer the question"
        ):
            text += " [1]"  # cite like a real model would, so the citation path is exercised
        hit_limit = len(tokens) >= params.max_tokens
        return Completion(
            text=text,
            finish_reason="stop" if matched or not hit_limit else "length",
            prompt_tokens=self._prompt_tokens(messages),
            completion_tokens=len(tokens),
        )

    async def stream(
        self, messages: Sequence[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[Delta]:
        tokens = self._tokens(messages, params)
        stops = StopSequenceFilter(params.stop)
        emitted = 0
        await asyncio.sleep(self._prefill_s(messages) + self._ttft_s)
        for i, token in enumerate(tokens):
            if i:
                await asyncio.sleep(self._itl_s)
            emitted += 1
            safe = stops.feed(token)
            if safe:
                yield Delta(safe)
            if stops.stopped:
                break
        tail = stops.flush()
        if tail:
            yield Delta(tail)
        hit_limit = emitted >= params.max_tokens and not stops.stopped
        yield Delta(
            "",
            finish_reason="length" if hit_limit else "stop",
            prompt_tokens=self._prompt_tokens(messages),
            completion_tokens=emitted,
        )

    async def ready(self) -> bool:
        return True
