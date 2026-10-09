"""An agent that decides when and what to search before it answers.

Loop, up to ``max_steps`` times:

1. **Plan.** The model sees the question and the passages found so far, and
   replies with one JSON action: ``{"action": "search", "query": "..."}`` or
   ``{"action": "answer"}``.
2. **Act.** A search runs against the vector index; new passages are added.

Then a final **generate** call answers from the collected passages with inline
``[n]`` citations.

Tool use is expressed as JSON in plain text rather than through the OpenAI
``tools`` field, because Triton's generate endpoint has no tool-calling API. This
keeps the agent identical across backends, which is what lets its end-to-end
latency be compared between vLLM and Triton.

If the model's plan is not valid JSON, the agent falls back to a single search
with the user's question, so it degrades to classic one-shot RAG instead of failing.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from inferscale.backends.base import Backend, GenerationParams
from inferscale.rag.store import DocumentStore, Hit
from inferscale.schemas import ChatMessage

PLANNER_SYSTEM = """You are a research agent answering questions from a documentation set.
Each turn, choose exactly one action:
- search: look up passages in the documentation. Use specific keywords, not the full question.
- answer: stop when the passages found so far are enough to answer.
Reply with one JSON object and nothing else, either
{"action": "search", "query": "<keywords>"}
or
{"action": "answer"}"""

ANSWER_SYSTEM = """Answer the question using only the numbered passages below.
Cite the passages you use inline, like [1] or [2][3].
If the passages do not contain the answer, say that the documentation does not cover it.
Be concise."""

_CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class Step:
    action: str  # "search" | "answer" | "fallback_search"
    query: str | None = None
    new_passages: int = 0
    plan_ms: float = 0.0
    retrieve_ms: float = 0.0
    raw_plan: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "query": self.query,
            "new_passages": self.new_passages,
            "plan_ms": round(self.plan_ms, 2),
            "retrieve_ms": round(self.retrieve_ms, 3),
        }


@dataclass
class AgentResult:
    answer: str
    passages: list[Hit]
    cited: set[int]
    steps: list[Step]
    timings_ms: dict[str, float]
    completion_tokens: int | None
    finish_reason: str

    def citations(self) -> list[dict[str, Any]]:
        return [
            {
                "index": n,
                "source": hit.chunk.source,
                "section": hit.chunk.heading,
                "score": round(hit.score, 4),
                "cited": n in self.cited,
                "snippet": _snippet(hit.chunk.text),
            }
            for n, hit in enumerate(self.passages, 1)
        ]


def _snippet(text: str, words: int = 40) -> str:
    parts = text.split()
    return " ".join(parts[:words]) + (" ..." if len(parts) > words else "")


def parse_action(text: str) -> dict[str, Any] | None:
    """Return the first JSON object in the model's reply that names an action."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("action") in {"search", "answer"}:
            return obj
    return None


def _format_passages(passages: Sequence[Hit], words: int = 120) -> str:
    if not passages:
        return "(none yet)"
    return "\n\n".join(
        f"[{n}] ({hit.chunk.citation})\n{_snippet(hit.chunk.text, words)}"
        for n, hit in enumerate(passages, 1)
    )


class RagAgent:
    def __init__(
        self,
        backend: Backend,
        store: DocumentStore,
        max_steps: int = 3,
        top_k: int = 4,
        max_passages: int = 8,
    ):
        self.backend = backend
        self.store = store
        self.max_steps = max_steps
        self.top_k = top_k
        self.max_passages = max_passages

    async def _retrieve(self, query: str, found: dict[int, Hit]) -> tuple[int, float]:
        start = time.perf_counter()
        # Embedding and search are CPU work: keep them off the event loop.
        hits = await asyncio.to_thread(self.store.search, query, self.top_k)
        elapsed = (time.perf_counter() - start) * 1000
        new = 0
        for hit in hits:
            current = found.get(hit.chunk.id)
            if current is None:
                new += 1
            if current is None or hit.score > current.score:
                found[hit.chunk.id] = hit
        return new, elapsed

    def _ranked(self, found: dict[int, Hit]) -> list[Hit]:
        return sorted(found.values(), key=lambda h: h.score, reverse=True)[: self.max_passages]

    async def run(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> AgentResult:
        question = next((m.content for m in reversed(messages) if m.role == "user"), "")
        history = list(messages[:-1]) if messages and messages[-1].role == "user" else []
        start = time.perf_counter()
        found: dict[int, Hit] = {}
        steps: list[Step] = []
        queries: set[str] = set()
        tokens = 0
        tokens_known = True
        plan_params = GenerationParams(max_tokens=96, temperature=0.0, top_p=1.0)

        for _ in range(self.max_steps):
            prompt = (
                f"Question: {question}\n\n"
                f"Passages found so far:\n{_format_passages(self._ranked(found), words=60)}\n\n"
                f"Searches already made: {sorted(queries) or 'none'}\n"
                "Next action (JSON only):"
            )
            t0 = time.perf_counter()
            plan = await self.backend.complete(
                [
                    ChatMessage(role="system", content=PLANNER_SYSTEM),
                    ChatMessage(role="user", content=prompt),
                ],
                plan_params,
            )
            plan_ms = (time.perf_counter() - t0) * 1000
            if plan.completion_tokens is None:
                tokens_known = False
            else:
                tokens += plan.completion_tokens

            action = parse_action(plan.text)
            if action is None:
                step = Step("fallback_search" if not queries else "answer", plan_ms=plan_ms)
                step.raw_plan = plan.text[:200]
                if not queries:
                    step.query = question
                    queries.add(question)
                    step.new_passages, step.retrieve_ms = await self._retrieve(question, found)
                steps.append(step)
                break

            query = str(action.get("query") or "").strip()
            if action["action"] == "answer" or not query or query.lower() in queries:
                steps.append(Step("answer", plan_ms=plan_ms))
                break
            queries.add(query.lower())
            step = Step("search", query=query, plan_ms=plan_ms)
            step.new_passages, step.retrieve_ms = await self._retrieve(query, found)
            steps.append(step)

        passages = self._ranked(found)
        t0 = time.perf_counter()
        answer = await self.backend.complete(
            [
                ChatMessage(role="system", content=ANSWER_SYSTEM),
                *history,
                ChatMessage(
                    role="user",
                    content=f"Passages:\n{_format_passages(passages)}\n\nQuestion: {question}",
                ),
            ],
            GenerationParams(max_tokens=max_tokens, temperature=temperature, top_p=1.0),
        )
        generate_ms = (time.perf_counter() - t0) * 1000
        if answer.completion_tokens is None:
            tokens_known = False
        else:
            tokens += answer.completion_tokens

        cited = {int(n) for n in _CITATION.findall(answer.text) if 1 <= int(n) <= len(passages)}
        timings = {
            "plan": round(sum(s.plan_ms for s in steps), 2),
            "retrieve": round(sum(s.retrieve_ms for s in steps), 3),
            "generate": round(generate_ms, 2),
            "total": round((time.perf_counter() - start) * 1000, 2),
        }
        return AgentResult(
            answer=answer.text,
            passages=passages,
            cited=cited,
            steps=steps,
            timings_ms=timings,
            completion_tokens=tokens if tokens_known else None,
            finish_reason=answer.finish_reason,
        )
