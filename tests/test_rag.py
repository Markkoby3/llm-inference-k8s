from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import httpx
import numpy as np
import pytest

from inferscale.app import create_app
from inferscale.backends import Backend, BackendError, Completion, Delta, GenerationParams
from inferscale.backends.mock import MockBackend
from inferscale.config import Settings
from inferscale.rag.agent import RagAgent, parse_action
from inferscale.rag.corpus import chunk_text, load_chunks
from inferscale.rag.embed import HashingEmbedder, stem, tokenize
from inferscale.rag.evaluate import evaluate, load_questions
from inferscale.rag.index import FaissIndex, NumpyIndex, faiss_available
from inferscale.rag.store import DocumentStore
from inferscale.schemas import ChatMessage

REPO = Path(__file__).resolve().parents[1]

DOCS = {
    "gpu.md": (
        "# GPUs\n\n## Memory\n\n"
        "The KV cache stores attention keys and values for every token.\n\n"
        "## Scheduling\n\nContinuous batching adds new requests to a running batch."
    ),
    "k8s.md": (
        "# Kubernetes\n\n## Probes\n\nStartup probes give slow containers time to load weights."
    ),
}


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    for name, text in DOCS.items():
        (tmp_path / name).write_text(text)
    return tmp_path


@pytest.fixture
def store(corpus: Path) -> DocumentStore:
    return DocumentStore.from_paths([corpus])


# ----------------------------------------------------------------- chunking, embedding


def test_chunks_follow_headings_and_carry_them():
    chunks = chunk_text("gpu.md", DOCS["gpu.md"])
    assert [c.heading for c in chunks] == ["Memory", "Scheduling"]
    assert chunks[0].text.startswith("Memory\n")
    assert chunks[0].citation == "gpu.md § Memory"


def test_long_paragraphs_split_with_overlap():
    words = [f"w{i}" for i in range(100)]
    chunks = chunk_text("long.md", " ".join(words), max_words=40, overlap_words=10)
    assert len(chunks) == 3
    assert chunks[1].text.split()[0] == "w30"  # 40 - 10 overlap


def test_headings_inside_code_blocks_are_not_sections():
    text = "# Real\n\n```bash\n# not a heading\necho hi\n```\n"
    assert [c.heading for c in chunk_text("x.md", text)] == ["Real"]


def test_stemming_and_stopwords():
    assert tokenize("The containers are scaling") == ["container", "scal"]
    assert tokenize("pods") == ["pods"]  # short words are left alone
    assert stem("p95") == "p95" and stem("bus") == "bus"


def test_hashing_embedder_is_normalized_and_deterministic():
    emb = HashingEmbedder(dim=256).fit(["kv cache memory", "batch scheduling"])
    a, b = emb.embed(["kv cache"]), emb.embed(["kv cache"])
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a), 1.0)
    assert np.allclose(emb.embed([""]), 0)


@pytest.mark.skipif(not faiss_available(), reason="faiss not installed")
def test_faiss_and_numpy_indexes_agree():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(50, 32)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query = vectors[7:8] + 0.01
    results = []
    for index in (NumpyIndex(32), FaissIndex(32)):
        index.add(vectors)
        _, ids = index.search(query, 5)
        results.append(ids[0].tolist())
    assert results[0] == results[1] and results[0][0] == 7


def test_store_ranks_the_relevant_section_first(store: DocumentStore):
    assert store.search("what does the kv cache store", 2)[0].chunk.heading == "Memory"
    assert store.search("startup probe for loading weights", 2)[0].chunk.source == "k8s.md"
    assert store.info()["chunks"] == 3


def test_default_corpus_is_the_project_docs():
    chunks = load_chunks([REPO / "docs", REPO / "README.md"])
    assert {"README.md", "design-decisions.md"} <= {c.source for c in chunks}


def test_retrieval_quality_does_not_regress():
    """Guards the measured retrieval quality on the labeled set (docs/agentic-rag.md)."""
    store = DocumentStore.from_paths([REPO / "docs", REPO / "README.md"])
    report = evaluate(store, load_questions(REPO / "benchmarks/rag/questions.jsonl"))
    assert report["recall@5"] >= 0.85
    assert report["mrr"] >= 0.7


# ------------------------------------------------------------------------- agent


def test_parse_action_finds_json_inside_chatter():
    assert parse_action('Sure! {"action": "search", "query": "kv cache"} done') == {
        "action": "search",
        "query": "kv cache",
    }
    assert parse_action('{"note": 1} {"action": "answer"}') == {"action": "answer"}
    assert parse_action("I will search for it") is None


class ScriptedBackend(Backend):
    """Returns planner replies in order, then a fixed answer."""

    name = "scripted"

    def __init__(self, plans: list[str], answer: str = "It stores keys and values [1]."):
        self.plans = list(plans)
        self.answer = answer
        self.calls: list[list[ChatMessage]] = []

    async def complete(
        self, messages: Sequence[ChatMessage], params: GenerationParams
    ) -> Completion:
        self.calls.append(list(messages))
        if "research agent" in messages[0].content:
            text = self.plans.pop(0) if self.plans else '{"action": "answer"}'
        else:
            text = self.answer
        return Completion(text, "stop", 10, len(text.split()))

    async def stream(
        self, messages: Sequence[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[Delta]:
        raise NotImplementedError
        yield  # pragma: no cover

    async def ready(self) -> bool:
        return True


QUESTION = [ChatMessage(role="user", content="What does the KV cache hold?")]


async def test_agent_searches_then_answers_with_citations(store: DocumentStore):
    backend = ScriptedBackend(['{"action": "search", "query": "kv cache"}', '{"action": "answer"}'])
    result = await RagAgent(backend, store).run(QUESTION)

    assert [s.action for s in result.steps] == ["search", "answer"]
    assert result.passages[0].chunk.heading == "Memory"
    assert result.cited == {1}
    assert result.citations()[0]["cited"] is True
    final_prompt = backend.calls[-1][-1].content
    assert "[1] (gpu.md § Memory)" in final_prompt
    assert set(result.timings_ms) == {"plan", "retrieve", "generate", "total"}
    assert result.completion_tokens is not None


async def test_agent_falls_back_to_one_shot_rag_on_bad_plan(store: DocumentStore):
    backend = ScriptedBackend(["I think I should look this up."])
    result = await RagAgent(backend, store).run(QUESTION)
    assert [s.action for s in result.steps] == ["fallback_search"]
    assert result.steps[0].query == QUESTION[0].content
    assert result.passages


async def test_agent_stops_on_repeated_query_and_respects_max_steps(store: DocumentStore):
    repeat = '{"action": "search", "query": "kv cache"}'
    result = await RagAgent(ScriptedBackend([repeat, repeat]), store).run(QUESTION)
    assert [s.action for s in result.steps] == ["search", "answer"]

    plans = [f'{{"action": "search", "query": "q{i}"}}' for i in range(10)]
    result = await RagAgent(ScriptedBackend(plans), store, max_steps=2).run(QUESTION)
    assert len(result.steps) == 2


async def test_agent_ignores_out_of_range_citations(store: DocumentStore):
    backend = ScriptedBackend(['{"action": "answer"}'], answer="See [9].")
    result = await RagAgent(backend, store).run(QUESTION)
    assert result.passages == [] and result.cited == set()


# ---------------------------------------------------------------------- endpoint


def _agent_client(store: DocumentStore | None, backend: Backend | None = None) -> httpx.AsyncClient:
    backend = backend or MockBackend(ttft_ms=1, inter_token_ms=0)
    settings = Settings(backends=(backend.name,), default_backend=backend.name)
    app = create_app(settings, backends={backend.name: backend}, store=store)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway")


async def test_agent_endpoint_runs_full_loop_with_mock(store: DocumentStore):
    async with _agent_client(store) as client:
        response = await client.post(
            "/v1/agent/chat",
            json={"messages": [{"role": "user", "content": "What does the KV cache store?"}]},
        )
        assert response.status_code == 200
        body = response.json()
        assert [s["action"] for s in body["steps"]] == ["search", "answer"]
        assert body["citations"] and body["retrieval"]["chunks"] == 3
        assert body["timings_ms"]["total"] >= body["timings_ms"]["generate"]
        metrics = (await client.get("/metrics")).text
        assert (
            'inferscale_agent_stage_seconds_count{backend="mock",stage="retrieve"} 1.0' in metrics
        )


async def test_agent_endpoint_validation_and_disabled():
    async with _agent_client(None) as client:
        response = await client.post(
            "/v1/agent/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "rag_disabled"


async def test_agent_endpoint_rejects_non_user_last_message(store: DocumentStore):
    async with _agent_client(store) as client:
        response = await client.post(
            "/v1/agent/chat", json={"messages": [{"role": "assistant", "content": "hi"}]}
        )
        assert response.status_code == 400


class _Down(ScriptedBackend):
    name = "down"

    async def complete(self, messages, params):
        raise BackendError("engine down", 503, "upstream_unavailable")


async def test_agent_endpoint_maps_backend_errors(store: DocumentStore):
    async with _agent_client(store, _Down([])) as client:
        response = await client.post(
            "/v1/agent/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status_code == 503
        assert json.loads(response.text)["error"]["code"] == "upstream_unavailable"
