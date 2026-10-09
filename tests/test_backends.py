"""Adapter tests against recorded-shape upstream responses (no GPU needed)."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from inferscale.backends import BackendError, GenerationParams
from inferscale.backends.triton import TritonBackend, _IncrementalText
from inferscale.backends.vllm import VLLMBackend
from inferscale.schemas import ChatMessage

MESSAGES = [ChatMessage(role="user", content="What is Triton?")]
PARAMS = GenerationParams(max_tokens=16, temperature=0.0, top_p=1.0)


def sse(*events: object) -> bytes:
    lines = [f"data: {e if isinstance(e, str) else json.dumps(e)}\n\n" for e in events]
    return "".join(lines).encode()


async def collect(agen):
    return [d async for d in agen]


# --------------------------------------------------------------------------- vLLM


@pytest.fixture
def vllm():
    client = httpx.AsyncClient(base_url="http://vllm:8000")
    return VLLMBackend(client, model="qwen")


@respx.mock
async def test_vllm_complete_forwards_params_and_parses_usage(vllm: VLLMBackend):
    route = respx.post("http://vllm:8000/v1/chat/completions").respond(
        json={
            "choices": [{"message": {"content": "An inference server."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }
    )
    params = GenerationParams(
        max_tokens=16, temperature=0.0, top_p=1.0, stop=("\n",), ignore_eos=True
    )
    result = await vllm.complete(MESSAGES, params)

    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "qwen" and sent["stream"] is False
    assert sent["stop"] == ["\n"] and sent["ignore_eos"] is True
    assert result.text == "An inference server."
    assert (result.prompt_tokens, result.completion_tokens) == (12, 4)


@respx.mock
async def test_vllm_stream_yields_deltas_then_usage(vllm: VLLMBackend):
    def chunk(text=None, finish=None):
        delta = {"content": text} if text is not None else {}
        return {"choices": [{"delta": delta, "finish_reason": finish}]}

    respx.post("http://vllm:8000/v1/chat/completions").respond(
        content=sse(
            chunk(""),
            chunk("An"),
            chunk(" inference"),
            chunk(finish="length"),
            {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 2}},
            "[DONE]",
        ),
        headers={"content-type": "text/event-stream"},
    )
    deltas = await collect(vllm.stream(MESSAGES, PARAMS))
    assert [d.text for d in deltas] == ["An", " inference", ""]
    assert deltas[-1].finish_reason == "length" and deltas[-1].completion_tokens == 2


@pytest.mark.parametrize(
    ("side_effect", "status"),
    [
        (httpx.ConnectError("refused"), 503),
        (httpx.ReadTimeout("slow"), 504),
    ],
)
@respx.mock
async def test_vllm_transport_errors_map_to_gateway_status(vllm, side_effect, status):
    respx.post("http://vllm:8000/v1/chat/completions").mock(side_effect=side_effect)
    with pytest.raises(BackendError) as err:
        await vllm.complete(MESSAGES, PARAMS)
    assert err.value.status_code == status


@respx.mock
async def test_vllm_upstream_4xx_is_preserved_and_5xx_becomes_502(vllm):
    route = respx.post("http://vllm:8000/v1/chat/completions")
    route.respond(400, json={"message": "prompt too long"})
    with pytest.raises(BackendError) as err:
        await vllm.complete(MESSAGES, PARAMS)
    assert err.value.status_code == 400 and "prompt too long" in err.value.message

    route.respond(500, text="CUDA out of memory")
    with pytest.raises(BackendError) as err:
        await collect(vllm.stream(MESSAGES, PARAMS))
    assert err.value.status_code == 502


@respx.mock
async def test_vllm_readiness(vllm):
    respx.get("http://vllm:8000/health").respond(200)
    assert await vllm.ready()
    respx.get("http://vllm:8000/health").mock(side_effect=httpx.ConnectError("down"))
    assert not await vllm.ready()


# ------------------------------------------------------------------------- Triton


@pytest.fixture
def triton():
    client = httpx.AsyncClient(base_url="http://triton:8000")
    return TritonBackend(client, model="llm", chat_template="chatml")


@respx.mock
async def test_triton_complete_renders_template_and_scalar_params(triton: TritonBackend):
    route = respx.post("http://triton:8000/v2/models/llm/generate").respond(
        json={"model_name": "llm", "text_output": "An inference server."}
    )
    params = GenerationParams(max_tokens=16, temperature=0.0, top_p=1.0, seed=3, ignore_eos=True)
    result = await triton.complete(MESSAGES, params)

    sent = json.loads(route.calls.last.request.content)
    assert sent["text_input"].startswith("<|im_start|>user\nWhat is Triton?<|im_end|>")
    assert sent["exclude_input_in_output"] is True
    assert sent["parameters"] == {
        "stream": False,
        "max_tokens": 16,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 3,
        "ignore_eos": True,
    }
    # Triton's generate parameters must be scalars: no list-valued stop field.
    assert all(not isinstance(v, list | dict) for v in sent["parameters"].values())
    assert result.text == "An inference server."
    assert result.completion_tokens is None  # gateway will estimate


@respx.mock
async def test_triton_applies_stop_sequences_gateway_side(triton: TritonBackend):
    respx.post("http://triton:8000/v2/models/llm/generate").respond(
        json={"text_output": "first line\nsecond line"}
    )
    params = GenerationParams(max_tokens=16, temperature=0.0, top_p=1.0, stop=("\n",))
    result = await triton.complete(MESSAGES, params)
    assert result.text == "first line" and result.finish_reason == "stop"


@pytest.mark.parametrize(
    "outputs",
    [
        ["An", " inference", " server"],  # incremental chunks
        ["An", "An inference", "An inference server"],  # cumulative chunks
    ],
    ids=["incremental", "cumulative"],
)
@respx.mock
async def test_triton_stream_handles_incremental_and_cumulative_chunks(triton, outputs):
    respx.post("http://triton:8000/v2/models/llm/generate_stream").respond(
        content=sse(*({"text_output": o} for o in outputs)),
        headers={"content-type": "text/event-stream"},
    )
    deltas = await collect(triton.stream(MESSAGES, PARAMS))
    assert "".join(d.text for d in deltas) == "An inference server"
    assert deltas[-1].completion_tokens == 3


@respx.mock
async def test_triton_stream_stops_on_template_end_marker_split_across_chunks(triton):
    respx.post("http://triton:8000/v2/models/llm/generate_stream").respond(
        content=sse(
            {"text_output": "Done"},
            {"text_output": "<|im_"},
            {"text_output": "end|>"},
            {"text_output": "garbage"},
        ),
        headers={"content-type": "text/event-stream"},
    )
    deltas = await collect(triton.stream(MESSAGES, PARAMS))
    assert "".join(d.text for d in deltas) == "Done"
    assert deltas[-1].finish_reason == "stop"


@respx.mock
async def test_triton_stream_error_event_raises(triton):
    respx.post("http://triton:8000/v2/models/llm/generate_stream").respond(
        content=sse({"text_output": "x"}, {"error": "engine died"}),
        headers={"content-type": "text/event-stream"},
    )
    with pytest.raises(BackendError, match="engine died"):
        await collect(triton.stream(MESSAGES, PARAMS))


@respx.mock
async def test_triton_readiness_requires_server_and_model(triton):
    respx.get("http://triton:8000/v2/health/ready").respond(200)
    respx.get("http://triton:8000/v2/models/llm/ready").respond(400)
    assert not await triton.ready()
    respx.get("http://triton:8000/v2/models/llm/ready").respond(200)
    assert await triton.ready()


def test_incremental_text_normalizer():
    norm = _IncrementalText()
    assert [norm.delta(t) for t in ["a", "ab", "abc"]] == ["a", "b", "c"]
    norm = _IncrementalText()
    assert [norm.delta(t) for t in ["a", "b", "c"]] == ["a", "b", "c"]
