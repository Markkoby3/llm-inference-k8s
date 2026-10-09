"""Replica routing: policies, bounded-load affinity, failover, discovery, NIM adapter."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import AsyncIterator, Sequence

import httpx
import pytest
import respx

from inferscale.app import create_app
from inferscale.backends import (
    Backend,
    BackendError,
    Completion,
    Delta,
    GenerationParams,
    MockBackend,
    NIMBackend,
    ReplicaPool,
    build_backend,
)
from inferscale.backends.pool import Replica, dns_discovery, rendezvous_score
from inferscale.config import Settings
from inferscale.prompt import prefix_key
from inferscale.schemas import ChatMessage

PARAMS = GenerationParams(max_tokens=4, temperature=0.0, top_p=1.0)


def convo(system: str, user: str = "hi") -> list[ChatMessage]:
    return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]


class Recorder(Backend):
    """Counts calls; optionally refuses connections (503) or stalls."""

    name = "rec"

    def __init__(self, fail: int | None = None, hold: asyncio.Event | None = None):
        self.calls = 0
        self.fail = fail
        self.hold = hold

    async def complete(
        self, messages: Sequence[ChatMessage], params: GenerationParams
    ) -> Completion:
        self.calls += 1
        if self.fail:
            raise BackendError("refused", self.fail, "upstream_unavailable")
        if self.hold:
            await self.hold.wait()
        return Completion("ok", "stop", 1, 1)

    async def stream(
        self, messages: Sequence[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[Delta]:
        self.calls += 1
        if self.fail:
            raise BackendError("refused", self.fail, "upstream_unavailable")
        yield Delta("ok")
        yield Delta("", finish_reason="stop")

    async def ready(self) -> bool:
        return not self.fail


def pool(n: int = 4, policy: str = "prefix_affinity", **kwargs) -> ReplicaPool:
    replicas = [Replica(f"r{i}", Recorder()) for i in range(n)]
    return ReplicaPool("rec", replicas, policy=policy, **kwargs)


# ---------------------------------------------------------------- prefix keys


def test_prefix_key_ignores_newest_message_and_truncates():
    a = prefix_key(convo("system prompt", "question one"))
    b = prefix_key(convo("system prompt", "a different question"))
    assert a == b
    assert prefix_key([ChatMessage(role="user", content="x" * 5000)], 100) == (
        "user\x1f" + "x" * 95
    )


# ------------------------------------------------------------------- policies


def test_round_robin_spreads_evenly():
    p = pool(4, "round_robin")
    picks = Counter(p.choose(convo("s")).id for _ in range(400))
    assert set(picks.values()) == {100}


def test_least_inflight_prefers_idle_replica():
    p = pool(3, "least_inflight")
    p.replicas[0].inflight, p.replicas[1].inflight, p.replicas[2].inflight = 5, 0, 2
    assert p.choose(convo("s")).id == "r1"


def test_affinity_is_sticky_per_prefix_and_spreads_prefixes():
    p = pool(4)
    owners = {p.choose(convo(f"system {k}", f"q{i}")).id for k in [7] for i in range(20)}
    assert len(owners) == 1  # same prefix, different questions -> same replica
    spread = Counter(p.choose(convo(f"system {k}")).id for k in range(400))
    assert len(spread) == 4 and min(spread.values()) > 60  # prefixes spread across replicas


def test_affinity_spills_when_owner_exceeds_load_bound():
    p = pool(4, load_factor=1.25)
    messages = convo("hot prefix")
    owner = p.choose(messages)
    owner.inflight = 4  # pool total 4, bound = ceil(1.25 * 5 / 4) = 2
    spilled = p.choose(messages)
    assert spilled.id != owner.id
    # Spill goes to the prefix's second-ranked replica, so it is stable too.
    key = prefix_key(messages, p.prefix_chars)
    ranked = sorted(p.replicas, key=lambda r: rendezvous_score(key, r.id), reverse=True)
    assert spilled.id == ranked[1].id


def test_rendezvous_moves_only_the_removed_replicas_keys():
    keys = [f"prefix-{i}" for i in range(2000)]
    ids = [f"r{i}" for i in range(5)]

    def owner(key: str, members: list[str]) -> str:
        return max(members, key=lambda r: rendezvous_score(key, r))

    before = {k: owner(k, ids) for k in keys}
    after = {k: owner(k, ids[:-1]) for k in keys}  # scale down: r4 removed
    moved = [k for k in keys if before[k] != after[k]]
    assert all(before[k] == "r4" for k in moved)
    assert 0.15 < len(moved) / len(keys) < 0.25  # ~1/5 of keys, nothing else churns


def test_invalid_policy_rejected():
    with pytest.raises(ValueError):
        pool(2, "random")
    with pytest.raises(ValueError):
        pool(2, load_factor=0.5)


# ------------------------------------------------------------ failover, health


async def test_refused_replica_fails_over_and_leaves_rotation():
    good, bad = Recorder(), Recorder(fail=503)
    p = ReplicaPool("rec", [Replica("bad", bad), Replica("good", good)], policy="round_robin")
    for _ in range(4):
        assert (await p.complete(convo("s"), PARAMS)).text == "ok"
    assert bad.calls == 1  # tried once, then cooled down
    assert good.calls == 4
    assert all(r.inflight == 0 for r in p.replicas)


async def test_stream_failover_before_first_token():
    good, bad = Recorder(), Recorder(fail=503)
    p = ReplicaPool("rec", [Replica("bad", bad), Replica("good", good)], policy="round_robin")
    deltas = [d async for d in p.stream(convo("s"), PARAMS)]
    assert [d.text for d in deltas] == ["ok", ""]
    assert all(r.inflight == 0 for r in p.replicas)


async def test_engine_errors_are_not_retried():
    flaky, good = Recorder(fail=502), Recorder()
    p = ReplicaPool("rec", [Replica("a", flaky), Replica("b", good)], policy="round_robin")
    with pytest.raises(BackendError) as err:
        await p.complete(convo("s"), PARAMS)
    assert err.value.status_code == 502 and good.calls == 0


async def test_all_replicas_down_raises_503():
    p = ReplicaPool("rec", [Replica("a", Recorder(fail=503)), Replica("b", Recorder(fail=503))])
    with pytest.raises(BackendError) as err:
        await p.complete(convo("s"), PARAMS)
    assert err.value.status_code == 503


async def test_inflight_tracked_while_requests_run():
    hold = asyncio.Event()
    p = ReplicaPool("rec", [Replica("a", Recorder(hold=hold))])
    task = asyncio.create_task(p.complete(convo("s"), PARAMS))
    await asyncio.sleep(0)
    assert p.replicas[0].inflight == 1
    hold.set()
    await task
    assert p.replicas[0].inflight == 0


async def test_ready_if_any_replica_ready_and_status_reports_each():
    p = ReplicaPool("rec", [Replica("a", Recorder(fail=503)), Replica("b", Recorder())])
    assert await p.ready()
    status = p.status()
    assert [r["ready"] for r in status["replicas"]] == [False, True]


# ------------------------------------------------------------------ discovery


async def test_discovery_adds_and_drains_replicas():
    membership = [["http://10.0.0.1:8000", "http://10.0.0.2:8000"], ["http://10.0.0.2:8000"]]
    made: list[str] = []

    async def discover() -> list[str]:
        return membership.pop(0)

    def factory(url: str) -> Backend:
        made.append(url)
        return Recorder()

    p = ReplicaPool("rec", [], discover=discover, factory=factory, refresh_s=3600)
    await p.start()
    assert [r.id for r in p.replicas] == ["http://10.0.0.1:8000", "http://10.0.0.2:8000"]
    survivor = p.replicas[1]
    await p.refresh()
    assert p.replicas == [survivor] and len(made) == 2  # existing replica object kept
    await p.aclose()


async def test_dns_discovery_formats_addresses():
    resolve = dns_discovery("dns://localhost:9000")
    urls = await resolve()
    assert urls and all(u.endswith(":9000") and u.startswith("http://") for u in urls)
    with pytest.raises(ValueError):
        dns_discovery("dns://")


# --------------------------------------------------------- registry and gateway


def test_registry_builds_pools_from_url_lists():
    settings = Settings(
        backends=("vllm",),
        default_backend="vllm",
        vllm_url="http://a:8000, http://b:8000",
        routing_policy="least_inflight",
    )
    backend = build_backend("vllm", settings)
    assert isinstance(backend, ReplicaPool)
    assert [r.id for r in backend.replicas] == ["http://a:8000", "http://b:8000"]
    assert backend.policy == "least_inflight"
    single = build_backend("vllm", Settings(backends=("vllm",), default_backend="vllm"))
    assert not isinstance(single, ReplicaPool)


async def test_gateway_reports_replicas_and_routing_metrics():
    settings = Settings(backends=("mock",), default_backend="mock", mock_replicas=3)
    app = create_app(settings)
    pool_backend = build_backend("mock", settings, app.state.metrics)
    for r in pool_backend.replicas:
        r.backend = MockBackend(ttft_ms=0, inter_token_ms=0)
    app.state.backends = {"mock": pool_backend}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as client:
        for i in range(6):
            body = {
                "messages": [
                    {"role": "system", "content": "shared"},
                    {"role": "user", "content": f"q{i}"},
                ],
                "max_tokens": 2,
            }
            assert (await client.post("/v1/chat/completions", json=body)).status_code == 200
        ready = (await client.get("/readyz")).json()
        assert len(ready["replicas"]["mock"]["replicas"]) == 3
        metrics = (await client.get("/metrics")).text
    assert 'inferscale_routing_decisions_total{backend="mock",decision="affinity",' in metrics
    served = [r["requests"] for r in ready["replicas"]["mock"]["replicas"]]
    assert sorted(served) == [0, 0, 6]  # one shared prefix -> one warm replica


# ------------------------------------------------------------------ NIM adapter


@respx.mock
async def test_nim_sends_api_key_and_uses_nim_health_route():
    settings = Settings(
        backends=("nim",),
        default_backend="nim",
        nim_api_key="nvapi-test",
        nim_model="meta/llama-3.1-8b-instruct",
    )
    nim = NIMBackend.for_url(settings, "http://nim:8000")
    route = respx.post("http://nim:8000/v1/chat/completions").respond(
        json={
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        }
    )
    result = await nim.complete(convo("s"), PARAMS)
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer nvapi-test"
    assert json.loads(request.content)["model"] == "meta/llama-3.1-8b-instruct"
    assert result.text == "hi"

    respx.get("http://nim:8000/v1/health/ready").respond(200)
    assert await nim.ready()


@respx.mock
async def test_nim_hosted_health_path_override():
    settings = Settings(backends=("nim",), default_backend="nim", nim_health_path="/v1/models")
    nim = NIMBackend.for_url(settings, "https://integrate.api.nvidia.com")
    respx.get("https://integrate.api.nvidia.com/v1/models").respond(200, json={"data": []})
    assert await nim.ready()


def test_api_key_is_not_in_settings_repr():
    assert "nvapi-secret" not in repr(Settings(nim_api_key="nvapi-secret"))


async def test_simulation_affinity_beats_round_robin_on_cache_hits():
    from inferscale.bench.routing import Workload, run_policy

    w = Workload(requests=400, concurrency=16, base_ttft_ms=1, prefill_ms_per_kchar=1)
    rr = await run_policy("round_robin", w)
    affinity = await run_policy("prefix_affinity", w)
    assert affinity["cache_hit_rate"] > rr["cache_hit_rate"] + 0.1
    assert affinity["load_imbalance"] < 2.0
