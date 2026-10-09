"""Route requests across several replicas of one engine.

A Kubernetes Service spreads connections at random, which throws away the
engines' prefix (KV) cache: two requests with the same system prompt land on
different pods and both pay full prefill. The pool routes at the request level
instead, with three policies:

* ``round_robin``: even spread, no state. A baseline.
* ``least_inflight``: send to the replica with the fewest active requests.
* ``prefix_affinity`` (default): rendezvous (highest-random-weight) hashing on
  the conversation prefix, so requests that share a prefix go to the same replica
  and hit its cache. Pure affinity would overload the replica that owns a popular
  prefix, so it is **bounded**: a replica may take a request only while its
  in-flight count is below ``load_factor`` x the pool average (consistent
  hashing with bounded loads, Mirrokni et al. 2018). Otherwise the request spills
  to the next replica in that key's rendezvous order, which keeps spill
  placement stable too.

Rendezvous hashing also keeps routing stable when replicas come and go: removing
one replica only moves the keys it owned, so the other replicas keep their warm
caches when the HPA scales the engine.

Replicas that refuse connections are taken out of rotation for a cooldown, and a
request that fails before reaching any engine (503) is retried once on another
replica. Timeouts and engine errors are not retried: the engine may already have
spent GPU time on them.

Membership can be static (a list of URLs) or discovered from DNS, which in
Kubernetes means a headless Service that returns one address per ready pod.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import logging
import math
import socket
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from inferscale.backends.base import Backend, BackendError, Completion, Delta, GenerationParams
from inferscale.prompt import prefix_key
from inferscale.schemas import ChatMessage

log = logging.getLogger("inferscale.pool")

POLICIES = ("round_robin", "least_inflight", "prefix_affinity")


@dataclass
class Replica:
    id: str
    backend: Backend
    inflight: int = 0
    requests: int = 0
    down_until: float = 0.0
    ready: bool | None = None

    def available(self, now: float) -> bool:
        return now >= self.down_until


def rendezvous_score(key: str, replica_id: str) -> int:
    digest = hashlib.blake2b(f"{key}\x00{replica_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _is_failover_safe(exc: BackendError) -> bool:
    # 503 = connection refused / engine not up: the request never reached it.
    return exc.status_code == 503


class ReplicaPool(Backend):
    def __init__(
        self,
        name: str,
        replicas: Sequence[Replica],
        policy: str = "prefix_affinity",
        prefix_chars: int = 1024,
        load_factor: float = 1.25,
        cooldown_s: float = 5.0,
        metrics: Any = None,
        discover: Callable[[], Awaitable[list[str]]] | None = None,
        factory: Callable[[str], Backend] | None = None,
        refresh_s: float = 10.0,
    ):
        if policy not in POLICIES:
            raise ValueError(f"unknown routing policy {policy!r}; expected one of {POLICIES}")
        if load_factor < 1.0:
            raise ValueError("load_factor must be >= 1.0")
        self.name = name
        self.replicas: list[Replica] = list(replicas)
        self.policy = policy
        self.prefix_chars = prefix_chars
        self.load_factor = load_factor
        self.cooldown_s = cooldown_s
        self._metrics = metrics
        self._discover = discover
        self._factory = factory
        self._refresh_s = refresh_s
        self._refresh_task: asyncio.Task[None] | None = None
        self._closing: set[asyncio.Task[None]] = set()
        self._rr = itertools.count()

    # ----------------------------------------------------------------- routing

    def choose(
        self, messages: Sequence[ChatMessage], exclude: frozenset[str] = frozenset()
    ) -> Replica:
        now = time.monotonic()
        pool = [r for r in self.replicas if r.id not in exclude]
        candidates = [r for r in pool if r.available(now)] or pool
        if not candidates:
            raise BackendError(f"{self.name}: no replicas available", 503, "upstream_unavailable")

        if self.policy == "round_robin":
            return candidates[next(self._rr) % len(candidates)]
        if self.policy == "least_inflight":
            start = next(self._rr) % len(candidates)  # rotate ties so they spread
            rotated = candidates[start:] + candidates[:start]
            return min(rotated, key=lambda r: r.inflight)

        key = prefix_key(messages, self.prefix_chars)
        order = sorted(candidates, key=lambda r: rendezvous_score(key, r.id), reverse=True)
        total = sum(r.inflight for r in candidates)
        bound = math.ceil(self.load_factor * (total + 1) / len(candidates))
        for rank, replica in enumerate(order):
            if replica.inflight < bound:
                self._record_decision("affinity" if rank == 0 else "spill")
                return replica
        self._record_decision("spill")  # unreachable: someone is always below the bound
        return order[0]

    def _record_decision(self, decision: str) -> None:
        if self._metrics is not None:
            self._metrics.routing_decisions.labels(self.name, self.policy, decision).inc()

    def _acquire(self, replica: Replica) -> None:
        replica.inflight += 1
        replica.requests += 1
        if self._metrics is not None:
            self._metrics.replica_requests.labels(self.name, replica.id).inc()
            self._metrics.replica_inflight.labels(self.name, replica.id).set(replica.inflight)

    def _release(self, replica: Replica) -> None:
        replica.inflight -= 1
        if self._metrics is not None:
            self._metrics.replica_inflight.labels(self.name, replica.id).set(replica.inflight)

    def _mark_down(self, replica: Replica, exc: BackendError) -> None:
        replica.down_until = time.monotonic() + self.cooldown_s
        log.warning(
            "%s replica %s out of rotation for %.0fs: %s",
            self.name,
            replica.id,
            self.cooldown_s,
            exc.message,
        )
        if self._metrics is not None:
            self._metrics.replica_failovers.labels(self.name).inc()

    # ----------------------------------------------------------------- Backend

    async def complete(
        self, messages: Sequence[ChatMessage], params: GenerationParams
    ) -> Completion:
        tried: set[str] = set()
        while True:
            replica = self.choose(messages, frozenset(tried))
            self._acquire(replica)
            try:
                return await replica.backend.complete(messages, params)
            except BackendError as exc:
                if not _is_failover_safe(exc):
                    raise
                self._mark_down(replica, exc)
                tried.add(replica.id)
                if len(tried) >= 2 or len(tried) >= len(self.replicas):
                    raise
            finally:
                self._release(replica)

    async def stream(
        self, messages: Sequence[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[Delta]:
        tried: set[str] = set()
        while True:
            replica = self.choose(messages, frozenset(tried))
            self._acquire(replica)
            deltas = replica.backend.stream(messages, params)
            try:
                first = await anext(deltas)
            except StopAsyncIteration:
                self._release(replica)
                return
            except BackendError as exc:
                self._release(replica)
                await deltas.aclose()
                if not _is_failover_safe(exc):
                    raise
                self._mark_down(replica, exc)
                tried.add(replica.id)
                if len(tried) >= 2 or len(tried) >= len(self.replicas):
                    raise
                continue
            try:
                yield first
                async for delta in deltas:
                    yield delta
                return
            finally:
                await deltas.aclose()
                self._release(replica)

    async def ready(self) -> bool:
        results = await asyncio.gather(*(r.backend.ready() for r in self.replicas))
        for replica, ok in zip(self.replicas, results, strict=True):
            replica.ready = ok
        return any(results)

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "policy": self.policy,
            "replicas": [
                {
                    "id": r.id,
                    "inflight": r.inflight,
                    "requests": r.requests,
                    "ready": r.ready,
                    "in_rotation": r.available(now),
                }
                for r in self.replicas
            ],
        }

    # -------------------------------------------------------------- membership

    async def start(self) -> None:
        if self._discover is None:
            return
        try:
            await self.refresh()
        except Exception:  # engines may not be resolvable yet; the loop retries
            log.exception("%s: initial replica discovery failed", self.name)
        self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def _refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(self._refresh_s)
            try:
                await self.refresh()
            except Exception:  # keep the last known membership on DNS hiccups
                log.exception("%s: replica discovery failed", self.name)

    async def refresh(self) -> None:
        assert self._discover is not None and self._factory is not None
        urls = await self._discover()
        if not urls:
            log.warning(
                "%s: discovery returned no replicas; keeping %d", self.name, len(self.replicas)
            )
            return
        current = {r.id: r for r in self.replicas}
        added = [u for u in urls if u not in current]
        removed = [r for r in self.replicas if r.id not in set(urls)]
        if not added and not removed:
            return
        self.replicas = [current.get(u) or Replica(u, self._factory(u)) for u in urls]
        log.info("%s replicas: +%s -%s", self.name, added, [r.id for r in removed])
        for replica in removed:
            task = asyncio.create_task(self._close_when_idle(replica))
            self._closing.add(task)
            task.add_done_callback(self._closing.discard)

    @staticmethod
    async def _close_when_idle(
        replica: Replica, poll_s: float = 1.0, max_wait_s: float = 600
    ) -> None:
        deadline = time.monotonic() + max_wait_s
        # Drain: a removed replica finishes the streams it is serving before its
        # connections close. Polling is fine at this frequency (once per scale-down).
        while replica.inflight and time.monotonic() < deadline:  # noqa: ASYNC110
            await asyncio.sleep(poll_s)
        await replica.backend.aclose()

    async def aclose(self) -> None:
        if self._refresh_task is not None:
            self._refresh_task.cancel()
        await asyncio.gather(*(r.backend.aclose() for r in self.replicas))


# ------------------------------------------------------------------ discovery


def dns_discovery(spec: str) -> Callable[[], Awaitable[list[str]]]:
    """``dns://host:port`` -> resolver returning ``http://<ip>:port`` per A/AAAA record.

    Point it at a headless Service: Kubernetes returns one record per ready pod,
    so the pool follows scale-ups, scale-downs and restarts automatically.
    """
    parts = urlsplit(spec)
    host, port = parts.hostname, parts.port or 8000
    if not host:
        raise ValueError(f"invalid discovery URL {spec!r}; expected dns://host:port")

    async def resolve() -> list[str]:
        infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
        addresses = sorted({info[4][0] for info in infos})
        return [f"http://[{a}]:{port}" if ":" in a else f"http://{a}:{port}" for a in addresses]

    return resolve


def split_urls(value: str) -> list[str]:
    return [u.strip() for u in value.split(",") if u.strip()]
