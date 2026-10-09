"""Prometheus metrics exposed at /metrics.

Bucket boundaries follow LLM serving, not web APIs: time-to-first-token is
usually tens to hundreds of milliseconds, while a full generation can take tens
of seconds.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        self.requests = Counter(
            "inferscale_requests_total",
            "Chat completion requests by backend, mode and outcome.",
            ["backend", "stream", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "inferscale_request_duration_seconds",
            "End-to-end request latency.",
            ["backend", "stream"],
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128),
            registry=self.registry,
        )
        self.ttft = Histogram(
            "inferscale_time_to_first_token_seconds",
            "Time from request arrival to the first streamed token.",
            ["backend"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4),
            registry=self.registry,
        )
        self.completion_tokens = Counter(
            "inferscale_completion_tokens_total",
            "Generated tokens (exact where the engine reports them, else estimated).",
            ["backend"],
            registry=self.registry,
        )
        self.inflight = Gauge(
            "inferscale_inflight_requests",
            "Requests currently being served. The HPA scales the gateway on this.",
            registry=self.registry,
        )
        self.rejected = Counter(
            "inferscale_rejected_total",
            "Requests rejected by admission control (429).",
            registry=self.registry,
        )
