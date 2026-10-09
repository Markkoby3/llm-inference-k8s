"""Aggregate per-request timings into the numbers a serving benchmark reports."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from inferscale.bench.client import RequestResult

PERCENTILES = (50, 90, 95, 99)


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (same method as numpy's default)."""
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _distribution_ms(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        # None, not NaN: NaN is not valid JSON.
        return {"mean": None, **{f"p{p}": None for p in PERCENTILES}}
    out = {"mean": 1000 * sum(values) / len(values)}
    out.update({f"p{p}": 1000 * percentile(values, p) for p in PERCENTILES})
    return {k: round(v, 2) for k, v in out.items()}


def summarize(
    results: Sequence[RequestResult], wall_time_s: float, concurrency: int
) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    errors = [r for r in results if not r.ok]
    output_tokens = sum(r.output_tokens for r in ok)
    error_samples = sorted({r.error or "unknown" for r in errors})[:5]
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "successes": len(ok),
        "errors": len(errors),
        "error_rate": round(len(errors) / len(results), 4) if results else 0.0,
        "wall_time_s": round(wall_time_s, 3),
        "request_throughput_rps": round(len(ok) / wall_time_s, 3) if wall_time_s else 0.0,
        "output_token_throughput_tps": round(output_tokens / wall_time_s, 1)
        if wall_time_s
        else 0.0,
        "mean_output_tokens": round(output_tokens / len(ok), 1) if ok else 0.0,
        "ttft_ms": _distribution_ms([r.ttft_s for r in ok if r.ttft_s is not None]),
        "tpot_ms": _distribution_ms([t for r in ok if (t := r.tpot_s) is not None]),
        "latency_ms": _distribution_ms([r.latency_s for r in ok]),
        "error_samples": error_samples,
    }
