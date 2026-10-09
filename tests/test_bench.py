from __future__ import annotations

import json
import math

import httpx
import pytest

from inferscale.bench import prompts
from inferscale.bench.__main__ import main as bench_main
from inferscale.bench.client import RequestResult
from inferscale.bench.report import compare_markdown, write_run
from inferscale.bench.runner import RunConfig, run_level
from inferscale.bench.stats import percentile, summarize


def test_percentile_matches_linear_interpolation():
    values = [1, 2, 3, 4]
    assert percentile(values, 50) == 2.5
    assert percentile(values, 0) == 1 and percentile(values, 100) == 4
    assert percentile([5], 99) == 5
    assert math.isnan(percentile([], 50))


def test_summary_excludes_failures_from_latency_and_counts_them():
    results = [
        RequestResult(True, latency_s=1.0, ttft_s=0.1, output_tokens=10),
        RequestResult(True, latency_s=2.0, ttft_s=0.2, output_tokens=10),
        RequestResult(False, latency_s=30.0, error="HTTP 503"),
    ]
    s = summarize(results, wall_time_s=2.0, concurrency=2)
    assert s["successes"] == 2 and s["errors"] == 1 and s["error_rate"] == pytest.approx(0.3333)
    assert s["output_token_throughput_tps"] == 10.0
    assert s["latency_ms"]["p50"] == 1500.0  # the 30 s failure is not in the distribution
    assert s["tpot_ms"]["p50"] == pytest.approx(150.0)  # (1.0-0.1)/9 and (2.0-0.2)/9 -> median
    assert s["error_samples"] == ["HTTP 503"]


def test_summary_of_all_failures_is_valid_json():
    s = summarize([RequestResult(False, 1.0, error="x")], 1.0, 1)
    json.dumps(s, allow_nan=False)


def test_synthetic_prompts_are_distinct_and_deterministic():
    a = prompts.synthetic(20, input_words=64, seed=1)
    assert len({p[-1]["content"] for p in a}) == 20
    assert a == prompts.synthetic(20, input_words=64, seed=1)


def test_jsonl_prompts(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text('{"prompt": "hi"}\n\n{"messages": [{"role": "user", "content": "yo"}]}\n')
    assert prompts.from_jsonl(path) == [
        [{"role": "user", "content": "hi"}],
        [{"role": "user", "content": "yo"}],
    ]
    path.write_text('{"text": "bad"}\n')
    with pytest.raises(ValueError, match="expected"):
        prompts.from_jsonl(path)


async def test_run_level_against_gateway(client: httpx.AsyncClient):
    summary = await run_level(
        client,
        prompts.synthetic(8, 16),
        concurrency=4,
        num_requests=12,
        cfg=RunConfig(max_tokens=6),
        warmup=2,
    )
    assert summary["requests"] == 12 and summary["errors"] == 0
    assert summary["mean_output_tokens"] == 6.0  # ignore_eos forces exact length
    assert summary["ttft_ms"]["p50"] is not None


async def test_run_level_records_errors(client: httpx.AsyncClient):
    summary = await run_level(
        client,
        prompts.synthetic(2, 16),
        concurrency=2,
        num_requests=4,
        cfg=RunConfig(max_tokens=4, backend="nope"),
    )
    assert summary["errors"] == 4
    assert summary["error_samples"][0].startswith("HTTP 400")


def _run(label: str, tps: float) -> dict:
    level = summarize([RequestResult(True, 1.0, 0.1, 10)], 1.0, 4)
    level["output_token_throughput_tps"] = tps
    return {
        "meta": {
            "label": label,
            "timestamp": "2026-10-08T00:00:00Z",
            "max_tokens": 10,
            "input_words": 16,
        },
        "levels": [level],
    }


def test_reports_written_and_compared(tmp_path):
    paths = write_run(_run("vllm", 1234.5), tmp_path)
    assert json.loads(paths["json"].read_text())["meta"]["label"] == "vllm"
    assert "| 4 |" in paths["md"].read_text()
    assert paths["csv"].read_text().splitlines()[0].startswith("concurrency,")

    table = compare_markdown([_run("vllm", 1234.5), _run("triton", 1100.0)])
    assert "vllm tok/s" in table and "triton tok/s" in table
    assert "| 4 | 1,234.5 |" in table


def test_cli_compare(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(_run("vllm", 10.0)))
    b.write_text(json.dumps(_run("triton", 9.0)))
    assert bench_main(["compare", str(a), str(b), "--out", str(tmp_path / "cmp.md")]) == 0
    assert "triton tok/s" in capsys.readouterr().out
    assert (tmp_path / "cmp.md").exists()
