"""Write benchmark runs as JSON (source of truth), CSV and Markdown, and compare runs."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def run_markdown(run: dict[str, Any]) -> str:
    meta = run["meta"]
    lines = [
        f"### {meta['label']}",
        "",
        f"Backend `{meta.get('backend') or 'default'}` · model `{meta.get('model', '?')}` · "
        f"GPU {meta.get('gpu') or 'not recorded'} · {meta['max_tokens']} output tokens · "
        f"~{meta['input_words']} input words · {meta['timestamp']}",
        "",
        "| Concurrency | Req/s | Output tok/s | TTFT p50 (ms) | TTFT p95 (ms) | TPOT p50 (ms) "
        "| Latency p50 (ms) | Latency p95 (ms) | Errors |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level in run["levels"]:
        lines.append(
            f"| {level['concurrency']} | {_fmt(level['request_throughput_rps'], 2)} "
            f"| {_fmt(level['output_token_throughput_tps'])} "
            f"| {_fmt(level['ttft_ms']['p50'])} | {_fmt(level['ttft_ms']['p95'])} "
            f"| {_fmt(level['tpot_ms']['p50'], 2)} "
            f"| {_fmt(level['latency_ms']['p50'])} | {_fmt(level['latency_ms']['p95'])} "
            f"| {level['errors']}/{level['requests']} |"
        )
    return "\n".join(lines) + "\n"


def write_run(run: dict[str, Any], out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{run['meta']['label']}-{run['meta']['timestamp'].replace(':', '').replace('-', '')}"
    paths = {
        "json": out / f"{stem}.json",
        "csv": out / f"{stem}.csv",
        "md": out / f"{stem}.md",
    }
    paths["json"].write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    paths["md"].write_text(run_markdown(run), encoding="utf-8")

    fields = [
        "concurrency",
        "requests",
        "errors",
        "request_throughput_rps",
        "output_token_throughput_tps",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
        "tpot_p50_ms",
        "tpot_p95_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
    ]
    with paths["csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for level in run["levels"]:
            writer.writerow(
                {
                    "concurrency": level["concurrency"],
                    "requests": level["requests"],
                    "errors": level["errors"],
                    "request_throughput_rps": level["request_throughput_rps"],
                    "output_token_throughput_tps": level["output_token_throughput_tps"],
                    **{
                        f"{metric}_{p}_ms": level[f"{metric}_ms"][p]
                        for metric in ("ttft", "tpot", "latency")
                        for p in ("p50", "p95", "p99")
                        if f"{metric}_{p}_ms" in fields
                    },
                }
            )
    return paths


def compare_markdown(runs: Iterable[dict[str, Any]]) -> str:
    """Side-by-side table: one row per concurrency level, one column group per run."""
    runs = list(runs)
    labels = [r["meta"]["label"] for r in runs]
    by_level: dict[int, dict[str, dict[str, Any]]] = {}
    for run in runs:
        for level in run["levels"]:
            by_level.setdefault(level["concurrency"], {})[run["meta"]["label"]] = level

    header = ["Concurrency"]
    for label in labels:
        header += [f"{label} tok/s", f"{label} TTFT p95 (ms)", f"{label} p95 latency (ms)"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---:|" * len(header)]
    for concurrency in sorted(by_level):
        row = [str(concurrency)]
        for label in labels:
            level = by_level[concurrency].get(label)
            if level is None:
                row += ["-", "-", "-"]
                continue
            row += [
                _fmt(level["output_token_throughput_tps"]),
                _fmt(level["ttft_ms"]["p95"]),
                _fmt(level["latency_ms"]["p95"]),
            ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def load_run(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
