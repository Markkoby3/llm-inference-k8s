"""CLI: ``inferscale-bench run ...`` and ``inferscale-bench compare ...``."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import platform
import sys
from pathlib import Path
from typing import Any

import httpx

from inferscale import __version__
from inferscale.bench import prompts as prompt_sets
from inferscale.bench.report import compare_markdown, load_run, run_markdown, write_run
from inferscale.bench.runner import RunConfig, make_client, run_level


def _int_list(value: str) -> list[int]:
    try:
        levels = [int(v) for v in value.split(",") if v.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {value!r}"
        ) from None
    if not levels or any(v < 1 for v in levels):
        raise argparse.ArgumentTypeError("concurrency levels must be positive integers")
    return levels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inferscale-bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="benchmark one endpoint across concurrency levels")
    run.add_argument("--url", default="http://localhost:8080", help="gateway base URL")
    run.add_argument("--backend", help="value for X-InferScale-Backend (default: gateway default)")
    run.add_argument("--label", help="name for this run in reports (default: backend name)")
    run.add_argument("--concurrency", type=_int_list, default=[1, 4, 16, 64])
    run.add_argument("--requests", type=int, default=200, help="measured requests per level")
    run.add_argument("--warmup", type=int, default=10, help="discarded requests before each level")
    run.add_argument("--max-tokens", type=int, default=256)
    run.add_argument("--input-words", type=int, default=256, help="synthetic prompt length")
    run.add_argument("--prompts", help="JSONL prompt file instead of synthetic prompts")
    run.add_argument("--request-rate", type=float, help="open-loop Poisson arrivals (req/s)")
    run.add_argument("--no-ignore-eos", action="store_true", help="let generations stop naturally")
    run.add_argument("--timeout", type=float, default=600.0)
    run.add_argument("--gpu", help="GPU description recorded with the results, e.g. 'A10G 24GB'")
    run.add_argument("--notes", default="")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out", default="benchmarks/results")

    compare = sub.add_parser("compare", help="side-by-side table from saved run JSON files")
    compare.add_argument("files", nargs="+")
    compare.add_argument("--out", help="write the table here as well as stdout")
    return parser


async def _served_model(client: httpx.AsyncClient) -> str:
    try:
        response = await client.get("/v1/models")
        return response.json()["data"][0]["id"]
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        return "unknown"


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    prompts = (
        prompt_sets.from_jsonl(args.prompts)
        if args.prompts
        else prompt_sets.synthetic(max(args.requests, 64), args.input_words, args.seed)
    )
    cfg = RunConfig(
        max_tokens=args.max_tokens,
        ignore_eos=not args.no_ignore_eos,
        backend=args.backend,
        request_rate=args.request_rate,
        seed=args.seed,
    )
    meta = {
        "label": args.label or args.backend or "default",
        "url": args.url,
        "backend": args.backend,
        "gpu": args.gpu,
        "notes": args.notes,
        "max_tokens": args.max_tokens,
        "input_words": args.input_words,
        "prompt_source": args.prompts or "synthetic",
        "requests_per_level": args.requests,
        "warmup": args.warmup,
        "ignore_eos": cfg.ignore_eos,
        "request_rate": args.request_rate,
        "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": f"inferscale-bench {__version__}",
        "python": platform.python_version(),
    }

    levels = []
    async with make_client(args.url, max(args.concurrency), args.timeout) as client:
        meta["model"] = await _served_model(client)
        for concurrency in args.concurrency:
            print(f"[{meta['label']}] concurrency={concurrency} ...", file=sys.stderr, flush=True)
            summary = await run_level(
                client, prompts, concurrency, args.requests, cfg, warmup=args.warmup
            )
            print(
                f"  {summary['output_token_throughput_tps']:.1f} tok/s, "
                f"TTFT p95 {summary['ttft_ms']['p95']} ms, "
                f"latency p95 {summary['latency_ms']['p95']} ms, "
                f"errors {summary['errors']}/{summary['requests']}",
                file=sys.stderr,
                flush=True,
            )
            levels.append(summary)
    return {"meta": meta, "levels": levels}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        run = asyncio.run(run_benchmark(args))
        paths = write_run(run, args.out)
        print(run_markdown(run))
        print(f"wrote {', '.join(str(p) for p in paths.values())}", file=sys.stderr)
        failed = sum(level["errors"] for level in run["levels"])
        return 1 if failed and failed == sum(level["requests"] for level in run["levels"]) else 0

    table = compare_markdown(load_run(f) for f in args.files)
    print(table)
    if args.out:
        Path(args.out).write_text(table, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
