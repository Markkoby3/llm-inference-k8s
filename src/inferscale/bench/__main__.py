"""CLI: ``inferscale-bench run | agent | retrieval | routing | compare``."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import platform
import sys
from pathlib import Path
from typing import Any

import httpx

from inferscale import __version__
from inferscale.bench import prompts as prompt_sets
from inferscale.bench.agent import agent_markdown, run_agent_level
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
    run.add_argument(
        "--model",
        help="model name sent with each request (default: discovered from /v1/models); "
        "lets the harness target an engine's own OpenAI server directly",
    )
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

    agent = sub.add_parser("agent", help="load-test the multi-step RAG agent endpoint")
    agent.add_argument("--url", default="http://localhost:8080")
    agent.add_argument("--backend")
    agent.add_argument("--label")
    agent.add_argument("--concurrency", type=_int_list, default=[1, 4, 16])
    agent.add_argument("--requests", type=int, default=48, help="requests per level")
    agent.add_argument("--max-tokens", type=int, default=256)
    agent.add_argument("--questions", default="benchmarks/rag/questions.jsonl")
    agent.add_argument("--timeout", type=float, default=600.0)
    agent.add_argument("--gpu")
    agent.add_argument("--out", default="benchmarks/results")

    retrieval = sub.add_parser("retrieval", help="recall@k / MRR of the vector search (offline)")
    retrieval.add_argument("--questions", default="benchmarks/rag/questions.jsonl")
    retrieval.add_argument("--corpus", nargs="*", help="files or directories (default: docs)")
    retrieval.add_argument("--embedder", default="hashing")
    retrieval.add_argument("--index", default="auto")
    retrieval.add_argument("--out", help="also write the JSON report here")

    routing = sub.add_parser(
        "routing", help="compare replica routing policies in a prefix-cache simulation"
    )
    routing.add_argument("--replicas", type=int, default=4)
    routing.add_argument("--prefixes", type=int, default=64)
    routing.add_argument("--zipf", type=float, default=1.1, help="prefix popularity skew")
    routing.add_argument("--cache", type=int, default=8, help="cached prefixes per replica")
    routing.add_argument("--slots", type=int, default=8, help="batch slots per replica")
    routing.add_argument("--requests", type=int, default=2000)
    routing.add_argument("--concurrency", type=int, default=28)
    routing.add_argument("--seed", type=int, default=0)

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
        meta["model"] = args.model or await _served_model(client)
        if meta["model"] != "unknown":
            cfg.model = meta["model"]
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


async def run_agent_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from inferscale.rag.evaluate import load_questions

    questions = [q.question for q in load_questions(args.questions)]
    meta = {
        "label": args.label or args.backend or "default",
        "url": args.url,
        "backend": args.backend,
        "gpu": args.gpu,
        "max_tokens": args.max_tokens,
        "questions": args.questions,
        "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": f"inferscale-bench {__version__}",
    }
    levels = []
    async with make_client(args.url, max(args.concurrency), args.timeout) as client:
        meta["model"] = await _served_model(client)
        for concurrency in args.concurrency:
            print(f"[agent {meta['label']}] concurrency={concurrency} ...", file=sys.stderr)
            levels.append(
                await run_agent_level(
                    client, questions, concurrency, args.requests, args.backend, args.max_tokens
                )
            )
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

    if args.command == "agent":
        run = asyncio.run(run_agent_benchmark(args))
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"agent-{run['meta']['label']}-{run['meta']['timestamp'].replace(':', '')}"
        (out / f"{stem}.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        (out / f"{stem}.md").write_text(agent_markdown(run), encoding="utf-8")
        print(agent_markdown(run))
        return 1 if all(lv["errors"] == lv["requests"] for lv in run["levels"]) else 0

    if args.command == "retrieval":
        from inferscale.rag.evaluate import evaluate, load_questions
        from inferscale.rag.store import DocumentStore

        store = DocumentStore.from_paths(args.corpus or None, args.embedder, args.index)
        report = evaluate(store, load_questions(args.questions))
        text = json.dumps(report, indent=2)
        print(text)
        if args.out:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
        return 0

    if args.command == "routing":
        from inferscale.bench.routing import Workload, compare, markdown

        workload = Workload(
            replicas=args.replicas,
            prefixes=args.prefixes,
            zipf_s=args.zipf,
            cache_per_replica=args.cache,
            slots_per_replica=args.slots,
            requests=args.requests,
            concurrency=args.concurrency,
            seed=args.seed,
        )
        print(markdown(asyncio.run(compare(workload)), workload))
        return 0

    table = compare_markdown(load_run(f) for f in args.files)
    print(table)
    if args.out:
        Path(args.out).write_text(table, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
