#!/usr/bin/env python3
"""Lightweight concurrent API benchmark; writes one JSON summary per run."""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx


def load_demos(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = (
        raw
        if isinstance(raw, list)
        else raw.get("queries", raw.get("examples", []))
    )
    result = []
    for item in rows:
        if isinstance(item.get("payload"), dict):
            item = item["payload"]
        result.append(
            {
                "db_id": item.get("db_id", item.get("database", "")),
                "question": item.get("question", item.get("query", "")),
                "evidence": item.get("evidence", ""),
            }
        )
    if not result:
        raise ValueError(f"No demo queries found in {path}")
    return result


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((p / 100) * (len(values) - 1))))
    return values[index]


async def main_async(args: argparse.Namespace) -> int:
    demos = load_demos(Path(args.demo_file))
    sem = asyncio.Semaphore(args.concurrency)
    base = args.base_url.rstrip("/")
    rows: list[dict] = []
    started = time.perf_counter()

    async with httpx.AsyncClient(
        base_url=base,
        timeout=args.timeout,
        trust_env=False,
    ) as client:

        async def run_one(index: int) -> None:
            async with sem:
                payload = {**demos[index % len(demos)], "mode": args.mode}
                t0 = time.perf_counter()
                try:
                    response = await client.post("/api/query", json=payload)
                    response.raise_for_status()
                    body = response.json()
                    timings = body.get("timings", {})
                    rows.append(
                        {
                            "ok": body.get("status") == "success",
                            "latency_ms": (time.perf_counter() - t0) * 1000,
                            "generation_ms": timings.get("generation_ms", 0),
                            "execution_ms": (
                                timings.get("candidate_execution_ms", 0)
                                + timings.get("display_execution_ms", 0)
                            ),
                        }
                    )
                except Exception as error:
                    rows.append(
                        {
                            "ok": False,
                            "latency_ms": (time.perf_counter() - t0) * 1000,
                            "error": str(error),
                        }
                    )

        await asyncio.gather(
            *(run_one(index) for index in range(args.requests))
        )
    elapsed = time.perf_counter() - started
    latencies = [r["latency_ms"] for r in rows]
    successes = [r for r in rows if r["ok"]]
    summary = {
        "request_count": args.requests,
        "success_count": len(successes),
        "error_count": len(rows) - len(successes),
        "qps": args.requests / elapsed if elapsed else 0.0,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_p99_ms": percentile(latencies, 99),
        "mean_generation_ms": (
            statistics.mean(row["generation_ms"] for row in successes)
            if successes
            else 0.0
        ),
        "mean_execution_ms": (
            statistics.mean(row["execution_ms"] for row in successes)
            if successes
            else 0.0
        ),
    }
    output = Path(f"app/results/benchmark_{args.mode}_c{args.concurrency}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0 if summary["error_count"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fast", "accurate"), default="fast")
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--demo-file", default="app/examples/demo_queries.json")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    if args.requests < 1:
        parser.error("--requests must be positive")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
