#!/usr/bin/env python3
"""Small end-to-end check for the deployed API."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


def load_demo(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    item = data[0] if isinstance(data, list) else data
    if "payload" in item and isinstance(item["payload"], dict):
        item = item["payload"]
    return {
        "db_id": item.get("db_id", item.get("database", "")),
        "question": item.get("question", item.get("query", "")),
        "evidence": item.get("evidence", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--demo-file", default="app/examples/demo_queries.json")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    payload = load_demo(Path(args.demo_file))
    with httpx.Client(
        base_url=base,
        timeout=240.0,
        trust_env=False,
    ) as client:
        health = client.get("/api/health")
        health.raise_for_status()
        databases = client.get("/api/databases")
        databases.raise_for_status()
        fast = client.post("/api/query", json={**payload, "mode": "fast"})
        fast.raise_for_status()
        accurate = client.post("/api/query", json={**payload, "mode": "accurate"})
        accurate.raise_for_status()
    h = health.json()
    f = fast.json()
    a = accurate.json()
    if not h.get("api_ready") or not h.get("model_ready"):
        raise RuntimeError("API or model is not ready.")
    database_rows = databases.json()
    database_count = (
        len(database_rows)
        if isinstance(database_rows, list)
        else int(database_rows.get("database_count", len(database_rows.get("databases", []))))
    )
    if database_count != 80:
        raise RuntimeError(f"Expected 80 databases, got {database_count}.")
    candidates = a.get("candidates", [])
    candidate_count = len(candidates) if isinstance(candidates, list) else int(
        a.get("vote", {}).get("candidate_count", a.get("candidate_count", 0))
    )
    if candidate_count != 8:
        raise RuntimeError("Accurate mode did not return eight candidates.")
    if f.get("status") != "success" or not (
        f.get("selected_candidate") or {}
    ).get("sql"):
        raise RuntimeError(f"Fast mode did not return executable SQL: {f.get('status')}")
    if a.get("status") != "success" or not (
        a.get("selected_candidate") or {}
    ).get("sql"):
        raise RuntimeError(
            f"Accurate mode did not return executable SQL: {a.get('status')}"
        )
    print(f"API ready: {h.get('api_ready', h.get('status'))}")
    print(f"Model ready: {h.get('model_ready')}")
    print(f"Database count: {database_count}")
    print(f"Fast status / latency: {f.get('status')} / {f.get('timings', {}).get('total_ms', f.get('latency_ms'))} ms")
    vote = a.get("vote", {})
    print(f"Accurate status / candidate count / vote support / latency: {a.get('status')} / {candidate_count} / {vote.get('support', a.get('vote_support'))} / {a.get('timings', {}).get('total_ms', a.get('latency_ms'))} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
