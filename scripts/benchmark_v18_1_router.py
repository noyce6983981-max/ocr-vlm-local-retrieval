"""Benchmark warm routing latency, queueing symptoms and fallback rates."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_vlm_retrieval.routing.service_client import (  # noqa: E402
    ServiceRouteDecision,
    request_v19_route_with_fallback,
)
from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402

DEFAULT_QUERIES = ROOT / "data/evaluation/v19/pilot/queries_reviewed.csv"
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/v19/router_benchmark.json"


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(probability * len(ordered)) - 1),
    )
    return ordered[index]


def benchmark_concurrency(
    *,
    service_url: str,
    queries: list[str],
    concurrency: int,
    request_count: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    selected = [queries[index % len(queries)] for index in range(request_count)]

    def invoke(query: str) -> tuple[float, ServiceRouteDecision]:
        started = time.perf_counter()
        decision = request_v19_route_with_fallback(
            service_url,
            query,
            timeout_seconds=timeout_seconds,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return elapsed_ms, decision

    wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(invoke, selected))
    wall_seconds = time.perf_counter() - wall_started
    latencies = [elapsed for elapsed, _ in results]
    decisions = [decision for _, decision in results]
    server_latencies = [
        decision.route_latency_ms
        for decision in decisions
        if decision.route_latency_ms is not None
    ]
    queue_estimates = [
        max(0.0, elapsed - server_latency)
        for elapsed, decision in results
        if (server_latency := decision.route_latency_ms) is not None
    ]
    return {
        "concurrency": concurrency,
        "request_count": request_count,
        "wall_seconds": round(wall_seconds, 4),
        "throughput_requests_per_second": round(request_count / wall_seconds, 4),
        "client_latency_ms": {
            "p50": round(percentile(latencies, 0.50), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "max": round(max(latencies, default=0.0), 3),
        },
        "server_route_latency_ms": {
            "p50": round(percentile(server_latencies, 0.50), 3),
            "p95": round(percentile(server_latencies, 0.95), 3),
        },
        "estimated_transport_and_queue_ms": {
            "p50": round(percentile(queue_estimates, 0.50), 3),
            "p95": round(percentile(queue_estimates, 0.95), 3),
        },
        "llm_count": sum(decision.llm_invoked for decision in decisions),
        "fallback_count": sum(
            decision.source == "rule_fallback" for decision in decisions
        ),
        "timeout_count": sum(
            decision.fallback_error_type == "TimeoutError"
            for decision in decisions
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-url", default="http://127.0.0.1:8765")
    parser.add_argument("--query", action="append")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.requests < 1:
        raise ValueError("--requests must be positive")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    if any(value < 1 for value in args.concurrency):
        raise ValueError("--concurrency values must be positive")
    queries = args.query or [
        "找右上角带校徽的封面",
        "哪份申请表右下角盖了红色印章？",
        "找山景照片里路牌写着海拔三千米的页面",
        "查一下材料里登记的最终验收日期",
    ]
    report = {
        "schema_version": 1,
        "study_id": "v19-selective-intervention-router-load-diagnostic",
        "service_url": args.service_url,
        "timeout_seconds": args.timeout_seconds,
        "results": [
            benchmark_concurrency(
                service_url=args.service_url,
                queries=queries,
                concurrency=value,
                request_count=args.requests,
                timeout_seconds=args.timeout_seconds,
            )
            for value in args.concurrency
        ],
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
