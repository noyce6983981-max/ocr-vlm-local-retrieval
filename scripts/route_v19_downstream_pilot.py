from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_vlm_retrieval.routing.hybrid_router import (  # noqa: E402
    HybridRouter,
)
from ocr_vlm_retrieval.routing.intervention import (  # noqa: E402
    guard_v18_transition,
)
from ocr_vlm_retrieval.routing.llm_router import LLMRouter  # noqa: E402
from ocr_vlm_retrieval.routing.rule_router import RuleRouter  # noqa: E402
from ocr_vlm_retrieval.routing.transformers_backend import (  # noqa: E402
    TransformersIntentBackend,
)
from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402
from scripts.run_intent_routing_study import percentile  # noqa: E402
from scripts import live_search  # noqa: E402

DEFAULT_QUERIES = (
    ROOT / "data/evaluation/v18/calibration/frozen_calibration_queries.jsonl"
)
DEFAULT_MODEL = ROOT / "models/v19-intent-qwen3-1.7b"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/evaluation/v19/downstream_pilot"
    / "v18_calibration_route_assignments.json"
)


def query_text(row: dict[str, Any]) -> str:
    """Read both legacy pilot rows and reviewed V19 E2E rows."""

    return " ".join(
        str(row.get("query") or row.get("query_text") or "").split()
    )


def development_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select only the tunable partition from the reviewed V19 set."""

    selected = [row for row in rows if row.get("split") == "development"]
    if not selected:
        raise ValueError("development-only pilot found no development rows")
    if any(row.get("split") != "development" for row in selected):
        raise ValueError("development-only pilot must not include holdout rows")
    return selected


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number} must be an object")
            rows.append(payload)
    if not rows:
        raise ValueError("downstream pilot query set must not be empty")
    query_ids = [str(row.get("query_id", "")) for row in rows]
    if not all(query_ids) or len(query_ids) != len(set(query_ids)):
        raise ValueError("downstream pilot query IDs must be non-empty and unique")
    return rows


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def route_rows(
    rows: list[dict[str, Any]],
    hybrid_router: HybridRouter,
    baseline_router: RuleRouter,
) -> tuple[list[dict[str, Any]], list[float]]:
    assignments: list[dict[str, Any]] = []
    latencies: list[float] = []
    for row in rows:
        query = query_text(row)
        if not query:
            raise ValueError(f"empty query for {row.get('query_id')}")
        baseline_decision = baseline_router.route(query)
        started = time.perf_counter()
        decision = hybrid_router.route(query)
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        guarded = guard_v18_transition(
            query=query,
            method="quality_hybrid",
            legacy_route=baseline_decision.route,
            candidate_route=decision.route,
            evidence=(decision.llm.evidence if decision.llm is not None else None),
            extract_strict_entity_term=live_search.extract_strict_entity_term,
            required_search_branches=live_search.required_search_branches,
        )
        assignments.append(
            {
                "query_id": str(row["query_id"]),
                "query": query,
                "query_role": row.get("query_role"),
                "source_item_id": (
                    row.get("source_item_id") or row.get("target_item_id")
                ),
                "neighbor_item_id": row.get("neighbor_item_id"),
                "gold_answerable": row.get("gold_answerable"),
                "content_stratum": row.get("content_stratum"),
                "route_latency_ms": round(latency_ms, 3),
                "legacy_route": baseline_decision.route,
                "calibrated_rule_route": decision.rule.route,
                "hybrid_route": decision.route,
                "route_changed": baseline_decision.route != decision.route,
                "guarded_route": guarded.applied_route,
                "guarded_route_changed": (
                    baseline_decision.route != guarded.applied_route
                ),
                "intervention_guard_reason": guarded.guard_reason,
                "legacy_branches": guarded.legacy_branches,
                "candidate_branches": guarded.candidate_branches,
                "llm_invoked": decision.rule.ambiguous,
                "decision_source": decision.source,
                "reason_codes": list(decision.rule.reason_codes),
                "fallback_error_type": decision.fallback_error_type,
                "guard_reason": decision.guard_reason,
                "evidence": (
                    decision.llm.evidence.to_mapping()
                    if decision.llm is not None
                    else None
                ),
            }
        )
    return assignments, latencies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--prompt-lookup-num-tokens", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--development-only",
        action="store_true",
        help=(
            "Select only split=development rows. This script deliberately "
            "does not provide a holdout-selection option."
        ),
    )
    parser.add_argument("--expected-query-count", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.queries)
    if args.development_only:
        rows = development_rows(rows)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[: args.limit]
    if (
        args.expected_query_count is not None
        and len(rows) != args.expected_query_count
    ):
        raise ValueError(
            "selected query count differs from --expected-query-count: "
            f"{len(rows)} != {args.expected_query_count}"
        )

    backend = TransformersIntentBackend(
        args.model,
        max_new_tokens=args.max_new_tokens,
        local_files_only=True,
        prompt_lookup_num_tokens=args.prompt_lookup_num_tokens,
    )
    llm_router = LLMRouter(backend)
    llm_router.route("找一张蓝色海边照片")
    baseline_router = RuleRouter.legacy()
    hybrid_router = HybridRouter(RuleRouter.v19_calibrated(), llm_router)
    assignments, latencies = route_rows(rows, hybrid_router, baseline_router)
    invoked_count = sum(row["llm_invoked"] for row in assignments)
    changed_count = sum(row["route_changed"] for row in assignments)
    guarded_changed_count = sum(
        row["guarded_route_changed"] for row in assignments
    )
    fallback_count = sum(
        row["decision_source"] == "rule_fallback" for row in assignments
    )
    payload = {
        "study_id": "v19-local-llm-structured-intent-routing",
        "split": (
            "v19_reviewed_development_only"
            if args.development_only
            else "v18_calibration_regression_only"
        ),
        "eligible_for_v19_final_claim": False,
        "source_query_sha256": file_sha256(args.queries),
        "model": backend.name,
        "prompt_lookup_num_tokens": args.prompt_lookup_num_tokens,
        "query_count": len(assignments),
        "llm_invoked_count": invoked_count,
        "llm_call_rate": invoked_count / len(assignments),
        "route_changed_count": changed_count,
        "route_changed_rate": changed_count / len(assignments),
        "guarded_route_changed_count": guarded_changed_count,
        "guarded_route_changed_rate": guarded_changed_count / len(assignments),
        "fallback_count": fallback_count,
        "p50_route_latency_ms": percentile(latencies, 0.50),
        "p95_route_latency_ms": percentile(latencies, 0.95),
        "assignments": assignments,
    }
    write_json_atomic(args.output, payload)
    print(
        "V19 downstream pilot routing: "
        f"n={len(assignments)}, calls={invoked_count}, "
        f"changed={changed_count}, fallbacks={fallback_count}"
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
