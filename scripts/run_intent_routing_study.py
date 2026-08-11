from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_vlm_retrieval.routing.evaluator import (  # noqa: E402
    RoutingSample,
    evaluate_routing,
)
from ocr_vlm_retrieval.routing.hybrid_router import HybridRouter  # noqa: E402
from ocr_vlm_retrieval.routing.llm_router import LLMRouter  # noqa: E402
from ocr_vlm_retrieval.routing.rule_router import RuleRouter  # noqa: E402
from ocr_vlm_retrieval.routing.schema import validate_route  # noqa: E402
from ocr_vlm_retrieval.routing.transformers_backend import (  # noqa: E402
    TransformersIntentBackend,
)
from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402

DEFAULT_QUERIES = ROOT / "data/evaluation/v19/pilot/queries_reviewed.csv"
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/v19/pilot/b0_rule_only.json"
DEFAULT_MODEL = ROOT / "models/v19-intent-qwen3-0.6b"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_queries(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"query_id", "family_id", "gold_route", "query_text"}
    if not rows:
        raise ValueError("query set must not be empty")
    missing = required - set(rows[0])
    if missing:
        raise ValueError("query set missing columns: " + ", ".join(sorted(missing)))
    query_ids = [row["query_id"].strip() for row in rows]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query_id values must be unique")
    for row in rows:
        row["gold_route"] = validate_route(row["gold_route"].strip())
        row["query_text"] = " ".join(row["query_text"].split())
        if not row["query_text"]:
            raise ValueError(f"empty query_text for {row['query_id']}")
    return rows


def split_name(rows: list[dict[str, str]]) -> str:
    statuses = {row.get("status", "").strip() for row in rows}
    recognized = {
        "human_reviewed_pilot_not_final",
        "primary_reviewed_formal_calibration",
        "primary_reviewed_formal_holdout",
        "double_reviewed_formal_calibration",
        "double_reviewed_formal_holdout",
        "adjudicated_formal_calibration_30pct_double_review",
        "adjudicated_formal_holdout_30pct_double_review",
    }
    if len(statuses) == 1 and next(iter(statuses)) in recognized:
        return next(iter(statuses))
    return "pilot_draft_not_final"


def validate_holdout_claim(
    *, query_path: Path, split: str, claim_path: Path | None, mode: str
) -> None:
    if split != "adjudicated_formal_holdout_30pct_double_review":
        return
    if claim_path is None:
        raise PermissionError("formal holdout requires the one-shot claim receipt")
    payload = json.loads(claim_path.read_text(encoding="utf-8"))
    if payload.get("status") != "v19_one_shot_holdout_claimed":
        raise PermissionError("V19 holdout claim is not valid")
    if payload.get("holdout_query_sha256") != file_sha256(query_path):
        raise PermissionError("V19 holdout claim references different queries")
    if mode not in payload.get("authorized_modes", []):
        raise PermissionError(f"mode is not authorized by holdout claim: {mode}")


def run_rule_only(rows: list[dict[str, str]]) -> dict[str, Any]:
    router = RuleRouter.legacy()
    samples: list[RoutingSample] = []
    predictions: list[dict[str, Any]] = []
    for row in rows:
        start = time.perf_counter()
        decision = router.route(row["query_text"])
        latency_ms = (time.perf_counter() - start) * 1000
        gold_route = validate_route(row["gold_route"])
        samples.append(
            RoutingSample(
                gold_route=gold_route,
                predicted_route=decision.route,
                rule_route=decision.route,
                llm_invoked=False,
            )
        )
        predictions.append(
            {
                "query_id": row["query_id"],
                "family_id": row["family_id"],
                "query_text": row["query_text"],
                "gold_route": gold_route,
                "predicted_route": decision.route,
                "correct": decision.route == gold_route,
                "ambiguous": decision.ambiguous,
                "reason_codes": list(decision.reason_codes),
                "latency_ms": latency_ms,
            }
        )
    metrics = evaluate_routing(samples)
    return {
        "study_id": "v19-local-llm-structured-intent-routing",
        "split": split_name(rows),
        "method": "B0_rule_only",
        "query_count": len(rows),
        "gold_route_counts": dict(
            Counter(row["gold_route"] for row in rows)
        ),
        "metrics": metrics.to_mapping(),
        "predictions": predictions,
    }


def run_human_oracle(rows: list[dict[str, str]]) -> dict[str, Any]:
    samples: list[RoutingSample] = []
    predictions: list[dict[str, Any]] = []
    for row in rows:
        gold_route = validate_route(row["gold_route"])
        samples.append(
            RoutingSample(
                gold_route=gold_route,
                predicted_route=gold_route,
                rule_route=gold_route,
                llm_invoked=False,
            )
        )
        predictions.append(
            {
                "query_id": row["query_id"],
                "family_id": row["family_id"],
                "query_text": row["query_text"],
                "gold_route": gold_route,
                "predicted_route": gold_route,
                "correct": True,
                "llm_invoked": False,
                "latency_ms": 0.0,
            }
        )
    metrics = evaluate_routing(samples)
    return {
        "study_id": "v19-local-llm-structured-intent-routing",
        "split": split_name(rows),
        "method": "B3_human_route_oracle",
        "diagnostic_only": True,
        "query_count": len(rows),
        "gold_route_counts": dict(Counter(row["gold_route"] for row in rows)),
        "metrics": metrics.to_mapping(),
        "predictions": predictions,
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def gpu_memory_gib() -> dict[str, float | None]:
    try:
        import torch
    except ImportError:
        return {"allocated_gib": None, "peak_allocated_gib": None}
    if not torch.cuda.is_available():
        return {"allocated_gib": None, "peak_allocated_gib": None}
    return {
        "allocated_gib": torch.cuda.memory_allocated() / 1024**3,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }


def reset_gpu_peak() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def run_model_method(
    rows: list[dict[str, str]],
    *,
    mode: str,
    model_path: str,
    max_new_tokens: int,
    load_in_4bit: bool,
    prompt_lookup_num_tokens: int | None,
) -> dict[str, Any]:
    rule_router = (
        RuleRouter.legacy()
        if mode == "llm_only"
        else RuleRouter.v19_calibrated()
    )
    backend = TransformersIntentBackend(
        model_path,
        max_new_tokens=max_new_tokens,
        load_in_4bit=load_in_4bit,
        local_files_only=True,
        prompt_lookup_num_tokens=prompt_lookup_num_tokens,
    )
    llm_router = LLMRouter(backend)
    hybrid_router = HybridRouter(rule_router, llm_router)

    warmup_started = time.perf_counter()
    llm_router.route("找一张蓝色海边照片")
    warmup_ms = (time.perf_counter() - warmup_started) * 1000
    reset_gpu_peak()

    samples: list[RoutingSample] = []
    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []
    for row in rows:
        query = row["query_text"]
        gold_route = validate_route(row["gold_route"])
        rule_decision = rule_router.route(query)
        started = time.perf_counter()
        llm_invoked = mode == "llm_only" or rule_decision.ambiguous
        used_fallback = False
        fallback_error_type: str | None = None
        evidence: dict[str, bool | float | None] | None = None

        if mode == "llm_only":
            try:
                llm_decision = llm_router.route(query)
                predicted_route = llm_decision.route
                evidence = llm_decision.evidence.to_mapping()
            except Exception as error:
                predicted_route = None
                fallback_error_type = type(error).__name__
        else:
            hybrid_decision = hybrid_router.route(query)
            predicted_route = hybrid_decision.route
            used_fallback = hybrid_decision.source == "rule_fallback"
            fallback_error_type = hybrid_decision.fallback_error_type
            if hybrid_decision.llm is not None:
                evidence = hybrid_decision.llm.evidence.to_mapping()

        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        samples.append(
            RoutingSample(
                gold_route=gold_route,
                predicted_route=predicted_route,
                rule_route=rule_decision.route,
                llm_invoked=llm_invoked,
                used_fallback=used_fallback,
                fallback_error_type=fallback_error_type,
            )
        )
        predictions.append(
            {
                "query_id": row["query_id"],
                "family_id": row["family_id"],
                "query_text": query,
                "gold_route": gold_route,
                "rule_route": rule_decision.route,
                "predicted_route": predicted_route,
                "correct": predicted_route == gold_route,
                "llm_invoked": llm_invoked,
                "used_fallback": used_fallback,
                "fallback_error_type": fallback_error_type,
                "evidence": evidence,
                "latency_ms": latency_ms,
            }
        )

    metrics = evaluate_routing(samples)
    return {
        "study_id": "v19-local-llm-structured-intent-routing",
        "split": split_name(rows),
        "method": "B1_llm_only" if mode == "llm_only" else "B2_hybrid",
        "model": backend.name,
        "query_count": len(rows),
        "gold_route_counts": dict(
            Counter(row["gold_route"] for row in rows)
        ),
        "metrics": metrics.to_mapping(),
        "operational": {
            "cold_warmup_ms": warmup_ms,
            "p50_route_latency_ms": percentile(latencies, 0.50),
            "p95_route_latency_ms": percentile(latencies, 0.95),
            "max_route_latency_ms": max(latencies, default=0.0),
            **gpu_memory_gib(),
        },
        "predictions": predictions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--mode",
        choices=("rule_only", "llm_only", "hybrid", "human_oracle"),
        default="rule_only",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--prompt-lookup-num-tokens", type=int)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-route-limit", type=int)
    parser.add_argument("--holdout-claim", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_queries(args.queries)
    source_split = split_name(rows)
    if source_split == "adjudicated_formal_holdout_30pct_double_review" and (
        args.limit is not None or args.per_route_limit is not None
    ):
        raise ValueError("formal holdout cannot be truncated")
    validate_holdout_claim(
        query_path=args.queries,
        split=source_split,
        claim_path=args.holdout_claim,
        mode=args.mode,
    )
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[: args.limit]
    if args.per_route_limit is not None:
        if args.per_route_limit < 1:
            raise ValueError("per-route-limit must be positive")
        selected: list[dict[str, str]] = []
        counts: Counter[str] = Counter()
        for row in rows:
            route = row["gold_route"]
            if counts[route] < args.per_route_limit:
                selected.append(row)
                counts[route] += 1
        rows = selected
    if args.mode == "rule_only":
        payload = run_rule_only(rows)
        output = args.output or DEFAULT_OUTPUT
    elif args.mode == "human_oracle":
        payload = run_human_oracle(rows)
        output = args.output or DEFAULT_OUTPUT.with_name("b3_human_oracle.json")
    else:
        payload = run_model_method(
            rows,
            mode=args.mode,
            model_path=args.model,
            max_new_tokens=args.max_new_tokens,
            load_in_4bit=args.load_in_4bit,
            prompt_lookup_num_tokens=args.prompt_lookup_num_tokens,
        )
        filename = "b1_llm_only.json" if args.mode == "llm_only" else "b2_hybrid.json"
        output = args.output or DEFAULT_OUTPUT.with_name(filename)
    write_json_atomic(output, payload)
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)
    print(
        f"V19 {payload['method']} study: "
        f"n={len(rows)}, accuracy={metrics['accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
