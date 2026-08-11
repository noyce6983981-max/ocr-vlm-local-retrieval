from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_vlm_retrieval.routing.evaluator import (  # noqa: E402
    RoutingSample,
    evaluate_routing,
)
from ocr_vlm_retrieval.routing.schema import validate_route  # noqa: E402
from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402

DEFAULT_BASELINE = ROOT / "outputs/evaluation/v19/pilot/b0_rule_only.json"
DEFAULT_CANDIDATE = (
    ROOT / "outputs/evaluation/v19/pilot/qwen3_1_7b_b2_hybrid.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/v19/pilot/b2_vs_b0.json"


def read_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"result must be a JSON object: {path}")
    return payload


def sample_from_prediction(row: dict[str, Any]) -> RoutingSample:
    predicted = row.get("predicted_route")
    return RoutingSample(
        gold_route=validate_route(str(row["gold_route"])),
        predicted_route=(
            validate_route(str(predicted)) if predicted is not None else None
        ),
        rule_route=validate_route(str(row.get("rule_route", predicted))),
        llm_invoked=bool(row.get("llm_invoked", False)),
        used_fallback=bool(row.get("used_fallback", False)),
        fallback_error_type=row.get("fallback_error_type"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = read_payload(args.baseline)
    candidate = read_payload(args.candidate)
    baseline_by_id = {
        row["query_id"]: row for row in baseline["predictions"]
    }
    candidate_by_id = {
        row["query_id"]: row for row in candidate["predictions"]
    }
    if set(baseline_by_id) != set(candidate_by_id):
        raise ValueError("baseline and candidate query IDs differ")

    invoked_ids = [
        query_id
        for query_id, row in candidate_by_id.items()
        if row.get("llm_invoked")
    ]
    baseline_subset = evaluate_routing(
        [sample_from_prediction(baseline_by_id[query_id]) for query_id in invoked_ids]
    )
    candidate_subset = evaluate_routing(
        [sample_from_prediction(candidate_by_id[query_id]) for query_id in invoked_ids]
    )
    fixed = [
        query_id
        for query_id in invoked_ids
        if not baseline_by_id[query_id]["correct"]
        and candidate_by_id[query_id]["correct"]
    ]
    regressed = [
        query_id
        for query_id in invoked_ids
        if baseline_by_id[query_id]["correct"]
        and not candidate_by_id[query_id]["correct"]
    ]
    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    result = {
        "baseline_method": baseline["method"],
        "candidate_method": candidate["method"],
        "query_count": len(candidate_by_id),
        "overall_accuracy_delta": (
            candidate_metrics["accuracy"] - baseline_metrics["accuracy"]
        ),
        "overall_macro_f1_delta": (
            candidate_metrics["macro_f1"] - baseline_metrics["macro_f1"]
        ),
        "invocation_subset_count": len(invoked_ids),
        "invocation_subset_baseline": baseline_subset.to_mapping(),
        "invocation_subset_candidate": candidate_subset.to_mapping(),
        "invocation_subset_macro_f1_delta": (
            candidate_subset.macro_f1 - baseline_subset.macro_f1
        ),
        "fixed_query_ids": fixed,
        "regressed_query_ids": regressed,
    }
    write_json_atomic(args.output, result)
    print(
        "B2 vs B0: "
        f"accuracy_delta={result['overall_accuracy_delta']:.4f}, "
        f"macro_f1_delta={result['overall_macro_f1_delta']:.4f}, "
        f"subset_macro_f1_delta={result['invocation_subset_macro_f1_delta']:.4f}, "
        f"fixed={len(fixed)}, regressed={len(regressed)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
