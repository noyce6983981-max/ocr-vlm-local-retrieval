from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
DEFAULT_ASSIGNMENTS = (
    ROOT
    / "outputs/evaluation/v19/downstream_pilot"
    / "v18_calibration_route_assignments.json"
)
DEFAULT_BASELINE_DIR = (
    ROOT / "outputs/evaluation/v18/calibration/retrieval/raw"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs/evaluation/v19/downstream_pilot/retrieval/raw"
)
DEFAULT_REPORT = (
    ROOT / "outputs/evaluation/v19/downstream_pilot/retrieval_comparison.json"
)
DEFAULT_LIBRARY = ROOT / "outputs/user_library"
DEFAULT_ATTRIBUTE_POLICY = ROOT / "config/v17_attribute_coverage.json"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def ranking_ids(payload: dict[str, Any]) -> list[str]:
    rankings = payload.get("rankings", {})
    rows = rankings.get("quality_hybrid", []) if isinstance(rankings, dict) else []
    return [str(row["item_id"]) for row in rows if row.get("item_id")]


def source_rank(payload: dict[str, Any], source_item_id: str) -> int | None:
    try:
        return ranking_ids(payload).index(source_item_id) + 1
    except ValueError:
        return None


def retrieval_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract user-visible and policy-sensitive fields for paired A/B."""

    rankings = ranking_ids(payload)
    acceptance = payload.get("acceptance", {})
    decision = (
        acceptance.get("quality_hybrid")
        if isinstance(acceptance, dict)
        else None
    )
    timings = payload.get("timings", {})
    routing = payload.get("v18_1_intent_routing", {})
    return {
        "top1_item_id": rankings[0] if rankings else None,
        "top3_item_ids": rankings[:3],
        "accepted": (
            decision.get("accepted") if isinstance(decision, dict) else None
        ),
        "acceptance_reason": (
            decision.get("reason") if isinstance(decision, dict) else None
        ),
        "acceptance_signal_name": (
            decision.get("signal_name")
            if isinstance(decision, dict)
            else None
        ),
        "acceptance_signal": (
            decision.get("signal") if isinstance(decision, dict) else None
        ),
        "acceptance_threshold": (
            decision.get("threshold") if isinstance(decision, dict) else None
        ),
        "executed_branches": payload.get("executed_branches"),
        "exploratory_query": payload.get("exploratory_query"),
        "retrieval_route": payload.get("retrieval_route"),
        "core_total_seconds": (
            timings.get("total_seconds") if isinstance(timings, dict) else None
        ),
        "route_latency_ms": (
            routing.get("route_latency_ms")
            if isinstance(routing, dict)
            else 0.0
        ),
        "wall_total_seconds": (
            routing.get("wall_total_seconds")
            if isinstance(routing, dict)
            else timings.get("total_seconds")
            if isinstance(timings, dict)
            else None
        ),
    }


def summarize_behavior(records: list[dict[str, Any]]) -> dict[str, Any]:
    transitions = Counter(
        (
            str(row["baseline"]["retrieval_route"]),
            str(row["candidate"]["retrieval_route"]),
        )
        for row in records
    )
    return {
        "top1_changed_count": sum(
            row["baseline"]["top1_item_id"]
            != row["candidate"]["top1_item_id"]
            for row in records
        ),
        "acceptance_changed_count": sum(
            row["baseline"]["accepted"] != row["candidate"]["accepted"]
            for row in records
        ),
        "executed_branches_changed_count": sum(
            row["baseline"]["executed_branches"]
            != row["candidate"]["executed_branches"]
            for row in records
        ),
        "exploratory_changed_count": sum(
            row["baseline"]["exploratory_query"]
            != row["candidate"]["exploratory_query"]
            for row in records
        ),
        "route_transition_counts": {
            f"{source}->{target}": count
            for (source, target), count in sorted(transitions.items())
        },
    }


def summarize_source_neighbor(
    records: list[dict[str, Any]],
    rank_field: str,
) -> dict[str, Any]:
    positives = [row for row in records if row["query_role"] == "positive"]
    negatives = [
        row
        for row in records
        if row["query_role"] == "single_condition_hard_negative"
    ]

    def rate(rows: list[dict[str, Any]], cutoff: int) -> float:
        if not rows:
            return 0.0
        matched = sum(
            row[rank_field] is not None and int(row[rank_field]) <= cutoff
            for row in rows
        )
        return matched / len(rows)

    return {
        "positive_count": len(positives),
        "hard_negative_count": len(negatives),
        "positive_source_hit_at_1": rate(positives, 1),
        "positive_source_hit_at_3": rate(positives, 3),
        "positive_source_hit_at_10": rate(positives, 10),
        "hard_negative_source_far_at_1": rate(negatives, 1),
        "hard_negative_source_far_at_3": rate(negatives, 3),
        "hard_negative_source_far_at_10": rate(negatives, 10),
    }


def run_candidate(
    assignment: dict[str, Any],
    *,
    output_path: Path,
    library_dir: Path,
    attribute_policy: Path,
    timeout: int,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_effective_route = (
        "mixed"
        if assignment["hybrid_route"] == "entity_exact"
        else assignment["hybrid_route"]
    )
    if output_path.is_file():
        existing = read_json(output_path)
        if (
            existing.get("query") == assignment["query"]
            and existing.get("retrieval_route") == expected_effective_route
        ):
            return existing
    command = [
        sys.executable,
        str(ROOT / "scripts/v18_1_live_search.py"),
        str(assignment["query"]),
        "--library-dir",
        str(library_dir),
        "--output",
        str(output_path),
        "--method",
        "quality_hybrid",
        "--rerank-top-k",
        "0",
        "--attribute-policy",
        str(attribute_policy),
        "--retrieval-route",
        str(assignment["hybrid_route"]),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"candidate retrieval failed for {assignment['query_id']}:\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    payload = read_json(output_path)
    if payload.get("query") != assignment["query"]:
        raise ValueError(f"candidate query mismatch for {assignment['query_id']}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--library-dir", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument(
        "--attribute-policy", type=Path, default=DEFAULT_ATTRIBUTE_POLICY
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--limit-changed", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    route_payload = read_json(args.assignments)
    assignments = list(route_payload["assignments"])
    changed = [row for row in assignments if row["route_changed"]]
    if args.limit_changed is not None:
        if args.limit_changed < 1:
            raise ValueError("limit-changed must be positive")
        selected_ids = {
            row["query_id"] for row in changed[: args.limit_changed]
        }
    else:
        selected_ids = {row["query_id"] for row in changed}

    records: list[dict[str, Any]] = []
    for assignment in assignments:
        query_id = str(assignment["query_id"])
        baseline_path = args.baseline_dir / f"{query_id}_v17.json"
        baseline = read_json(baseline_path)
        if baseline.get("query") != assignment["query"]:
            raise ValueError(f"baseline query mismatch for {query_id}")
        candidate = baseline
        candidate_path: Path | None = None
        if query_id in selected_ids:
            candidate_path = args.output_dir / f"{query_id}_b21.json"
            candidate = run_candidate(
                assignment,
                output_path=candidate_path,
                library_dir=args.library_dir,
                attribute_policy=args.attribute_policy,
                timeout=args.timeout,
            )
            if candidate.get("library_revision") != baseline.get("library_revision"):
                raise ValueError(f"library revision mismatch for {query_id}")
        source_item_id = str(assignment["source_item_id"])
        records.append(
            {
                **assignment,
                "baseline_effective_route": baseline.get("retrieval_route"),
                "candidate_effective_route": candidate.get("retrieval_route"),
                "baseline_source_rank": source_rank(baseline, source_item_id),
                "candidate_source_rank": source_rank(candidate, source_item_id),
                "baseline": retrieval_snapshot(baseline),
                "candidate": retrieval_snapshot(candidate),
                "candidate_payload": (
                    str(candidate_path.resolve().relative_to(ROOT))
                    if candidate_path is not None
                    else None
                ),
            }
        )

    baseline_metrics = summarize_source_neighbor(records, "baseline_source_rank")
    candidate_metrics = summarize_source_neighbor(records, "candidate_source_rank")
    changed_records = [row for row in records if row["query_id"] in selected_ids]
    report = {
        "study_id": "v19-local-llm-structured-intent-routing",
        "split": "v18_calibration_regression_only",
        "eligible_for_v19_final_claim": False,
        "query_count": len(records),
        "changed_query_count": len(changed_records),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta": {
            key: candidate_metrics[key] - baseline_metrics[key]
            for key in baseline_metrics
            if key not in {"positive_count", "hard_negative_count"}
        },
        "changed_subset": {
            "baseline": summarize_source_neighbor(
                changed_records, "baseline_source_rank"
            ),
            "candidate": summarize_source_neighbor(
                changed_records, "candidate_source_rank"
            ),
        },
        "behavior": summarize_behavior(records),
        "records": records,
    }
    from ocr_vlm_retrieval.runtime.cache import write_json_atomic

    write_json_atomic(args.report, report)
    print(
        "V19 downstream retrieval pilot: "
        f"n={len(records)}, changed={len(changed_records)}, "
        "positive_top1_delta="
        f"{report['delta']['positive_source_hit_at_1']:.4f}, "
        "hard_negative_far1_delta="
        f"{report['delta']['hard_negative_source_far_at_1']:.4f}"
    )
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
