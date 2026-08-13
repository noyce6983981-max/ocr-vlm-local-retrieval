from __future__ import annotations

import argparse
import json
import math
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
DEFAULT_BASELINE_DIR = ROOT / "outputs/evaluation/v18/calibration/retrieval/raw"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/evaluation/v19/downstream_pilot/retrieval/raw"
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
        acceptance.get("quality_hybrid") if isinstance(acceptance, dict) else None
    )
    timings = payload.get("timings", {})
    routing = payload.get("v18_1_intent_routing", {})
    return {
        "top1_item_id": rankings[0] if rankings else None,
        "top3_item_ids": rankings[:3],
        "accepted": (decision.get("accepted") if isinstance(decision, dict) else None),
        "acceptance_reason": (
            decision.get("reason") if isinstance(decision, dict) else None
        ),
        "acceptance_signal_name": (
            decision.get("signal_name") if isinstance(decision, dict) else None
        ),
        "acceptance_signal": (
            decision.get("signal") if isinstance(decision, dict) else None
        ),
        "acceptance_threshold": (
            decision.get("threshold") if isinstance(decision, dict) else None
        ),
        "attribute_coverage_guard_applied": (
            bool(decision.get("attribute_coverage_guard_applied"))
            if isinstance(decision, dict)
            else False
        ),
        "upstream_gate_passed": (
            decision.get("upstream_gate_passed") if isinstance(decision, dict) else None
        ),
        "executed_branches": payload.get("executed_branches"),
        "exploratory_query": payload.get("exploratory_query"),
        "retrieval_route": payload.get("retrieval_route"),
        "core_total_seconds": (
            timings.get("total_seconds") if isinstance(timings, dict) else None
        ),
        "route_latency_ms": (
            routing.get("route_latency_ms") if isinstance(routing, dict) else 0.0
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
            row["baseline"]["top1_item_id"] != row["candidate"]["top1_item_id"]
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
    positives = [
        row
        for row in records
        if row["query_role"]
        in {"positive", "answerable_positive", "paraphrase_positive"}
    ]
    negatives = [
        row
        for row in records
        if row["query_role"]
        in {"single_condition_hard_negative", "unanswerable_neighbor"}
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


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(probability * len(ordered)) - 1),
    )
    return ordered[index]


def summarize_e2e(records: list[dict[str, Any]], snapshot_field: str) -> dict[str, Any]:
    positives = [row for row in records if bool(row.get("gold_answerable"))]
    negatives = [row for row in records if not bool(row.get("gold_answerable"))]
    hard_negatives = [
        row
        for row in negatives
        if row.get("query_role") == "single_condition_hard_negative"
    ]
    neighbor_negatives = [
        row for row in negatives if row.get("query_role") == "unanswerable_neighbor"
    ]

    def accepted(row: dict[str, Any]) -> bool:
        return bool(row[snapshot_field].get("accepted"))

    def rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    def relevant_ids(row: dict[str, Any]) -> set[str]:
        explicit = {
            str(item_id)
            for item_id in row.get("gold_relevant_item_ids", [])
            if str(item_id).strip()
        }
        if explicit:
            return explicit
        source = str(row.get("source_item_id") or "").strip()
        return {source} if source else set()

    positive_top1 = sum(
        accepted(row) and row[snapshot_field].get("top1_item_id") in relevant_ids(row)
        for row in positives
    )
    positive_recall3 = sum(
        bool(relevant_ids(row) & set(row[snapshot_field].get("top3_item_ids", [])))
        for row in positives
    )
    negative_false_accepts = sum(accepted(row) for row in negatives)
    hard_negative_false_accepts = sum(accepted(row) for row in hard_negatives)
    neighbor_false_accepts = sum(accepted(row) for row in neighbor_negatives)
    wall_seconds = [
        float(value)
        for row in records
        if (value := row[snapshot_field].get("wall_total_seconds")) is not None
    ]
    return {
        "answerable_count": len(positives),
        "negative_count": len(negatives),
        "positive_top1_accuracy": rate(positive_top1, len(positives)),
        "end_to_end_accuracy": rate(
            positive_top1 + len(negatives) - negative_false_accepts,
            len(records),
        ),
        "recall_at_3": rate(positive_recall3, len(positives)),
        "negative_correct_reject_rate": rate(
            len(negatives) - negative_false_accepts, len(negatives)
        ),
        "negative_far": rate(negative_false_accepts, len(negatives)),
        "hard_negative_far": rate(hard_negative_false_accepts, len(hard_negatives)),
        "neighbor_negative_far": rate(neighbor_false_accepts, len(neighbor_negatives)),
        "warm_wall_p95_seconds": percentile(wall_seconds, 0.95),
    }


def with_route_latency(
    snapshot: dict[str, Any], route_latency_ms: float
) -> dict[str, Any]:
    enriched = dict(snapshot)
    core = enriched.get("core_total_seconds")
    if core is not None:
        enriched["wall_total_seconds"] = float(core) + route_latency_ms / 1000
    return enriched


def run_baseline(
    assignment: dict[str, Any],
    *,
    output_path: Path,
    library_dir: Path,
    attribute_policy: Path,
    timeout: int,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file():
        existing = read_json(output_path)
        if (
            existing.get("query") == assignment["query"]
            and existing.get("retrieval_route") == assignment["legacy_route"]
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
            f"baseline retrieval failed for {assignment['query_id']}:\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    return read_json(output_path)


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
        if assignment["guarded_route"] == "entity_exact"
        else assignment["guarded_route"]
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
        str(assignment["guarded_route"]),
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
    parser.add_argument(
        "--materialize-baseline",
        action="store_true",
        help=(
            "Create missing legacy quality-hybrid payloads from precomputed "
            "caches. This is not the V18 L1 verifier baseline."
        ),
    )
    parser.add_argument("--baseline-suffix", default="_v18_frozen")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    route_payload = read_json(args.assignments)
    assignments = list(route_payload["assignments"])
    changed = [row for row in assignments if row["guarded_route_changed"]]
    if args.limit_changed is not None:
        if args.limit_changed < 1:
            raise ValueError("limit-changed must be positive")
        selected_ids = {row["query_id"] for row in changed[: args.limit_changed]}
    else:
        selected_ids = {row["query_id"] for row in changed}

    records: list[dict[str, Any]] = []
    for assignment in assignments:
        query_id = str(assignment["query_id"])
        baseline_path = args.baseline_dir / f"{query_id}{args.baseline_suffix}.json"
        baseline = (
            run_baseline(
                assignment,
                output_path=baseline_path,
                library_dir=args.library_dir,
                attribute_policy=args.attribute_policy,
                timeout=args.timeout,
            )
            if args.materialize_baseline
            else read_json(baseline_path)
        )
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
        baseline_snapshot = retrieval_snapshot(baseline)
        candidate_snapshot = with_route_latency(
            retrieval_snapshot(candidate),
            float(assignment.get("route_latency_ms", 0.0)),
        )
        records.append(
            {
                **assignment,
                "baseline_effective_route": baseline.get("retrieval_route"),
                "candidate_effective_route": candidate.get("retrieval_route"),
                "baseline_source_rank": source_rank(baseline, source_item_id),
                "candidate_source_rank": source_rank(candidate, source_item_id),
                "baseline": baseline_snapshot,
                "candidate": candidate_snapshot,
                "candidate_payload": (
                    str(candidate_path.resolve().relative_to(ROOT))
                    if candidate_path is not None
                    else None
                ),
            }
        )

    baseline_metrics = summarize_source_neighbor(records, "baseline_source_rank")
    candidate_metrics = summarize_source_neighbor(records, "candidate_source_rank")
    baseline_e2e = summarize_e2e(records, "baseline")
    candidate_e2e = summarize_e2e(records, "candidate")
    source_delta = {
        key: float(candidate_metrics[key]) - float(baseline_metrics[key])
        for key in baseline_metrics
        if key not in {"positive_count", "hard_negative_count"}
    }
    e2e_delta = {
        key: float(candidate_e2e[key]) - float(baseline_e2e[key])
        for key in baseline_e2e
        if key not in {"answerable_count", "negative_count"}
    }
    changed_records = [row for row in records if row["query_id"] in selected_ids]
    report = {
        "study_id": "v19-local-llm-structured-intent-routing",
        "baseline_method": "legacy_quality_hybrid_attribute_scaffold",
        "baseline_warning": (
            "These retrieval payloads run with rerank_top_k=0 and therefore "
            "must not be described as the frozen V18 L1 verifier method."
        ),
        "split": route_payload.get("split"),
        "eligible_for_v19_final_claim": False,
        "query_count": len(records),
        "changed_query_count": len(changed_records),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta": source_delta,
        "e2e": {
            "baseline": baseline_e2e,
            "candidate": candidate_e2e,
            "delta": e2e_delta,
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
        "end_to_end_delta="
        f"{e2e_delta['end_to_end_accuracy']:.4f}, "
        "hard_negative_far1_delta="
        f"{source_delta['hard_negative_source_far_at_1']:.4f}"
    )
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
