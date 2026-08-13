"""Diagnose V19.1 condition completeness on the old V19 development split."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_vlm_retrieval.gating.candidate_verification import (  # noqa: E402
    load_ocr_lines,
)
from ocr_vlm_retrieval.gating.literal_evidence_v19_1 import (  # noqa: E402
    select_v19_1_literal_candidate,
)
from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402
from scripts.evaluate_v19_ocr_literal_override import (  # noqa: E402
    paired_family_bootstrap,
    positive_recall_at_k,
    summarize_by_stratum,
)
from scripts.score_v19_v18_l1_development import (  # noqa: E402
    read_json,
    summarize,
)

EVALUATION_ROOT = ROOT / "outputs/evaluation/v19/selective_intervention"
DEFAULT_RETRIEVAL = EVALUATION_ROOT / "development_colqwen2_v1.json"
DEFAULT_BASELINE = EVALUATION_ROOT / "development_true_v18_l1_top3.json"
DEFAULT_PREDECESSOR = (
    EVALUATION_ROOT / "development_colqwen2_literal_evidence_override.json"
)
DEFAULT_OCR_ROOT = ROOT / "outputs/user_library/ocr/json"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/evaluation/v19_1/condition_completeness"
    / "development_diagnostic.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--predecessor", type=Path, default=DEFAULT_PREDECESSOR)
    parser.add_argument("--ocr-root", type=Path, default=DEFAULT_OCR_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--ocr-min-confidence", type=float, default=0.35)
    parser.add_argument("--ocr-fuzzy-threshold", type=float, default=0.88)
    return parser.parse_args()


def _metric_delta(
    candidate_summary: dict[str, Any], reference_summary: dict[str, Any]
) -> dict[str, float]:
    keys = (
        "positive_selected_relevant",
        "negative_correct_reject_rate",
        "end_to_end_accuracy",
        "positive_recall_at_3",
        "false_accept_rate",
        "false_reject_rate",
    )
    return {
        key: round(float(candidate_summary[key]) - float(reference_summary[key]), 8)
        for key in keys
    }


def _with_recall(payload: dict[str, Any]) -> dict[str, Any]:
    summary = summarize(payload["results"])
    summary["positive_recall_at_3"] = positive_recall_at_k(payload["results"], cutoff=3)
    return summary


def main() -> int:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    retrieval = read_json(args.retrieval)
    if retrieval.get("split") != "development_only":
        raise ValueError("V19.1 diagnostic may read the old V19 development split only")
    baseline_payload = read_json(args.baseline)
    predecessor_payload = read_json(args.predecessor)
    retrieval_by_id = {
        str(row["query_id"]): row for row in retrieval.get("results", [])
    }
    baseline_ids = {str(row["query_id"]) for row in baseline_payload.get("results", [])}
    predecessor_ids = {
        str(row["query_id"]) for row in predecessor_payload.get("results", [])
    }
    if not baseline_ids or baseline_ids != retrieval_by_id.keys():
        raise ValueError("baseline and retrieval query IDs must be identical")
    if predecessor_ids != baseline_ids:
        raise ValueError("predecessor and baseline query IDs must be identical")

    decisions: list[dict[str, Any]] = []
    candidate_results: list[dict[str, Any]] = []
    for baseline in baseline_payload["results"]:
        query_id = str(baseline["query_id"])
        retrieval_row = retrieval_by_id[query_id]
        item_ids = [
            str(item_id)
            for item_id in retrieval_row.get("ranking_item_ids", [])[: args.top_k]
        ]
        lines_by_item = {
            item_id: load_ocr_lines(
                args.ocr_root / f"{item_id}.json",
                minimum_confidence=args.ocr_min_confidence,
            )
            for item_id in item_ids
        }
        decision = select_v19_1_literal_candidate(
            str(retrieval_row["query"]),
            item_ids,
            lines_by_item,
            fuzzy_threshold=args.ocr_fuzzy_threshold,
        )
        if not decision["eligible"]:
            result = {**baseline, "decision_source": "v18_l1_fallback"}
        else:
            selected = decision["selected_item_id"]
            result = {
                **baseline,
                "candidate_item_ids": item_ids,
                "scores": [],
                "selected_item_id": selected,
                "selected_retrieval_rank": (
                    item_ids.index(str(selected)) + 1 if selected is not None else None
                ),
                "accepted": decision["accepted"],
                "top_score": None,
                "decision_source": "v19_1_complete_literal_evidence_override",
            }
        candidate_results.append(result)
        decisions.append(
            {
                "query_id": query_id,
                "query_role": retrieval_row.get("query_role"),
                "gold_answerable": retrieval_row.get("gold_answerable"),
                **decision,
            }
        )

    candidate_payload = {"results": candidate_results}
    baseline_summary = _with_recall(baseline_payload)
    predecessor_summary = _with_recall(predecessor_payload)
    candidate_summary = _with_recall(candidate_payload)
    reason_counts = Counter(str(row["reason"]) for row in decisions)
    decision_source_counts = Counter(
        str(row["decision_source"]) for row in candidate_results
    )
    incomplete_counts = Counter(
        str(source)
        for row in decisions
        for source in row["completeness"]["missing_condition_sources"]
    )
    payload = {
        "status": "old_v19_development_diagnostic_only",
        "method": "v19_1_colqwen2_top3_complete_literal_evidence",
        "split": "development_only",
        "eligible_for_promotion": False,
        "governance": {
            "v19_holdout_read": False,
            "v19_holdout_rerun": False,
            "purpose": "hypothesis screening before fresh V19.1 data",
        },
        "parameters": {
            "top_k": args.top_k,
            "ocr_min_confidence": args.ocr_min_confidence,
            "ocr_fuzzy_threshold": args.ocr_fuzzy_threshold,
            "eligibility_input": "query_text_only",
            "runtime_inputs": ["query_text", "candidate_ocr"],
            "forbidden_runtime_inputs": [
                "content_stratum",
                "gold_answerable",
                "gold_relevant_item_ids",
            ],
        },
        "baseline_v18": baseline_summary,
        "predecessor_v19": predecessor_summary,
        "candidate_v19_1": candidate_summary,
        "delta_vs_v18": _metric_delta(candidate_summary, baseline_summary),
        "delta_vs_v19": _metric_delta(candidate_summary, predecessor_summary),
        "paired_family_bootstrap_vs_v18": paired_family_bootstrap(
            baseline_payload["results"], candidate_results
        ),
        "paired_family_bootstrap_vs_v19": paired_family_bootstrap(
            predecessor_payload["results"], candidate_results
        ),
        "decision_audit": {
            "eligibility_reason_counts": dict(sorted(reason_counts.items())),
            "decision_source_counts": dict(sorted(decision_source_counts.items())),
            "missing_condition_source_counts": dict(sorted(incomplete_counts.items())),
        },
        "baseline_by_stratum": summarize_by_stratum(baseline_payload["results"]),
        "predecessor_by_stratum": summarize_by_stratum(predecessor_payload["results"]),
        "candidate_by_stratum": summarize_by_stratum(candidate_results),
        "decisions": decisions,
        "results": candidate_results,
    }
    write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "baseline_v18": baseline_summary,
                "predecessor_v19": predecessor_summary,
                "candidate_v19_1": candidate_summary,
                "delta_vs_v18": payload["delta_vs_v18"],
                "delta_vs_v19": payload["delta_vs_v19"],
                "decision_audit": payload["decision_audit"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
