"""Evaluate the frozen V19 literal gate on human-reviewed V19.1 development."""

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
from ocr_vlm_retrieval.gating.literal_evidence import (  # noqa: E402
    literal_override_is_eligible,
    select_literal_candidate,
)
from ocr_vlm_retrieval.gating.ocr_literals import OcrLiteralGroup  # noqa: E402
from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402
from scripts.evaluate_v19_ocr_literal_override import (  # noqa: E402
    paired_family_bootstrap,
    positive_recall_at_k,
    summarize_by_stratum,
)
from scripts.run_v19_downstream_retrieval_pilot import read_json  # noqa: E402
from scripts.score_v19_v18_l1_development import summarize  # noqa: E402

EVALUATION_ROOT = ROOT / "outputs/evaluation/v19_1/condition_completeness"
DEFAULT_BASELINE = EVALUATION_ROOT / "development_v18_l1_top3_human_reviewed.json"
DEFAULT_RETRIEVAL = EVALUATION_ROOT / "development_colqwen2_human_reviewed.json"
DEFAULT_OCR_ROOT = ROOT / "outputs/user_library/ocr/json"
DEFAULT_OUTPUT = (
    EVALUATION_ROOT / "development_v19_predecessor_human_reviewed.json"
)
EXPECTED_SPLIT = "v19_1_human_reviewed_development_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--ocr-root", type=Path, default=DEFAULT_OCR_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--ocr-min-confidence", type=float, default=0.35)
    parser.add_argument("--ocr-fuzzy-threshold", type=float, default=0.88)
    return parser.parse_args()


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = summarize(rows)
    result["positive_recall_at_3"] = positive_recall_at_k(rows, cutoff=3)
    return result


def main() -> int:
    args = parse_args()
    if args.top_k != 3:
        raise ValueError("frozen V19 predecessor requires top-k=3")
    baseline_payload = read_json(args.baseline)
    retrieval_payload = read_json(args.retrieval)
    if baseline_payload.get("split") != EXPECTED_SPLIT:
        raise ValueError("baseline must be human-reviewed V19.1 development")
    if retrieval_payload.get("split") != EXPECTED_SPLIT:
        raise ValueError("retrieval must be human-reviewed V19.1 development")
    baseline_sha = str(baseline_payload.get("source_assignment_sha256", ""))
    if not baseline_sha or baseline_sha != retrieval_payload.get(
        "source_assignment_sha256"
    ):
        raise ValueError("baseline and retrieval assignment hashes must match")
    baseline_results = [dict(row) for row in baseline_payload.get("results", [])]
    retrieval_by_id = {
        str(row["query_id"]): dict(row)
        for row in retrieval_payload.get("results", [])
    }
    if {str(row["query_id"]) for row in baseline_results} != retrieval_by_id.keys():
        raise ValueError("baseline and retrieval query IDs must match")

    candidate_results: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for baseline in baseline_results:
        query_id = str(baseline["query_id"])
        retrieval = retrieval_by_id[query_id]
        item_ids = [
            str(item_id)
            for item_id in retrieval.get("ranking_item_ids", [])[: args.top_k]
        ]
        lines_by_item = {
            item_id: load_ocr_lines(
                args.ocr_root / f"{item_id}.json",
                minimum_confidence=args.ocr_min_confidence,
            )
            for item_id in item_ids
        }
        decision = select_literal_candidate(
            str(retrieval["query"]),
            item_ids,
            lines_by_item,
            fuzzy_threshold=args.ocr_fuzzy_threshold,
        )
        groups = tuple(
            OcrLiteralGroup(
                label=str(group["label"]),
                variants=tuple(str(value) for value in group["variants"]),
                source=str(group["source"]),
            )
            for group in decision["constraint_groups"]
        )
        eligible = literal_override_is_eligible(str(retrieval["query"]), groups)
        if not eligible:
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
                "decision_source": "frozen_v19_literal_evidence",
            }
        candidate_results.append(result)
        decisions.append(
            {
                "query_id": query_id,
                "query_role": retrieval["query_role"],
                "gold_answerable": bool(retrieval["gold_answerable"]),
                "override_eligible": eligible,
                **decision,
            }
        )

    baseline_summary = _summary(baseline_results)
    candidate_summary = _summary(candidate_results)
    delta = {
        key: round(float(candidate_summary[key]) - float(baseline_summary[key]), 8)
        for key in (
            "positive_selected_relevant",
            "negative_correct_reject_rate",
            "end_to_end_accuracy",
            "positive_recall_at_3",
            "false_accept_rate",
            "false_reject_rate",
        )
    }
    source_counts = Counter(str(row["decision_source"]) for row in candidate_results)
    payload = {
        "status": "human_reviewed_development_diagnostic_only",
        "method": "frozen_v19_colqwen2_top3_literal_evidence",
        "split": EXPECTED_SPLIT,
        "eligible_for_promotion": False,
        "holdout_opened": False,
        "source_assignment_sha256": baseline_sha,
        "parameters": {
            "top_k": args.top_k,
            "ocr_min_confidence": args.ocr_min_confidence,
            "ocr_fuzzy_threshold": args.ocr_fuzzy_threshold,
            "runtime_inputs": ["query_text", "candidate_ocr"],
        },
        "baseline_v18": baseline_summary,
        "predecessor_v19": candidate_summary,
        "delta_vs_v18": delta,
        "paired_family_bootstrap_vs_v18": paired_family_bootstrap(
            baseline_results, candidate_results
        ),
        "decision_source_counts": dict(sorted(source_counts.items())),
        "baseline_by_stratum": summarize_by_stratum(baseline_results),
        "predecessor_by_stratum": summarize_by_stratum(candidate_results),
        "decisions": decisions,
        "results": candidate_results,
    }
    write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "baseline_v18": baseline_summary,
                "predecessor_v19": candidate_summary,
                "delta_vs_v18": delta,
                "decision_source_counts": payload["decision_source_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
