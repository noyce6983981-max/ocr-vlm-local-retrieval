"""Evaluate a deterministic OCR-literal override on V19 development only."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Mapping, Sequence
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
from ocr_vlm_retrieval.gating.ocr_literals import (  # noqa: E402
    OcrLiteralGroup,
)
from ocr_vlm_retrieval.gating.literal_evidence import (  # noqa: E402
    literal_override_is_eligible,
    select_literal_candidate,
)
from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402
from scripts.score_v19_v18_l1_development import (  # noqa: E402
    read_json,
    summarize,
)

DEFAULT_RETRIEVAL = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention"
    / "development_colqwen2_v1.json"
)
DEFAULT_BASELINE = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention"
    / "development_true_v18_l1_top3.json"
)
DEFAULT_OCR_ROOT = ROOT / "outputs/user_library/ocr/json"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/evaluation/v19/selective_intervention"
    / "development_colqwen2_literal_evidence_override.json"
)


def summarize_by_stratum(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    strata = sorted({str(row.get("content_stratum", "unknown")) for row in results})
    return {
        stratum: summarize(
            [row for row in results if str(row.get("content_stratum")) == stratum]
        )
        for stratum in strata
    }


def positive_recall_at_k(results: Sequence[Mapping[str, Any]], *, cutoff: int) -> float:
    positives = [row for row in results if bool(row.get("gold_answerable"))]
    if not positives:
        return 0.0
    hits = sum(
        bool(
            {str(item_id) for item_id in row.get("candidate_item_ids", [])[:cutoff]}
            & {str(item_id) for item_id in row.get("gold_relevant_item_ids", [])}
        )
        for row in positives
    )
    return hits / len(positives)


def _decision_is_correct(row: Mapping[str, Any]) -> bool:
    if not bool(row.get("gold_answerable")):
        return not bool(row.get("accepted"))
    selected = row.get("selected_item_id")
    relevant = {str(value) for value in row.get("gold_relevant_item_ids", [])}
    return selected is not None and str(selected) in relevant


def paired_family_bootstrap(
    baseline_results: Sequence[Mapping[str, Any]],
    candidate_results: Sequence[Mapping[str, Any]],
    *,
    samples: int = 10_000,
    seed: int = 20_260_812,
) -> dict[str, Any]:
    """Bootstrap paired E2E gain while preserving four-query family clusters."""

    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    baseline_by_id = {str(row["query_id"]): row for row in baseline_results}
    candidate_by_id = {str(row["query_id"]): row for row in candidate_results}
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("paired bootstrap requires identical query IDs")
    families: dict[str, list[tuple[bool, bool]]] = {}
    for query_id, baseline in baseline_by_id.items():
        family_id = query_id.rsplit("_q", 1)[0]
        families.setdefault(family_id, []).append(
            (
                _decision_is_correct(baseline),
                _decision_is_correct(candidate_by_id[query_id]),
            )
        )
    family_rows = list(families.values())
    if not family_rows:
        raise ValueError("paired bootstrap requires at least one family")
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(samples):
        sampled_pairs = [pair for _ in family_rows for pair in rng.choice(family_rows)]
        delta = sum(
            int(candidate_correct) - int(baseline_correct)
            for baseline_correct, candidate_correct in sampled_pairs
        ) / len(sampled_pairs)
        deltas.append(delta)
    deltas.sort()

    def percentile(probability: float) -> float:
        index = round((len(deltas) - 1) * probability)
        return deltas[index]

    paired_rows = [pair for family in family_rows for pair in family]
    observed_delta = sum(
        int(candidate_correct) - int(baseline_correct)
        for baseline_correct, candidate_correct in paired_rows
    ) / len(paired_rows)
    return {
        "unit": "query_family",
        "family_count": len(family_rows),
        "samples": samples,
        "seed": seed,
        "observed_e2e_delta": observed_delta,
        "confidence_interval_95": [percentile(0.025), percentile(0.975)],
        "improved_query_count": sum(
            not baseline_correct and candidate_correct
            for baseline_correct, candidate_correct in paired_rows
        ),
        "regressed_query_count": sum(
            baseline_correct and not candidate_correct
            for baseline_correct, candidate_correct in paired_rows
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--ocr-root", type=Path, default=DEFAULT_OCR_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--ocr-min-confidence", type=float, default=0.35)
    parser.add_argument("--ocr-fuzzy-threshold", type=float, default=0.88)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    retrieval = read_json(args.retrieval)
    if retrieval.get("split") != "development_only":
        raise ValueError("OCR override may read V19 development only")
    baseline_payload = read_json(args.baseline)
    retrieval_by_id = {
        str(row["query_id"]): row for row in retrieval.get("results", [])
    }
    decisions: list[dict[str, Any]] = []
    candidate_results: list[dict[str, Any]] = []
    for baseline in baseline_payload.get("results", []):
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
        decision = select_literal_candidate(
            str(retrieval_row["query"]),
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
        eligible = literal_override_is_eligible(str(retrieval_row["query"]), groups)
        if not eligible:
            candidate_results.append({**baseline, "decision_source": "v18_l1"})
            decisions.append(
                {
                    "query_id": query_id,
                    "query_role": retrieval_row.get("query_role"),
                    "gold_answerable": retrieval_row.get("gold_answerable"),
                    "override_eligible": False,
                    **decision,
                }
            )
            continue
        candidate_results.append(
            {
                **baseline,
                "candidate_item_ids": item_ids,
                "scores": [],
                "selected_item_id": decision["selected_item_id"],
                "selected_retrieval_rank": (
                    item_ids.index(str(decision["selected_item_id"])) + 1
                    if decision["selected_item_id"] is not None
                    else None
                ),
                "accepted": decision["accepted"],
                "top_score": None,
                "decision_source": "deterministic_literal_evidence_override",
            }
        )
        decisions.append(
            {
                "query_id": query_id,
                "query_role": retrieval_row.get("query_role"),
                "gold_answerable": retrieval_row.get("gold_answerable"),
                "override_eligible": True,
                **decision,
            }
        )
    baseline_summary = summarize(baseline_payload["results"])
    candidate_summary = summarize(candidate_results)
    baseline_summary["positive_recall_at_3"] = positive_recall_at_k(
        baseline_payload["results"], cutoff=3
    )
    candidate_summary["positive_recall_at_3"] = positive_recall_at_k(
        candidate_results, cutoff=3
    )
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
    payload = {
        "status": "development_diagnostic_only",
        "method": "colqwen2_top3_deterministic_literal_evidence_override",
        "split": "development_only",
        "eligible_for_final_claim": False,
        "parameters": {
            "top_k": args.top_k,
            "ocr_min_confidence": args.ocr_min_confidence,
            "ocr_fuzzy_threshold": args.ocr_fuzzy_threshold,
            "date_match_policy": "exact_variant_only",
            "eligibility_input": "query_text_only",
            "eligibility_policy": (
                "exact Chinese name/entity; high-precision date, phone suffix, or "
                "translation alias; or an explicit 浏览 request with a named literal"
            ),
            "forbidden_runtime_inputs": ["content_stratum", "gold_answerable"],
        },
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "delta": delta,
        "promotion_check": {
            "minimum_e2e_gain": 0.05,
            "minimum_recall_at_3_gain": 0.05,
            "e2e_gain_passed": delta["end_to_end_accuracy"] >= 0.05,
            "recall_at_3_gain_passed": delta["positive_recall_at_3"] >= 0.05,
            "far_non_inferior": delta["false_accept_rate"] <= 0.0,
            "positive_non_inferior": delta["positive_selected_relevant"] >= 0.0,
        },
        "paired_family_bootstrap": paired_family_bootstrap(
            baseline_payload["results"], candidate_results
        ),
        "baseline_by_stratum": summarize_by_stratum(baseline_payload["results"]),
        "candidate_by_stratum": summarize_by_stratum(candidate_results),
        "decisions": decisions,
        "results": candidate_results,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps({"baseline": baseline_summary}, ensure_ascii=False, indent=2))
    print(json.dumps({"candidate": candidate_summary}, ensure_ascii=False, indent=2))
    print(json.dumps({"delta": delta}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
