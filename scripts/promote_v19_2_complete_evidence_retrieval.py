"""Promote full-corpus exact evidence ahead of global-similarity candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
from ocr_vlm_retrieval.gating.ocr_literals_v19_2 import (  # noqa: E402
    complete_explicit_evidence_item_ids,
    extract_v19_2_literal_groups,
)
from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402
from ocr_vlm_retrieval.runtime.late_interaction import (  # noqa: E402
    positive_retrieval_summary,
)
from scripts.run_v19_downstream_retrieval_pilot import read_json  # noqa: E402

EVALUATION_ROOT = ROOT / "outputs/evaluation/v19_2/automatic_optimization"
DEFAULT_RETRIEVAL = EVALUATION_ROOT / "development_colqwen2_machine.json"
DEFAULT_OUTPUT = EVALUATION_ROOT / "development_evidence_first.json"
DEFAULT_OCR_ROOT = ROOT / "outputs/user_library/ocr/json"
SPLIT = "v19_2_automatic_development_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--ocr-root", type=Path, default=DEFAULT_OCR_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ocr-min-confidence", type=float, default=0.35)
    return parser.parse_args()


def promote_ranking(ranking: list[str], evidence_ids: list[str]) -> list[str]:
    """Prepend complete evidence while preserving both source orders."""

    promoted = list(dict.fromkeys(str(value) for value in evidence_ids))
    promoted_set = set(promoted)
    return [*promoted, *(value for value in ranking if value not in promoted_set)]


def main() -> int:
    args = parse_args()
    retrieval = read_json(args.retrieval)
    if retrieval.get("split") != SPLIT:
        raise ValueError("evidence promotion may read V19.2 development only")
    results = [dict(row) for row in retrieval.get("results", [])]
    if len(results) != 48:
        raise ValueError("expected 48 retrieval rows")
    item_ids = sorted(path.stem for path in args.ocr_root.glob("*.json"))
    lines_by_item = {
        item_id: load_ocr_lines(
            args.ocr_root / f"{item_id}.json",
            minimum_confidence=args.ocr_min_confidence,
        )
        for item_id in item_ids
    }
    promoted_results: list[dict[str, Any]] = []
    promotion_counts: list[int] = []
    for row in results:
        groups = extract_v19_2_literal_groups(str(row["query"]))
        evidence_ids = complete_explicit_evidence_item_ids(groups, lines_by_item)
        ranking = promote_ranking(
            [str(value) for value in row.get("ranking_item_ids", [])],
            evidence_ids,
        )
        relevant = {str(value) for value in row.get("gold_relevant_item_ids", [])}
        relevant_rank = next(
            (
                rank
                for rank, item_id in enumerate(ranking, start=1)
                if item_id in relevant
            ),
            None,
        )
        promotion_counts.append(len(evidence_ids))
        promoted_results.append(
            {
                **row,
                "ranking_item_ids": ranking,
                "scores": [],
                "relevant_rank": relevant_rank,
                "complete_evidence_item_ids": evidence_ids,
            }
        )
    summary = positive_retrieval_summary(promoted_results)
    payload = {
        "schema_version": 1,
        "status": "machine_development_evidence_first_diagnostic_only",
        "method": "complete_explicit_evidence_then_colqwen2",
        "split": SPLIT,
        "eligible_for_final_claim": False,
        "human_review_used": False,
        "future_holdout_opened": False,
        "source_retrieval_sha256": hashlib.sha256(
            args.retrieval.read_bytes()
        ).hexdigest(),
        "source_assignment_sha256": retrieval.get("source_assignment_sha256"),
        "summary": summary,
        "queries_with_complete_evidence": sum(value > 0 for value in promotion_counts),
        "maximum_complete_evidence_pages": max(promotion_counts, default=0),
        "results": promoted_results,
    }
    write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "summary": summary,
                "queries_with_complete_evidence": payload[
                    "queries_with_complete_evidence"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
