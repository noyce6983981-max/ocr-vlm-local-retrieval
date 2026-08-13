"""Audit frozen V18 retrieval recall on V19.1 development."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402
from scripts.run_v19_downstream_retrieval_pilot import (  # noqa: E402
    ranking_ids,
    read_json,
)

DEFAULT_ASSIGNMENTS = (
    ROOT
    / "outputs/evaluation/v19_1/condition_completeness"
    / "development_assignments_machine.json"
)
DEFAULT_RETRIEVAL_DIR = (
    ROOT
    / "outputs/evaluation/v19_1/condition_completeness/retrieval"
    / "development_v18_frozen"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/evaluation/v19_1/condition_completeness"
    / "development_v18_retrieval_audit.json"
)
DEVELOPMENT_SPLITS = frozenset(
    {
        "v19_1_machine_draft_development_only",
        "v19_1_human_reviewed_development_only",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--retrieval-dir", type=Path, default=DEFAULT_RETRIEVAL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if row["gold_answerable"]]

    def recall(cutoff: int) -> float:
        if not positives:
            return 0.0
        hits = sum(
            bool(row["source_rank"] is not None and row["source_rank"] <= cutoff)
            for row in positives
        )
        return float(hits / len(positives))

    return {
        "query_count": len(rows),
        "positive_count": len(positives),
        "positive_recall_at_1": recall(1),
        "positive_recall_at_3": recall(3),
        "positive_recall_at_10": recall(10),
        "positive_recall_at_20": recall(20),
    }


def main() -> int:
    args = parse_args()
    payload = read_json(args.assignments)
    split = str(payload.get("split", ""))
    if split not in DEVELOPMENT_SPLITS:
        raise ValueError("audit may read V19.1 development only")
    rows: list[dict[str, Any]] = []
    for assignment in payload.get("assignments", []):
        query_id = str(assignment["query_id"])
        retrieval = read_json(args.retrieval_dir / f"{query_id}_v18_frozen.json")
        if retrieval.get("query") != assignment.get("query"):
            raise ValueError(f"retrieval query mismatch for {query_id}")
        ranking = ranking_ids(retrieval)
        source_id = str(assignment["source_item_id"])
        rows.append(
            {
                "query_id": query_id,
                "family_id": assignment["family_id"],
                "query_role": assignment["query_role"],
                "content_stratum": assignment["content_stratum"],
                "gold_answerable": bool(assignment["gold_answerable"]),
                "source_item_id": source_id,
                "source_rank": (
                    ranking.index(source_id) + 1 if source_id in ranking else None
                ),
                "ranking_item_ids": ranking[:20],
            }
        )
    by_stratum_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum_rows[str(row["content_stratum"])].append(row)
    result = {
        "status": (
            "human_reviewed_development_diagnostic_only"
            if split == "v19_1_human_reviewed_development_only"
            else "machine_draft_development_diagnostic_only"
        ),
        "split": split,
        "eligible_for_promotion": False,
        "holdout_opened": False,
        "summary": _summary(rows),
        "by_stratum": {
            stratum: _summary(group)
            for stratum, group in sorted(by_stratum_rows.items())
        },
        "results": rows,
    }
    write_json_atomic(args.output, result)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(result["by_stratum"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
