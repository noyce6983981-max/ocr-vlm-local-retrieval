"""Prepare fresh V19.1 source families disjoint from V18 and all V19 families."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_v19_selective_intervention_review import (  # noqa: E402
    _atomic_jsonl,
    _load_jsonl,
    build_source_families,
    load_excluded_item_ids,
    load_ocr_character_counts,
)

DEFAULT_PROTOCOL = (
    ROOT / "data/evaluation/v19_1_condition_completeness/protocol_draft.json"
)
DEFAULT_V18_EVALUATION_ROOT = ROOT / "data/evaluation/v18"
DEFAULT_V19_QUERIES = (
    ROOT / "records/private/v19/selective_intervention/reviewed_queries.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "records/private/v19_1/condition_completeness"
    / "source_families_draft.jsonl"
)
DEFAULT_RECEIPT = (
    ROOT
    / "records/private/v19_1/condition_completeness"
    / "source_family_selection_receipt.json"
)


def _v19_source_ids(path: Path) -> set[str]:
    excluded: set[str] = set()
    for row in _load_jsonl(path):
        for key in ("target_item_id", "neighbor_item_id"):
            item_id = str(row.get(key, "")).strip()
            if item_id:
                excluded.add(item_id)
    if not excluded:
        raise ValueError("V19 exclusion list is empty")
    return excluded


def _rename_families(families: list[dict[str, Any]]) -> None:
    counts: Counter[str] = Counter()
    for family in families:
        stratum = str(family["content_stratum"])
        counts[stratum] += 1
        family["family_id"] = f"v19_1_cc_{stratum}_{counts[stratum]:02d}"
        family["selection_status"] = "fresh_source_bound_pending_query_draft"
        family["review_status"] = "pending_human_review_not_frozen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--v18-evaluation-root", type=Path, default=DEFAULT_V18_EVALUATION_ROOT
    )
    parser.add_argument("--v19-queries", type=Path, default=DEFAULT_V19_QUERIES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--seed", default="v19-1-condition-completeness-source-v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    library_root = args.library_root.resolve()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8-sig"))
    manifest_path = library_root / "outputs/user_library/manifest.jsonl"
    manifest = _load_jsonl(manifest_path)
    v18_excluded = load_excluded_item_ids(args.v18_evaluation_root.resolve())
    if not v18_excluded:
        raise ValueError(
            "V18 exclusion list is empty; provide the archived V18 evaluation root"
        )
    v19_excluded = _v19_source_ids(args.v19_queries)
    all_excluded = v18_excluded | v19_excluded
    ocr_counts = load_ocr_character_counts(
        library_root / "outputs/user_library/ocr/summary.csv"
    )
    families = build_source_families(
        manifest,
        stratum_counts=protocol["content_strata"],
        excluded_item_ids=all_excluded,
        ocr_character_counts=ocr_counts,
        seed=args.seed,
    )
    _rename_families(families)

    manifest_by_id = {str(row["item_id"]): row for row in manifest}
    selected_ids = {
        str(family[role]["item_id"])
        for family in families
        for role in ("target", "neighbor")
    }
    selected_groups = {
        str(manifest_by_id[item_id].get("hard_negative_group", ""))
        for item_id in selected_ids
    }
    excluded_groups = {
        str(row.get("hard_negative_group", ""))
        for row in manifest
        if str(row.get("item_id", "")) in all_excluded
    }
    overlap_ids = selected_ids & all_excluded
    overlap_groups = selected_groups & excluded_groups
    if overlap_ids or overlap_groups:
        raise AssertionError("V19.1 source selection overlaps predecessor families")

    _atomic_jsonl(args.output, families)
    split_counts = Counter(str(family["split"]) for family in families)
    stratum_counts = Counter(str(family["content_stratum"]) for family in families)
    receipt = {
        "schema_version": 1,
        "study_id": protocol["study_id"],
        "status": "fresh_source_families_selected_pending_human_review",
        "seed": args.seed,
        "family_count": len(families),
        "source_item_count": len(selected_ids),
        "split_counts": dict(sorted(split_counts.items())),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "excluded_v18_item_count": len(v18_excluded),
        "excluded_v19_item_count": len(v19_excluded),
        "selected_predecessor_item_overlap_count": len(overlap_ids),
        "selected_predecessor_group_overlap_count": len(overlap_groups),
        "human_review_required": True,
        "method_frozen": False,
        "holdout_opened": False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
