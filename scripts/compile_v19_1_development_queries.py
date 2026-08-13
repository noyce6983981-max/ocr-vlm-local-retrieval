"""Compile machine-draft V19.1 development queries without opening holdout OCR."""

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

from ocr_vlm_retrieval.routing.rule_router import RuleRouter  # noqa: E402
from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402

DEFAULT_FAMILIES = (
    ROOT
    / "records/private/v19_1/condition_completeness"
    / "source_families_draft.jsonl"
)
DEFAULT_DRAFTS = (
    ROOT
    / "records/private/v19_1/condition_completeness"
    / "development_query_drafts_machine.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/evaluation/v19_1/condition_completeness"
    / "development_assignments_machine.json"
)
ROLES = (
    "answerable_positive",
    "paraphrase_positive",
    "single_condition_hard_negative",
    "unanswerable_neighbor",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", type=Path, default=DEFAULT_FAMILIES)
    parser.add_argument("--drafts", type=Path, default=DEFAULT_DRAFTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_families = _read_jsonl(args.families)
    development = {
        str(row["family_id"]): row
        for row in all_families
        if row.get("split") == "development"
    }
    draft_payload = _read_json(args.drafts)
    if draft_payload.get("holdout_ocr_opened") is not False:
        raise ValueError("development drafting receipt must keep holdout OCR closed")
    drafts = {str(row["family_id"]): row for row in draft_payload.get("families", [])}
    if development.keys() != drafts.keys():
        raise ValueError(
            "development source families and query drafts must be identical"
        )

    router = RuleRouter.legacy()
    assignments: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    for family_id, family in development.items():
        draft = drafts[family_id]
        target_id = str(family["target"]["item_id"])
        neighbor_id = str(family["neighbor"]["item_id"])
        if draft["target_item_id"] != target_id:
            raise ValueError(f"target mismatch for {family_id}")
        if draft["neighbor_item_id"] != neighbor_id:
            raise ValueError(f"neighbor mismatch for {family_id}")
        for index, role in enumerate(ROLES, start=1):
            query = " ".join(str(draft["query_drafts"][role]).split())
            normalized = query.casefold()
            if not query or normalized in seen_queries:
                raise ValueError(f"empty or duplicate query in {family_id}: {role}")
            seen_queries.add(normalized)
            answerable = role in {"answerable_positive", "paraphrase_positive"}
            assignments.append(
                {
                    "query_id": f"{family_id}_q{index}",
                    "family_id": family_id,
                    "query": query,
                    "query_role": role,
                    "content_stratum": family["content_stratum"],
                    "gold_answerable": answerable,
                    "gold_relevant_item_ids": [target_id] if answerable else [],
                    "source_item_id": target_id,
                    "neighbor_item_id": neighbor_id,
                    "changed_condition_kind": draft["changed_condition_kind"],
                    "legacy_route": router.route(query).route,
                    "review_status": "machine_draft_pending_human_review",
                }
            )
    if len(assignments) != 48:
        raise AssertionError(f"expected 48 development queries, got {len(assignments)}")
    source_sha256 = hashlib.sha256(args.drafts.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "study_id": "v19-1-condition-completeness-e2e",
        "status": "machine_draft_development_only_pending_human_review",
        "split": "v19_1_machine_draft_development_only",
        "eligible_for_promotion": False,
        "holdout_ocr_opened": False,
        "holdout_queries_compiled": False,
        "source_query_sha256": source_sha256,
        "family_count": len(development),
        "query_count": len(assignments),
        "assignments": assignments,
    }
    write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "status",
                    "split",
                    "family_count",
                    "query_count",
                    "holdout_ocr_opened",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
