"""Compile the fully human-reviewed V19.1 development query set."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_vlm_retrieval.routing.rule_router import RuleRouter  # noqa: E402
from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402

PRIVATE_DIR = ROOT / "records/private/v19_1/condition_completeness"
DEFAULT_FAMILIES = PRIVATE_DIR / "source_families_draft.jsonl"
DEFAULT_DRAFTS = PRIVATE_DIR / "development_query_drafts_machine.json"
DEFAULT_REVIEWS = PRIVATE_DIR / "reviewer_01_development_query_reviews.jsonl"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/evaluation/v19_1/condition_completeness"
    / "development_assignments_human_reviewed.json"
)
ROLES = (
    "answerable_positive",
    "paraphrase_positive",
    "single_condition_hard_negative",
    "unanswerable_neighbor",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", type=Path, default=DEFAULT_FAMILIES)
    parser.add_argument("--drafts", type=Path, default=DEFAULT_DRAFTS)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


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


def _snapshot_sha256(family: Mapping[str, Any]) -> str:
    material = json.dumps(
        dict(family), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def main() -> int:
    args = parse_args()
    source_rows = [
        row for row in _read_jsonl(args.families) if row.get("split") == "development"
    ]
    draft_payload = _read_json(args.drafts)
    if draft_payload.get("holdout_ocr_opened") is not False:
        raise ValueError("holdout OCR must remain closed during development compilation")
    drafts = {str(row["family_id"]): row for row in draft_payload.get("families", [])}
    sources = {str(row["family_id"]): row for row in source_rows}
    reviews = {str(row["family_id"]): row for row in _read_jsonl(args.reviews)}
    if len(source_rows) != 12 or len(sources) != 12:
        raise ValueError("expected 12 unique development source families")
    if sources.keys() != drafts.keys() or sources.keys() != reviews.keys():
        raise ValueError("all 12 source, draft, and review family IDs must match")

    router = RuleRouter.legacy()
    assignments: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    reviewer_ids: set[str] = set()
    for family_id in sorted(sources):
        family = {**sources[family_id], **drafts[family_id]}
        review = reviews[family_id]
        if review.get("decision") != "human_approved_development_family":
            raise ValueError(f"family not approved: {family_id}")
        if review.get("family_snapshot_sha256") != _snapshot_sha256(family):
            raise ValueError(f"stale family review: {family_id}")
        reviewer_id = str(review.get("reviewer_id", "")).strip()
        if not reviewer_id:
            raise ValueError(f"reviewer ID missing: {family_id}")
        reviewer_ids.add(reviewer_id)
        changed_condition_kind = str(
            review.get("changed_condition_kind", "")
        ).strip()
        if not changed_condition_kind:
            raise ValueError(f"changed condition missing: {family_id}")
        query_texts = review.get("query_texts")
        if not isinstance(query_texts, dict) or set(query_texts) != set(ROLES):
            raise ValueError(f"four reviewed query roles required: {family_id}")
        target_id = str(family["target"]["item_id"])
        neighbor_id = str(family["neighbor"]["item_id"])
        for index, role in enumerate(ROLES, start=1):
            query = " ".join(str(query_texts[role]).split())
            normalized = query.casefold()
            if len(query) < 8 or normalized in seen_queries:
                raise ValueError(f"short or duplicate reviewed query: {family_id}/{role}")
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
                    "changed_condition_kind": changed_condition_kind,
                    "legacy_route": router.route(query).route,
                    "review_status": "human_approved_development_family",
                    "reviewer_id": reviewer_id,
                }
            )
    if len(assignments) != 48:
        raise AssertionError(f"expected 48 reviewed queries, got {len(assignments)}")

    source_review_sha256 = hashlib.sha256(args.reviews.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "study_id": "v19-1-condition-completeness-e2e",
        "status": "human_reviewed_development_only",
        "split": "v19_1_human_reviewed_development_only",
        "eligible_for_promotion": False,
        "holdout_ocr_opened": False,
        "holdout_queries_compiled": False,
        "source_review_sha256": source_review_sha256,
        "reviewer_ids": sorted(reviewer_ids),
        "family_count": len(sources),
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
                    "reviewer_ids",
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
