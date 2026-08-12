from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.v19_selective_intervention_review_app import (
    ROLES,
    family_snapshot_sha256,
    load_families,
    load_reviews,
)

DEFAULT_PRIVATE_DIR = PROJECT_ROOT / "records/private/v19/selective_intervention"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def compile_review(
    *,
    families: list[dict[str, Any]],
    reviews: Mapping[str, Mapping[str, Any]],
    completion: Mapping[str, Any],
    reviewer_id: str,
) -> list[dict[str, Any]]:
    family_ids = {str(family["family_id"]) for family in families}
    if str(completion.get("reviewer_id", "")) != reviewer_id:
        raise ValueError("completion reviewer does not match")
    if set(completion.get("seen_family_ids", [])) != family_ids:
        raise ValueError("completion receipt does not cover every family")
    snapshot_material = "\n".join(
        family_snapshot_sha256(family) for family in families
    )
    expected_bundle = hashlib.sha256(snapshot_material.encode("utf-8")).hexdigest()
    if str(completion.get("family_bundle_sha256", "")) != expected_bundle:
        raise ValueError("family bundle changed after completion")
    invalid_review_ids = set(reviews) - family_ids
    if invalid_review_ids:
        raise ValueError(f"review contains unknown families: {invalid_review_ids}")
    replacements = sorted(
        family_id
        for family_id, review in reviews.items()
        if str(review.get("pair_decision", "")) == "replace_pair"
    )
    if replacements:
        raise ValueError(
            "source pairs require replacement before compilation: "
            + ", ".join(replacements)
        )

    rows: list[dict[str, Any]] = []
    for family in families:
        family_id = str(family["family_id"])
        review = reviews.get(family_id)
        if review and str(review.get("family_snapshot_sha256", "")) != (
            family_snapshot_sha256(family)
        ):
            raise ValueError(f"family changed after review: {family_id}")
        query_texts = (
            review["query_texts"] if review else family["query_drafts"]
        )
        condition = (
            str(review["changed_condition_kind"])
            if review
            else str(family["changed_condition_kind"])
        )
        target_id = str(family["target"]["item_id"])
        neighbor_id = str(family["neighbor"]["item_id"])
        for index, role in enumerate(ROLES, start=1):
            answerable = role in {"answerable_positive", "paraphrase_positive"}
            rows.append(
                {
                    "query_id": f"{family_id}_q{index}",
                    "family_id": family_id,
                    "split": family["split"],
                    "content_stratum": family["content_stratum"],
                    "query_role": role,
                    "query_text": str(query_texts[role]).strip(),
                    "gold_answerable": answerable,
                    "gold_relevant_item_ids": [target_id] if answerable else [],
                    "target_item_id": target_id,
                    "neighbor_item_id": neighbor_id,
                    "changed_condition_kind": condition,
                    "query_review_status": (
                        "human_corrected" if review else "human_accepted_unchanged"
                    ),
                    "query_reviewer_id": reviewer_id,
                    "status": "reviewed_not_frozen",
                }
            )
    texts = [str(row["query_text"]) for row in rows]
    expected_query_count = len(families) * len(ROLES)
    if len(rows) != expected_query_count or len(texts) != len(set(texts)):
        raise ValueError(
            "reviewed query set must contain four unique queries per family"
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile reviewed V19 drafts without freezing or running evaluation."
    )
    parser.add_argument("--reviewer-id", default="reviewer_01")
    parser.add_argument("--private-dir", type=Path, default=DEFAULT_PRIVATE_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    private_dir = args.private_dir.resolve()
    review_path = private_dir / f"{args.reviewer_id}_family_corrections.jsonl"
    completion_path = private_dir / f"{args.reviewer_id}_completion.json"
    if not completion_path.is_file():
        raise SystemExit("Human-review completion receipt is missing")
    families = load_families()
    reviews = load_reviews(review_path)
    completion = json.loads(completion_path.read_text(encoding="utf-8-sig"))
    rows = compile_review(
        families=families,
        reviews=reviews,
        completion=completion,
        reviewer_id=args.reviewer_id,
    )
    if len(families) != 50 or len(rows) != 200:
        raise SystemExit("Production review must contain 50 families / 200 queries")
    query_path = private_dir / "reviewed_queries.jsonl"
    receipt_path = private_dir / "review_compile_receipt.json"
    _atomic_jsonl(query_path, rows)
    _atomic_json(
        receipt_path,
        {
            "schema_version": 1,
            "study_id": "v19-selective-intervention-e2e",
            "status": "reviewed_not_frozen",
            "reviewer_id": args.reviewer_id,
            "family_count": len(families),
            "query_count": len(rows),
            "corrected_family_count": len(reviews),
            "reviewed_queries_sha256": _sha256(query_path),
            "completion_sha256": _sha256(completion_path),
            "eligible_for_final_evaluation": False,
            "next_required_steps": [
                "replace any rejected source pairs",
                "run development-only retrieval audit",
                "lock candidate parameters",
                "authorize exactly one final holdout run",
            ],
        },
    )
    print(f"Compiled {len(rows)} reviewed queries to {query_path}")
    print("No freeze receipt or evaluation authorization was created")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
