"""Validate and apply blinded model-review proposals with an audit trail."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.blind_query_study import save_relevance_review  # noqa: E402


VALID_PROPOSAL_DECISIONS = {
    "answerable",
    "no_answer",
    "excluded",
    "ambiguous",
}
FINAL_DECISIONS = {"answerable", "no_answer", "excluded"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_safe_bundle(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {row["query_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Safe review bundle contains duplicate query IDs.")
    return result


def validate_model_review_batch(
    path: Path,
    *,
    safe_bundle: dict[str, dict[str, Any]],
    expected_query_ids: set[str],
    expected_bundle_sha256: str,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("bundle_sha256") != expected_bundle_sha256:
        raise ValueError(f"{path.name}: safe-bundle hash mismatch")
    reviews = list(payload.get("reviews", []))
    query_ids = [str(row.get("query_id", "")) for row in reviews]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError(f"{path.name}: duplicate query IDs")
    if set(query_ids) != expected_query_ids:
        missing = sorted(expected_query_ids - set(query_ids))
        extra = sorted(set(query_ids) - expected_query_ids)
        raise ValueError(
            f"{path.name}: query coverage mismatch; missing={missing}, extra={extra}"
        )
    normalized: list[dict[str, Any]] = []
    for source in reviews:
        query_id = str(source["query_id"])
        decision = str(source.get("decision", ""))
        if decision not in VALID_PROPOSAL_DECISIONS:
            raise ValueError(f"{query_id}: invalid proposal decision")
        try:
            confidence = float(source.get("confidence"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{query_id}: invalid confidence") from error
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{query_id}: confidence must be between 0 and 1")
        allowed_ids = {
            row["item_id"] for row in safe_bundle[query_id]["candidates"]
        }
        relevant_ids = list(dict.fromkeys(source.get("relevant_item_ids", [])))
        uncertain_ids = list(dict.fromkeys(source.get("uncertain_item_ids", [])))
        unknown = sorted((set(relevant_ids) | set(uncertain_ids)) - allowed_ids)
        if unknown:
            raise ValueError(f"{query_id}: unknown candidate IDs {unknown}")
        if decision == "answerable" and not relevant_ids:
            raise ValueError(f"{query_id}: answerable proposal lacks relevant IDs")
        if decision in {"no_answer", "excluded"} and relevant_ids:
            raise ValueError(f"{query_id}: non-answerable proposal has relevant IDs")
        notes = " ".join(str(source.get("notes_zh", "")).split())
        if not notes:
            raise ValueError(f"{query_id}: review note is required")
        normalized.append(
            {
                "query_id": query_id,
                "decision": decision,
                "relevant_item_ids": relevant_ids,
                "confidence": confidence,
                "notes_zh": notes,
                "uncertain_item_ids": uncertain_ids,
                "reviewer": str(payload.get("reviewer", path.stem)),
            }
        )
    normalized.sort(key=lambda row: row["query_id"])
    return normalized


def partition_model_reviews(
    reviews: Iterable[dict[str, Any]],
    *,
    confidence_threshold: float = 0.90,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    auto: list[dict[str, Any]] = []
    needs_second_pass: list[dict[str, Any]] = []
    for row in reviews:
        is_auto = (
            row["decision"] in FINAL_DECISIONS
            and float(row["confidence"]) >= confidence_threshold
            and not row.get("uncertain_item_ids")
        )
        (auto if is_auto else needs_second_pass).append(row)
    return auto, needs_second_pass


def apply_model_reviews(
    study_dir: Path,
    reviews: Iterable[dict[str, Any]],
    *,
    known_item_ids: set[str],
    reviewer_type: str,
) -> int:
    count = 0
    for row in reviews:
        save_relevance_review(
            study_dir,
            {
                "query_id": row["query_id"],
                "decision": row["decision"],
                "relevant_item_ids": ";".join(row["relevant_item_ids"]),
                "evidence_scope": "candidate_pool",
                "reviewer_type": reviewer_type,
                "reviewer_id": row["reviewer"],
                "review_confidence": f"{float(row['confidence']):.3f}",
                "human_notes": row["notes_zh"],
            },
            known_item_ids=known_item_ids,
        )
        count += 1
    return count


def review_signature(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return (
        str(row["decision"]),
        tuple(sorted(str(item_id) for item_id in row["relevant_item_ids"])),
    )


def consensus_model_reviews(
    review_sets: list[list[dict[str, Any]]],
    *,
    minimum_confidence: float = 0.80,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Accept an exact clean majority; return all other queries unresolved."""
    if len(review_sets) < 2:
        raise ValueError("Consensus requires at least two independent review sets.")
    maps = [
        {row["query_id"]: row for row in review_set}
        for review_set in review_sets
    ]
    query_ids = set(maps[0])
    if any(set(mapping) != query_ids for mapping in maps[1:]):
        raise ValueError("Consensus review sets do not cover the same queries.")
    required_votes = len(review_sets) // 2 + 1
    accepted: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for query_id in sorted(query_ids):
        rows = [mapping[query_id] for mapping in maps]
        counts = Counter(review_signature(row) for row in rows)
        signature, vote_count = counts.most_common(1)[0]
        agreeing = [row for row in rows if review_signature(row) == signature]
        clean_agreeing = [
            row
            for row in agreeing
            if float(row["confidence"]) >= minimum_confidence
            and not row.get("uncertain_item_ids")
        ]
        if (
            vote_count >= required_votes
            and signature[0] in FINAL_DECISIONS
            and len(clean_agreeing) >= required_votes
        ):
            accepted.append(
                {
                    "query_id": query_id,
                    "decision": signature[0],
                    "relevant_item_ids": list(signature[1]),
                    "confidence": sum(
                        float(row["confidence"]) for row in clean_agreeing
                    )
                    / len(clean_agreeing),
                    "notes_zh": "；".join(
                        f"{row['reviewer']}：{row['notes_zh']}"
                        for row in clean_agreeing
                    ),
                    "uncertain_item_ids": [],
                    "reviewer": "+".join(
                        row["reviewer"] for row in clean_agreeing
                    ),
                }
            )
        else:
            unresolved.append(
                {
                    "query_id": query_id,
                    "votes": [
                        {
                            "reviewer": row["reviewer"],
                            "decision": row["decision"],
                            "relevant_item_ids": row["relevant_item_ids"],
                            "confidence": row["confidence"],
                            "notes_zh": row["notes_zh"],
                            "uncertain_item_ids": row.get(
                                "uncertain_item_ids", []
                            ),
                        }
                        for row in rows
                    ],
                }
            )
    return accepted, unresolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", nargs="+", type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--bundle-protocol", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    safe_bundle = read_safe_bundle(args.bundle)
    protocol = json.loads(args.bundle_protocol.read_text(encoding="utf-8"))
    combined: list[dict[str, Any]] = []
    for path in args.batch:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_ids = {
            str(row["query_id"]) for row in payload.get("reviews", [])
        }
        combined.extend(
            validate_model_review_batch(
                path,
                safe_bundle=safe_bundle,
                expected_query_ids=expected_ids,
                expected_bundle_sha256=protocol["bundle_sha256"],
            )
        )
    if len(combined) != len({row["query_id"] for row in combined}):
        raise ValueError("Combined batches contain duplicate query IDs.")
    auto, second_pass = partition_model_reviews(combined)
    summary = {
        "review_count": len(combined),
        "auto_count": len(auto),
        "second_pass_count": len(second_pass),
        "auto_query_ids": [row["query_id"] for row in auto],
        "second_pass_query_ids": [row["query_id"] for row in second_pass],
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
