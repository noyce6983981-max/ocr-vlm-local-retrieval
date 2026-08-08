"""Reusable blinded-review packet and coverage validation helpers."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

FORBIDDEN_BLIND_FIELDS = {
    "dataset",
    "expected_answer",
    "rank",
    "retrieval_method",
    "score",
    "source_filename",
    "source_path",
    "system_decision",
}


def build_blind_review_packet(
    *,
    query_id: str,
    query: str,
    candidates: Sequence[Mapping[str, Any]],
    study_fingerprint: str,
    split: str,
) -> dict[str, Any]:
    """Copy only reviewer-safe candidate assets into a relevance packet."""

    safe_candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, candidate in enumerate(candidates, start=1):
        item_id = str(candidate.get("item_id", "")).strip()
        image_path = str(candidate.get("image_path", "")).strip()
        if not item_id or not image_path or item_id in seen:
            raise ValueError("candidates need unique item IDs and image paths")
        leaked = FORBIDDEN_BLIND_FIELDS & set(candidate)
        if leaked:
            raise ValueError(
                "candidate contains blind fields: " + ", ".join(sorted(leaked))
            )
        seen.add(item_id)
        safe = {
            "candidate_id": f"C{index:02d}",
            "item_id": item_id,
            "review_asset": {"image_path": image_path},
        }
        ocr_evidence = str(candidate.get("ocr_evidence", "")).strip()
        if ocr_evidence:
            safe["system_ocr_evidence"] = ocr_evidence
        safe_candidates.append(safe)
    packet = {
        "query_id": query_id,
        "query": query,
        "split": split,
        "study_fingerprint": study_fingerprint,
        "candidates": safe_candidates,
    }
    packet["pool_sha256"] = review_pool_fingerprint(packet)
    return packet


def review_pool_fingerprint(packet: Mapping[str, Any]) -> str:
    material = {
        "query_id": str(packet["query_id"]),
        "query": str(packet["query"]),
        "split": str(packet["split"]),
        "study_fingerprint": str(packet["study_fingerprint"]),
        "item_ids": [str(candidate["item_id"]) for candidate in packet["candidates"]],
    }
    return hashlib.sha256(
        json.dumps(
            material, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def validate_independent_candidate_reviews(
    packets: Sequence[Mapping[str, Any]],
    judgments: Iterable[Mapping[str, Any]],
    *,
    required_reviewers: int = 2,
) -> dict[str, tuple[str, ...]]:
    """Require every independent reviewer to label every pooled candidate."""

    if required_reviewers < 2:
        raise ValueError("required_reviewers must be at least two")
    expected: dict[str, set[str]] = {}
    for packet in packets:
        query_id = str(packet.get("query_id", "")).strip()
        if not query_id or query_id in expected:
            raise ValueError("packets need unique query IDs")
        expected[query_id] = {
            str(candidate["item_id"]) for candidate in packet["candidates"]
        }
    reviewers_by_query: dict[str, set[str]] = defaultdict(set)
    seen_pairs: set[tuple[str, str]] = set()
    for judgment in judgments:
        query_id = str(judgment.get("query_id", "")).strip()
        reviewer_id = str(judgment.get("reviewer_id", "")).strip()
        pair = (query_id, reviewer_id)
        if query_id not in expected or not reviewer_id or pair in seen_pairs:
            raise ValueError("judgments need unique known query/reviewer pairs")
        relevance = judgment.get("candidate_relevance")
        if not isinstance(relevance, Mapping):
            raise ValueError(
                f"candidate_relevance is missing for {query_id}/{reviewer_id}"
            )
        actual = {str(item_id) for item_id in relevance}
        if actual != expected[query_id]:
            missing = sorted(expected[query_id] - actual)
            unknown = sorted(actual - expected[query_id])
            raise ValueError(
                f"candidate coverage mismatch for {query_id}/{reviewer_id}; "
                f"missing={missing}, unknown={unknown}"
            )
        if any(not isinstance(value, bool) for value in relevance.values()):
            raise ValueError("candidate relevance values must be booleans")
        seen_pairs.add(pair)
        reviewers_by_query[query_id].add(reviewer_id)
    for query_id in expected:
        count = len(reviewers_by_query[query_id])
        if count != required_reviewers:
            raise ValueError(
                f"query {query_id} needs {required_reviewers} reviewers; found {count}"
            )
    return {
        query_id: tuple(sorted(reviewers))
        for query_id, reviewers in sorted(reviewers_by_query.items())
    }
