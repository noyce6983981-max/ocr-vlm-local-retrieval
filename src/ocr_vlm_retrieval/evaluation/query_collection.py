"""Leakage-aware query proposal and freeze utilities for the V17 study."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ocr_vlm_retrieval.gating.attribute_coverage import (
    AttributePlan,
    decompose_visual_query,
)

VALID_REVIEW_ACTIONS = {"approve", "rewrite", "reject"}
COLOR_SUBSTITUTIONS = {
    "red": "blue",
    "blue": "red",
    "green": "yellow",
    "yellow": "purple",
    "white": "black",
    "black": "white",
    "purple": "orange",
    "orange": "green",
    "pink": "brown",
    "brown": "pink",
    "gray": "green",
    "grey": "green",
    "golden": "blue",
}
RELATION_SUBSTITUTIONS = (
    ("to the left of", "to the right of"),
    ("to the right of", "to the left of"),
    ("on the left", "on the right"),
    ("on the right", "on the left"),
    ("in front of", "behind"),
    ("on the back of", "on the front of"),
    ("on the front of", "on the back of"),
    ("behind", "in front of"),
    ("above", "below"),
    ("below", "above"),
    ("under", "over"),
    ("over", "under"),
    ("beside", "between"),
    ("next to", "behind"),
    ("inside", "outside"),
    ("on", "under"),
)


def normalize_query(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def query_fingerprint(rows: Iterable[Mapping[str, Any]]) -> str:
    canonical = [
        {
            "query_id": str(row["query_id"]),
            "query": normalize_query(str(row["query"])),
            "split": str(row["split"]),
            "group_id": str(row["group_id"]),
        }
        for row in sorted(rows, key=lambda source: str(source["query_id"]))
    ]
    return hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def extract_textvqa_question(row: Mapping[str, Any]) -> str:
    for tag in row.get("source_tags", []) or []:
        value = str(tag)
        if value.startswith("question="):
            return " ".join(value.split("=", 1)[1].split())
    return ""


def _study_group(row: Mapping[str, Any]) -> str:
    hard_group = str(row.get("hard_negative_group", "")).strip()
    if hard_group:
        return f"source-neighbor:{hard_group}"
    source_group = str(row.get("source_group_id", "")).strip()
    if source_group:
        return f"source:{source_group}"
    return f"item:{row['item_id']}"


def _stable_key(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode()).hexdigest()


def _replace_word(text: str, source: str, target: str) -> str | None:
    match = re.search(
        rf"(?<![a-z0-9]){re.escape(source)}(?![a-z0-9])",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return text[: match.start()] + target + text[match.end() :]


def near_neighbor_transform(question: str) -> dict[str, str]:
    """Change one necessary condition without inventing an unrelated query."""

    for source, target in COLOR_SUBSTITUTIONS.items():
        transformed = _replace_word(question, source, target)
        if transformed:
            return {
                "query": transformed,
                "transformation_type": "color_substitution",
                "source_value": source,
                "replacement_value": target,
            }
    for source, target in RELATION_SUBSTITUTIONS:
        transformed = _replace_word(question, source, target)
        if transformed:
            return {
                "query": transformed,
                "transformation_type": "relation_substitution",
                "source_value": source,
                "replacement_value": target,
            }
    raise ValueError(f"No conservative near-neighbor transform for: {question}")


def _candidate_rows(
    manifest: Iterable[Mapping[str, Any]],
    policy: Mapping[str, Any],
    excluded_queries: set[str],
) -> list[dict[str, Any]]:
    by_query: dict[str, dict[str, Any]] = {}
    for source in manifest:
        if "TextVQA" not in str(source.get("public_source_name", "")):
            continue
        question = extract_textvqa_question(source)
        normalized = normalize_query(question)
        if not question or normalized in excluded_queries:
            continue
        plan = decompose_visual_query(question, policy)
        if not plan.compositional:
            continue
        candidate = {
            "query": question,
            "source_item_id": str(source["item_id"]),
            "group_id": _study_group(source),
            "source_question": question,
            "source_dataset": str(source.get("public_source_name", "")),
            "source_license": str(source.get("license", "")),
            "source_file": str(source.get("public_source_file", "")),
            "attribute_plan": plan.to_dict(),
        }
        existing = by_query.get(normalized)
        if existing is None or _stable_key(
            str(candidate["source_item_id"])
        ) < _stable_key(
            str(existing["source_item_id"])
        ):
            by_query[normalized] = candidate
    return sorted(
        by_query.values(),
        key=lambda row: _stable_key(str(row["group_id"]), str(row["query"])),
    )


def _assign_source_splits(rows: Sequence[dict[str, Any]]) -> dict[str, str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)
    target = len(rows) // 2
    calibration_count = 0
    assignments: dict[str, str] = {}
    for group_id, group_rows in sorted(
        grouped.items(), key=lambda pair: _stable_key(pair[0])
    ):
        group_size = len(group_rows)
        if calibration_count + group_size <= target:
            split = "calibration"
            calibration_count += group_size
        else:
            split = "holdout"
        assignments[group_id] = split
    return assignments


def build_textvqa_compositional_proposals(
    manifest: Iterable[Mapping[str, Any]],
    policy: Mapping[str, Any],
    *,
    excluded_queries: Iterable[str] = (),
    answerable_seed_count: int = 56,
    total_count: int = 80,
) -> list[dict[str, Any]]:
    """Build source-question seeds plus unverified hard-negative proposals."""

    if answerable_seed_count < 1 or total_count <= answerable_seed_count:
        raise ValueError("proposal counts must include seeds and hard negatives")
    excluded = {normalize_query(value) for value in excluded_queries if value}
    candidates = _candidate_rows(manifest, policy, excluded)
    if len(candidates) < answerable_seed_count:
        raise ValueError(
            f"Need {answerable_seed_count} compositional source questions; "
            f"found {len(candidates)}"
        )
    selected = candidates[:answerable_seed_count]
    split_by_group = _assign_source_splits(selected)
    source_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    proposals: list[dict[str, Any]] = []
    for row in selected:
        split = split_by_group[str(row["group_id"])]
        source_by_split[split].append(row)
        proposals.append(
            {
                **row,
                "split": split,
                "route": "compositional_visual",
                "query_family": "+".join(
                    sorted(
                        {
                            requirement["kind"]
                            for requirement in row["attribute_plan"]["requirements"]
                        }
                    )
                ),
                "query_origin": "human_authored_textvqa_source_question",
                "expected_answerability": "answerable_seed_requires_pool_review",
                "transformation_type": "none",
                "query_review_status": "pending_human_review",
            }
        )

    target_by_split = {"calibration": total_count // 2, "holdout": total_count // 2}
    for split in ("calibration", "holdout"):
        needed = target_by_split[split] - len(source_by_split[split])
        transformable: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for row in source_by_split[split]:
            try:
                transform = near_neighbor_transform(str(row["query"]))
            except ValueError:
                continue
            transformed_plan: AttributePlan = decompose_visual_query(
                transform["query"], policy
            )
            if transformed_plan.compositional:
                transformable.append(
                    (
                        row,
                        {**transform, "attribute_plan": transformed_plan.to_dict()},
                    )
                )
        transformable.sort(
            key=lambda pair: _stable_key(
                str(pair[0]["group_id"]), str(pair[0]["query"]), "negative"
            )
        )
        if len(transformable) < needed:
            raise ValueError(
                f"Split {split} needs {needed} hard negatives; "
                f"only {len(transformable)} are transformable"
            )
        for source, transform in transformable[:needed]:
            proposals.append(
                {
                    **source,
                    "query": transform["query"],
                    "attribute_plan": transform["attribute_plan"],
                    "split": split,
                    "route": "compositional_visual",
                    "query_family": "near_neighbor_"
                    + str(transform["transformation_type"]),
                    "query_origin": "near_neighbor_transformation_proposal",
                    "expected_answerability": "no_answer_probe_unverified",
                    "transformation_type": transform["transformation_type"],
                    "transformation_source_value": transform["source_value"],
                    "transformation_replacement_value": transform[
                        "replacement_value"
                    ],
                    "query_review_status": "pending_human_review",
                }
            )

    proposals.sort(
        key=lambda row: (
            0 if row["split"] == "calibration" else 1,
            _stable_key(str(row["group_id"]), str(row["query"])),
        )
    )
    for index, row in enumerate(proposals, start=1):
        row["query_id"] = f"v17_visual_{index:03d}"
    validate_query_rows(proposals, expected_count=total_count)
    return proposals


def validate_query_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_count: int
) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} rows, got {len(rows)}")
    query_ids = [str(row.get("query_id", "")) for row in rows]
    if any(not value for value in query_ids) or len(query_ids) != len(set(query_ids)):
        raise ValueError("query_id values must be present and unique")
    normalized = [normalize_query(str(row.get("query", ""))) for row in rows]
    empty_query = any(not value for value in normalized)
    duplicate_query = len(normalized) != len(set(normalized))
    if empty_query or duplicate_query:
        raise ValueError("query values must be present and unique")
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = str(row.get("split", ""))
        if split not in {"calibration", "holdout"}:
            raise ValueError(f"Invalid split: {split!r}")
        splits_by_group[str(row["group_id"])].add(split)
    leaked = sorted(
        group
        for group, splits in splits_by_group.items()
        if len(splits) > 1
    )
    if leaked:
        raise ValueError("Groups cross splits: " + ", ".join(leaked))


def proposal_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "query_count": len(rows),
        "query_set_sha256": query_fingerprint(rows),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "origin_counts": dict(
            sorted(Counter(row["query_origin"] for row in rows).items())
        ),
        "transformation_counts": dict(
            sorted(Counter(row["transformation_type"] for row in rows).items())
        ),
        "group_count": len({row["group_id"] for row in rows}),
        "status": "proposals_only_not_human_ground_truth",
    }


def freeze_reviewed_queries(
    proposals: Sequence[Mapping[str, Any]],
    reviews: Iterable[Mapping[str, Any]],
    policy: Mapping[str, Any],
    *,
    expected_count: int,
    excluded_queries: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Freeze only complete, explicitly human-approved query reviews."""

    review_by_id: dict[str, Mapping[str, Any]] = {}
    for review in reviews:
        query_id = str(review.get("query_id", "")).strip()
        if not query_id or query_id in review_by_id:
            raise ValueError("Reviews need unique query_id values")
        action = str(review.get("review_action", "")).strip()
        reviewer_id = str(review.get("reviewer_id", "")).strip()
        if action not in VALID_REVIEW_ACTIONS or not reviewer_id:
            raise ValueError(f"Incomplete human query review: {query_id}")
        review_by_id[query_id] = review
    proposal_ids = {str(row["query_id"]) for row in proposals}
    if set(review_by_id) != proposal_ids:
        missing = sorted(proposal_ids - set(review_by_id))
        unknown = sorted(set(review_by_id) - proposal_ids)
        raise ValueError(
            f"Review coverage mismatch; missing={missing}, unknown={unknown}"
        )

    excluded = {normalize_query(value) for value in excluded_queries if value}
    frozen: list[dict[str, Any]] = []
    for source in proposals:
        review = review_by_id[str(source["query_id"])]
        action = str(review["review_action"])
        if action == "reject":
            continue
        query = str(source["query"])
        if action == "rewrite":
            query = " ".join(str(review.get("reviewed_query", "")).split())
            if not query:
                raise ValueError(f"Rewrite is empty: {source['query_id']}")
        if normalize_query(query) in excluded:
            raise ValueError(f"Frozen query overlaps an excluded set: {query}")
        plan = decompose_visual_query(query, policy)
        if not plan.compositional:
            raise ValueError(f"Reviewed query is no longer compositional: {query}")
        frozen.append(
            {
                "query_id": source["query_id"],
                "query": query,
                "split": source["split"],
                "group_id": source["group_id"],
                "route": source["route"],
                "query_family": source["query_family"],
                "query_origin": source["query_origin"],
                "transformation_type": source.get("transformation_type", "none"),
                "review_status": "human_query_approved",
                "query_reviewer_id": str(review["reviewer_id"]),
                "attribute_plan": plan.to_dict(),
            }
        )
    validate_query_rows(frozen, expected_count=expected_count)
    return frozen
