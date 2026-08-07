"""High-recall pooling for corpus-answerability review."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from ocr_vlm_retrieval.evaluation.judgments import CORPUS_ANSWERABILITY_TASK

DEFAULT_HIGH_RECALL_RUN_DEPTHS: dict[str, int] = {
    "dense_text": 100,
    "bm25": 100,
    "global_visual": 100,
    "v16_quality_hybrid": 100,
    "v17_quality_hybrid": 100,
    "candidate_verifier": 50,
}
HIGH_RECALL_EXPANSION_SOURCES = (
    "same_source",
    "near_duplicate",
    "category_neighbor",
    "counterfactual",
)
REVIEW_ASSET_FIELDS = ("image_path", "thumbnail_path")


def _item_id(row: Mapping[str, Any]) -> str:
    return str(row.get("item_id", "")).strip()


def _review_asset(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: row[field]
        for field in REVIEW_ASSET_FIELDS
        if row.get(field) is not None
    }


def _pool_sha256(
    query_id: str,
    candidates: Sequence[Mapping[str, Any]],
    run_depths: Mapping[str, int],
) -> str:
    material = {
        "task_id": CORPUS_ANSWERABILITY_TASK,
        "query_id": query_id,
        "run_depths": dict(sorted(run_depths.items())),
        "candidates": [
            {
                "item_id": str(row["item_id"]),
                "retrieval_evidence": row["retrieval_evidence"],
                "expansion_sources": row["expansion_sources"],
            }
            for row in candidates
        ],
    }
    return hashlib.sha256(
        json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def build_high_recall_answerability_pool(
    query_id: str,
    ranked_runs: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    run_depths: Mapping[str, int] | None = None,
    expansions: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Build an auditable high-recall pool without assigning answerability."""

    normalized_query_id = str(query_id).strip()
    if not normalized_query_id:
        raise ValueError("query_id is required")
    depths = dict(run_depths or DEFAULT_HIGH_RECALL_RUN_DEPTHS)
    if not depths or any(depth < 1 for depth in depths.values()):
        raise ValueError("Every configured retrieval depth must be positive")
    missing_runs = sorted(set(depths) - set(ranked_runs))
    if missing_runs:
        raise ValueError("Missing high-recall runs: " + ", ".join(missing_runs))

    pooled: dict[str, dict[str, Any]] = {}

    def candidate(item_id: str, source: Mapping[str, Any]) -> dict[str, Any]:
        row = pooled.setdefault(
            item_id,
            {
                "item_id": item_id,
                "retrieval_evidence": {},
                "expansion_sources": [],
                "review_asset": {},
            },
        )
        if not row["review_asset"]:
            row["review_asset"] = _review_asset(source)
        return row

    retrieval_runs: list[dict[str, Any]] = []
    for run_id, depth in sorted(depths.items()):
        ranking = ranked_runs[run_id]
        seen: set[str] = set()
        returned = 0
        for rank, source in enumerate(ranking[:depth], start=1):
            item_id = _item_id(source)
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            returned += 1
            candidate(item_id, source)["retrieval_evidence"][run_id] = rank
        retrieval_runs.append(
            {
                "run_id": run_id,
                "configured_depth": depth,
                "returned_unique_count": returned,
            }
        )

    normalized_expansions = expansions or {}
    unknown_expansions = sorted(
        set(normalized_expansions) - set(HIGH_RECALL_EXPANSION_SOURCES)
    )
    if unknown_expansions:
        raise ValueError(
            "Unknown high-recall expansion sources: "
            + ", ".join(unknown_expansions)
        )
    for source_id in HIGH_RECALL_EXPANSION_SOURCES:
        for source in normalized_expansions.get(source_id, []):
            item_id = _item_id(source)
            if not item_id:
                continue
            row = candidate(item_id, source)
            if source_id not in row["expansion_sources"]:
                row["expansion_sources"].append(source_id)

    candidates = sorted(
        pooled.values(),
        key=lambda row: (
            min(row["retrieval_evidence"].values(), default=10**9),
            -len(row["retrieval_evidence"]),
            str(row["item_id"]),
        ),
    )
    for row in candidates:
        row["retrieval_evidence"] = dict(
            sorted(row["retrieval_evidence"].items())
        )
        row["expansion_sources"] = sorted(row["expansion_sources"])
    pool_sha256 = _pool_sha256(normalized_query_id, candidates, depths)
    return {
        "task_id": CORPUS_ANSWERABILITY_TASK,
        "query_id": normalized_query_id,
        "pool_sha256": pool_sha256,
        "retrieval_runs": retrieval_runs,
        "expansion_sources": [
            source_id
            for source_id in HIGH_RECALL_EXPANSION_SOURCES
            if normalized_expansions.get(source_id)
        ],
        "candidate_count": len(candidates),
        "candidates": candidates,
        "review_requirement": {
            "independent_reviewer_count": 2,
            "adjudicate_conflicts": True,
        },
        "answerability_status": "pending_independent_human_review",
    }


def blind_corpus_answerability_packet(
    audit_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove item IDs, retrieval evidence, and source metadata for reviewers."""

    if audit_row.get("task_id") != CORPUS_ANSWERABILITY_TASK:
        raise ValueError("Expected a corpus-answerability audit row")
    candidates_source = audit_row.get("candidates", [])
    if not isinstance(candidates_source, Sequence):
        raise ValueError("candidates must be a sequence")
    candidates = [
        {
            "candidate_id": f"C{index:04d}",
            "review_asset": dict(source.get("review_asset", {})),
        }
        for index, source in enumerate(candidates_source, start=1)
    ]
    return {
        "task_id": CORPUS_ANSWERABILITY_TASK,
        "query_id": str(audit_row["query_id"]),
        "pool_sha256": str(audit_row["pool_sha256"]),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
