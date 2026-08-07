"""Human-evaluation utilities for the V17 retrieval study.

The audit pool retains method provenance, while the reviewer packet deliberately
contains neither method names nor ranks. Judgments are stored once per
``(query_id, reviewer_id)`` so two reviewers never overwrite one another.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ocr_vlm_retrieval.evaluation.judgments import normalize_pool_judgment

REVIEW_METADATA_FIELDS = (
    "title",
    "source_name",
    "source_relpath",
    "page_number",
    "thumbnail_path",
    "image_path",
    "ocr_text_preview",
)
REVIEW_ASSET_FIELDS = ("thumbnail_path", "image_path")


def _item_id(row: Mapping[str, Any]) -> str:
    return str(row.get("item_id", "")).strip()


def pool_ranked_runs(
    ranked_runs: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    top_per_run: int = 20,
    rrf_constant: float = 60.0,
) -> list[dict[str, Any]]:
    """Return the deduplicated Top-k union from independently versioned runs.

    Unlike the legacy product pool, this study pool intentionally permits V16
    and V17 outputs to coexist. Method/version provenance remains only in the
    audit copy and must not be shown to reviewers.
    """

    if not ranked_runs:
        raise ValueError("At least one ranked run is required")
    if top_per_run < 1:
        raise ValueError("top_per_run must be positive")
    if rrf_constant <= 0:
        raise ValueError("rrf_constant must be positive")

    pooled: dict[str, dict[str, Any]] = {}
    for run_id, ranking in ranked_runs.items():
        normalized_run_id = str(run_id).strip()
        if not normalized_run_id:
            raise ValueError("run IDs must not be empty")
        seen_in_run: set[str] = set()
        for rank, source in enumerate(ranking[:top_per_run], start=1):
            item_id = _item_id(source)
            if not item_id or item_id in seen_in_run:
                continue
            seen_in_run.add(item_id)
            row = pooled.setdefault(
                item_id,
                {
                    "item_id": item_id,
                    "pool_score": 0.0,
                    "best_rank": rank,
                    "run_ranks": {},
                    "run_scores": {},
                    "review_metadata": {},
                },
            )
            row["pool_score"] += 1.0 / (rrf_constant + rank)
            row["best_rank"] = min(int(row["best_rank"]), rank)
            row["run_ranks"][normalized_run_id] = rank
            if source.get("score") is not None:
                row["run_scores"][normalized_run_id] = round(
                    float(source["score"]), 8
                )
            for field in REVIEW_METADATA_FIELDS:
                missing_metadata = field not in row["review_metadata"]
                if missing_metadata and source.get(field) is not None:
                    row["review_metadata"][field] = source[field]

    ordered = sorted(
        pooled.values(),
        key=lambda row: (
            -float(row["pool_score"]),
            int(row["best_rank"]),
            str(row["item_id"]),
        ),
    )
    for row in ordered:
        row["pool_score"] = round(float(row["pool_score"]), 10)
        row["run_ranks"] = dict(sorted(row["run_ranks"].items()))
        row["run_scores"] = dict(sorted(row["run_scores"].items()))
    return ordered


def blind_candidate_pool(
    query_id: str,
    pooled_rows: Sequence[Mapping[str, Any]],
    *,
    study_fingerprint: str,
) -> list[dict[str, Any]]:
    """Create a deterministic reviewer order with all run evidence removed."""

    query_id = str(query_id).strip()
    study_fingerprint = str(study_fingerprint).strip()
    if not query_id or not study_fingerprint:
        raise ValueError("query_id and study_fingerprint are required")
    item_ids = [_item_id(row) for row in pooled_rows]
    if any(not item_id for item_id in item_ids):
        raise ValueError("Every pooled row must contain an item_id")
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Pooled rows must contain unique item IDs")

    ordered = sorted(
        item_ids,
        key=lambda item_id: hashlib.sha256(
            f"{study_fingerprint}\0{query_id}\0{item_id}".encode()
        ).hexdigest(),
    )
    asset_by_id = {
        _item_id(row): {
            field: row.get("review_metadata", {}).get(field)
            for field in REVIEW_ASSET_FIELDS
            if row.get("review_metadata", {}).get(field) is not None
        }
        for row in pooled_rows
    }
    return [
        {
            "candidate_id": f"C{index:03d}",
            "item_id": item_id,
            "review_asset": asset_by_id[item_id],
        }
        for index, item_id in enumerate(ordered, start=1)
    ]


def _normalize_judgments(
    judgments: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    by_query: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for source in judgments:
        query_id = str(source.get("query_id", "")).strip()
        reviewer_id = str(source.get("reviewer_id", "")).strip()
        if not query_id or not reviewer_id:
            raise ValueError("Every judgment needs query_id and reviewer_id")
        if reviewer_id in by_query[query_id]:
            raise ValueError(
                f"Duplicate judgment for query={query_id}, reviewer={reviewer_id}"
            )
        normalized = normalize_pool_judgment(source)
        by_query[query_id][reviewer_id] = {
            "pool_relevance": normalized["pool_relevance"],
            "candidate_relevance": normalized["candidate_relevance"],
        }
    return by_query


def validate_annotation_coverage(
    queries: Iterable[Mapping[str, Any]],
    judgments: Iterable[Mapping[str, Any]],
    *,
    calibration_double_fraction: float = 0.30,
    candidate_ids_by_query: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Enforce full holdout double review and calibration overlap."""

    if not 0.0 <= calibration_double_fraction <= 1.0:
        raise ValueError("calibration_double_fraction must be in [0, 1]")
    query_rows = list(queries)
    query_ids = [str(row.get("query_id", "")).strip() for row in query_rows]
    if any(not query_id for query_id in query_ids):
        raise ValueError("Every query needs a query_id")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query_id values must be unique")
    normalized = _normalize_judgments(judgments)
    unknown = sorted(set(normalized) - set(query_ids))
    if unknown:
        raise ValueError("Judgments contain unknown query IDs: " + ", ".join(unknown))

    counts_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    failures: list[str] = []
    calibration_total = 0
    calibration_double = 0
    for row in query_rows:
        query_id = str(row["query_id"]).strip()
        split = str(row.get("split", "")).strip()
        if split not in {"calibration", "holdout"}:
            raise ValueError(f"Invalid split for {query_id}: {split!r}")
        reviewer_count = len(normalized.get(query_id, {}))
        counts_by_split[split][str(reviewer_count)] += 1
        if split == "holdout" and reviewer_count < 2:
            failures.append(f"{query_id}: holdout needs two reviewers")
        if split == "calibration":
            calibration_total += 1
            calibration_double += int(reviewer_count >= 2)
            if reviewer_count < 1:
                failures.append(f"{query_id}: calibration needs one reviewer")
        if candidate_ids_by_query is not None:
            expected = set(candidate_ids_by_query.get(query_id, ()))
            if not expected:
                failures.append(f"{query_id}: candidate pool is missing")
            for reviewer_id, judgment in normalized.get(query_id, {}).items():
                actual = set(judgment["candidate_relevance"])
                if actual != expected:
                    missing = sorted(expected - actual)
                    extra = sorted(actual - expected)
                    failures.append(
                        f"{query_id}/{reviewer_id}: incomplete candidate coverage; "
                        f"missing={missing}, extra={extra}"
                    )
    required_double = math.ceil(calibration_total * calibration_double_fraction)
    if calibration_double < required_double:
        failures.append(
            "calibration double-review coverage is "
            f"{calibration_double}/{calibration_total}; need at least {required_double}"
        )
    return {
        "valid": not failures,
        "failures": failures,
        "query_count": len(query_rows),
        "counts_by_split_and_reviewer_count": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(counts_by_split.items())
        },
        "calibration_double_review_fraction": (
            calibration_double / calibration_total if calibration_total else 0.0
        ),
    }


def _cohen_kappa_binary(pairs: Sequence[tuple[bool, bool]]) -> float | None:
    if not pairs:
        return None
    observed = sum(left == right for left, right in pairs) / len(pairs)
    left_positive = sum(left for left, _ in pairs) / len(pairs)
    right_positive = sum(right for _, right in pairs) / len(pairs)
    expected = (
        left_positive * right_positive
        + (1.0 - left_positive) * (1.0 - right_positive)
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1.0 - expected)


def _cohen_kappa_multiclass(
    pairs: Sequence[tuple[str, str]],
) -> float | None:
    if not pairs:
        return None
    observed = sum(left == right for left, right in pairs) / len(pairs)
    categories = {value for pair in pairs for value in pair}
    expected = sum(
        (sum(left == category for left, _ in pairs) / len(pairs))
        * (sum(right == category for _, right in pairs) / len(pairs))
        for category in categories
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1.0 - expected)


def agreement_report(
    judgments: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report agreement over queries with exactly two independent reviewers."""

    normalized = _normalize_judgments(judgments)
    pool_relevance_pairs: list[tuple[str, str]] = []
    relevance_pairs: list[tuple[bool, bool]] = []
    conflict_queries: list[str] = []
    conflict_types: Counter[str] = Counter()
    double_reviewed = 0
    for query_id, by_reviewer in sorted(normalized.items()):
        if len(by_reviewer) != 2:
            continue
        double_reviewed += 1
        first, second = [by_reviewer[key] for key in sorted(by_reviewer)]
        pool_relevance_pair = (
            str(first["pool_relevance"]),
            str(second["pool_relevance"]),
        )
        pool_relevance_pairs.append(pool_relevance_pair)
        first_items = set(first["candidate_relevance"])
        second_items = set(second["candidate_relevance"])
        if first_items != second_items:
            raise ValueError(
                f"Candidate coverage differs between reviewers for {query_id}"
            )
        query_conflict = pool_relevance_pair[0] != pool_relevance_pair[1]
        if query_conflict:
            conflict_types["pool_relevance"] += 1
        for item_id in sorted(first_items):
            pair = (
                first["candidate_relevance"][item_id],
                second["candidate_relevance"][item_id],
            )
            relevance_pairs.append(pair)
            if pair[0] != pair[1]:
                query_conflict = True
                conflict_types["candidate_relevance"] += 1
        if query_conflict:
            conflict_queries.append(query_id)

    def raw_agreement(pairs: Sequence[tuple[object, object]]) -> float | None:
        if not pairs:
            return None
        return sum(left == right for left, right in pairs) / len(pairs)

    adjudication_rate = (
        len(conflict_queries) / double_reviewed if double_reviewed else None
    )
    return {
        "double_reviewed_query_count": double_reviewed,
        "pool_relevance_pair_count": len(pool_relevance_pairs),
        "pool_relevance_raw_agreement": raw_agreement(pool_relevance_pairs),
        "pool_relevance_cohen_kappa": _cohen_kappa_multiclass(
            pool_relevance_pairs
        ),
        "candidate_pair_count": len(relevance_pairs),
        "candidate_raw_agreement": raw_agreement(relevance_pairs),
        "candidate_cohen_kappa": _cohen_kappa_binary(relevance_pairs),
        "conflict_query_ids": conflict_queries,
        "conflict_type_counts": dict(sorted(conflict_types.items())),
        "adjudication_rate": adjudication_rate,
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def grouped_paired_bootstrap(
    rows: Iterable[Mapping[str, Any]],
    *,
    baseline_field: str,
    contender_field: str,
    group_field: str = "group_id",
    repetitions: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 17,
) -> dict[str, Any]:
    """Estimate a paired mean difference while resampling query groups."""

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        group_id = str(row.get(group_field, "")).strip()
        if not group_id:
            raise ValueError(f"Every row needs {group_field}")
        grouped[group_id].append(
            (float(row[baseline_field]), float(row[contender_field]))
        )
    if not grouped:
        raise ValueError("At least one paired row is required")
    group_ids = sorted(grouped)
    observed_pairs = [pair for group_id in group_ids for pair in grouped[group_id]]
    observed_baseline = sum(left for left, _ in observed_pairs) / len(observed_pairs)
    observed_contender = sum(right for _, right in observed_pairs) / len(observed_pairs)

    generator = random.Random(seed)
    sampled_differences: list[float] = []
    for _ in range(repetitions):
        sampled_groups = [generator.choice(group_ids) for _ in group_ids]
        sampled_pairs = [
            pair for group_id in sampled_groups for pair in grouped[group_id]
        ]
        sampled_differences.append(
            sum(right - left for left, right in sampled_pairs)
            / len(sampled_pairs)
        )
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "row_count": len(observed_pairs),
        "group_count": len(group_ids),
        "baseline_mean": observed_baseline,
        "contender_mean": observed_contender,
        "paired_difference": observed_contender - observed_baseline,
        "confidence_level": confidence_level,
        "confidence_interval": [
            _percentile(sampled_differences, alpha),
            _percentile(sampled_differences, 1.0 - alpha),
        ],
        "repetitions": repetitions,
        "seed": seed,
        "sampling_unit": group_field,
    }
