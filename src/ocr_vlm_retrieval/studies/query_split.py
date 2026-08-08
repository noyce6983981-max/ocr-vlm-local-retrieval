"""Leakage-aware source selection and paired-query freezing for studies."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

SPLITS = ("calibration", "holdout")
LANGUAGES = ("zh", "en")
STRATA = (
    "attribute_object_binding",
    "spatial_relation",
    "ocr_visual_condition",
    "multi_condition_scene",
)
QUERY_ROLES = ("positive", "single_condition_hard_negative")
CHANGED_CONDITION_KINDS = {
    "object",
    "scene",
    "color",
    "relation",
    "attribute_binding",
    "ocr_text",
}

ALLOWED_CHANGED_KINDS_BY_STRATUM = {
    "attribute_object_binding": {"object", "color", "attribute_binding"},
    "spatial_relation": {"relation"},
    "ocr_visual_condition": {
        "object",
        "color",
        "relation",
        "attribute_binding",
        "ocr_text",
    },
    "multi_condition_scene": {
        "object",
        "scene",
        "color",
        "relation",
        "attribute_binding",
    },
}

_STRATUM_CATEGORIES = {
    "attribute_object_binding": {
        "natural_no_text",
        "ppt_poster_slide",
        "scene_text",
    },
    "spatial_relation": {
        "natural_no_text",
        "ppt_poster_slide",
        "scene_text",
    },
    "ocr_visual_condition": {
        "clear_document",
        "complex_academic",
        "ppt_poster_slide",
        "scene_text",
        "software_web_code",
        "table_form_ticket",
    },
    "multi_condition_scene": {
        "natural_no_text",
        "ppt_poster_slide",
        "scene_text",
    },
}


def _stable_key(seed: str, *values: str) -> str:
    return hashlib.sha256("\0".join((seed, *values)).encode("utf-8")).hexdigest()


def normalize_query(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def source_group_id(row: Mapping[str, Any]) -> str:
    """Match V17's grouping rule so predecessor source neighbors are excluded."""

    hard_group = str(row.get("hard_negative_group", "")).strip()
    if hard_group:
        return f"source-neighbor:{hard_group}"
    source_group = str(row.get("source_group_id", "")).strip()
    if source_group:
        return f"source:{source_group}"
    return f"item:{row['item_id']}"


def source_identity_keys(row: Mapping[str, Any]) -> frozenset[str]:
    """Return source, hard-negative, and perceptual identities for deduplication."""

    keys = {source_group_id(row), f"item:{row['item_id']}"}
    source_group = str(row.get("source_group_id", "")).strip()
    hard_group = str(row.get("hard_negative_group", "")).strip()
    perceptual_group = str(row.get("perceptual_group", "")).strip()
    if source_group:
        keys.add(f"source:{source_group}")
    if hard_group:
        keys.add(f"source-neighbor:{hard_group}")
    if perceptual_group:
        keys.add(f"perceptual:{perceptual_group}")
    return frozenset(keys)


def _eligible(
    row: Mapping[str, Any],
    ocr: Mapping[str, Any],
    stratum: str,
) -> bool:
    if bool(row.get("privacy_review_required", False)):
        return False
    if str(row.get("dedup_role", "")).strip() == "duplicate":
        return False
    if not str(row.get("source_path", "")).strip():
        return False
    category = str(row.get("taxonomy_v1_category") or row.get("category", "")).strip()
    if category not in _STRATUM_CATEGORIES[stratum]:
        return False
    if stratum != "ocr_visual_condition":
        return True
    status = str(ocr.get("status", "")).strip()
    characters = int(float(ocr.get("character_count", 0) or 0))
    boxes = int(float(ocr.get("text_box_count", 0) or 0))
    return (
        bool(row.get("has_text_expected", False))
        and status in {"cached", "completed", "success"}
        and characters >= 8
        and boxes >= 1
    )


def _diverse_order(
    rows: Sequence[Mapping[str, Any]], *, seed: str, stratum: str
) -> list[Mapping[str, Any]]:
    by_dataset: dict[str, deque[Mapping[str, Any]]] = defaultdict(deque)
    for row in rows:
        dataset = str(
            row.get("public_source_name") or row.get("dataset_name") or "unknown"
        )
        by_dataset[dataset].append(row)
    for dataset, sources in by_dataset.items():
        by_dataset[dataset] = deque(
            sorted(
                sources,
                key=lambda row: _stable_key(
                    seed, stratum, dataset, str(row["item_id"])
                ),
            )
        )
    dataset_order = sorted(
        by_dataset, key=lambda value: _stable_key(seed, stratum, value)
    )
    ordered: list[Mapping[str, Any]] = []
    while any(by_dataset.values()):
        for dataset in dataset_order:
            if by_dataset[dataset]:
                ordered.append(by_dataset[dataset].popleft())
    return ordered


def _specificity_order(
    rows: Sequence[Mapping[str, Any]],
    ocr_by_item: Mapping[str, Mapping[str, Any]],
    *,
    seed: str,
    stratum: str,
) -> list[Mapping[str, Any]]:
    """Prefer sources eligible for fewer strata before consuming shared sources."""

    by_eligible_count: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        ocr = ocr_by_item.get(str(row["item_id"]), {})
        count = sum(_eligible(row, ocr, candidate) for candidate in STRATA)
        by_eligible_count[count].append(row)
    ordered: list[Mapping[str, Any]] = []
    for count in sorted(by_eligible_count):
        ordered.extend(
            _diverse_order(by_eligible_count[count], seed=seed, stratum=stratum)
        )
    return ordered


def build_authoring_queue(
    manifest: Iterable[Mapping[str, Any]],
    ocr_by_item: Mapping[str, Mapping[str, Any]],
    *,
    excluded_identity_keys: Iterable[str] = (),
    dominant_colors_by_item: Mapping[str, Sequence[str]] | None = None,
    seed: str = "v18-condition-aware-listwise-160",
    sources_per_stratum: int = 20,
) -> list[dict[str, Any]]:
    """Select balanced, predecessor-disjoint sources without running retrieval."""

    if sources_per_stratum < 4 or sources_per_stratum % 4:
        raise ValueError("sources_per_stratum must be positive and divisible by four")
    rows = [dict(row) for row in manifest]
    excluded = {str(value) for value in excluded_identity_keys if value}
    selected_keys: set[str] = set()
    chosen_by_stratum: dict[str, list[Mapping[str, Any]]] = {}
    colors = dominant_colors_by_item or {}

    selection_order = (
        "ocr_visual_condition",
        "spatial_relation",
        "attribute_object_binding",
        "multi_condition_scene",
    )
    for stratum in selection_order:
        candidates = [
            row
            for row in rows
            if _eligible(row, ocr_by_item.get(str(row["item_id"]), {}), stratum)
        ]
        selected: list[Mapping[str, Any]] = []
        for row in _specificity_order(
            candidates, ocr_by_item, seed=seed, stratum=stratum
        ):
            identities = source_identity_keys(row)
            if identities & excluded or identities & selected_keys:
                continue
            selected.append(row)
            selected_keys.update(identities)
            if len(selected) == sources_per_stratum:
                break
        if len(selected) != sources_per_stratum:
            raise ValueError(
                f"Need {sources_per_stratum} eligible sources for {stratum}; "
                f"selected {len(selected)}"
            )
        chosen_by_stratum[stratum] = selected

    queue: list[dict[str, Any]] = []
    per_cell = sources_per_stratum // 4
    slots = [
        (split, language)
        for split in SPLITS
        for language in LANGUAGES
        for _ in range(per_cell)
    ]
    for stratum in STRATA:
        selected = sorted(
            chosen_by_stratum[stratum],
            key=lambda row: _stable_key(seed, stratum, str(row["item_id"])),
        )
        assigned_slots = sorted(
            slots,
            key=lambda slot: _stable_key(seed, stratum, slot[0], slot[1]),
        )
        for row, (split, language) in zip(selected, assigned_slots, strict=True):
            item_id = str(row["item_id"])
            ocr = ocr_by_item.get(item_id, {})
            queue.append(
                {
                    "source_item_id": item_id,
                    "group_id": source_group_id(row),
                    "split": split,
                    "stratum": stratum,
                    "language_target": language,
                    "image_path": str(row.get("source_path", "")),
                    "ocr_excerpt": str(ocr.get("ocr_excerpt", ""))[:600],
                    "dominant_colors": list(colors.get(item_id, ()))[:3],
                    "source_dataset": str(row.get("public_source_name", "")),
                    "authoring_status": "pending_human_authoring",
                }
            )
    queue.sort(
        key=lambda row: (
            STRATA.index(str(row["stratum"])),
            SPLITS.index(str(row["split"])),
            LANGUAGES.index(str(row["language_target"])),
            _stable_key(seed, str(row["source_item_id"])),
        )
    )
    for index, row in enumerate(queue, start=1):
        row["authoring_id"] = f"v18_source_{index:03d}"
    validate_authoring_queue(
        queue,
        sources_per_stratum=sources_per_stratum,
        excluded_identity_keys=excluded,
    )
    return queue


def validate_authoring_queue(
    rows: Sequence[Mapping[str, Any]],
    *,
    sources_per_stratum: int,
    excluded_identity_keys: Iterable[str] = (),
) -> None:
    expected = sources_per_stratum * len(STRATA)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} authoring sources, got {len(rows)}")
    item_ids = [str(row.get("source_item_id", "")) for row in rows]
    authoring_ids = [str(row.get("authoring_id", "")) for row in rows]
    if any(not value for value in item_ids + authoring_ids):
        raise ValueError("authoring and source item IDs must be present")
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("source items must be unique")
    if len(authoring_ids) != len(set(authoring_ids)):
        raise ValueError("authoring IDs must be unique")
    excluded = set(excluded_identity_keys)
    if any(str(row["group_id"]) in excluded for row in rows):
        raise ValueError("authoring queue overlaps excluded predecessor groups")
    stratum_counts = Counter(str(row.get("stratum", "")) for row in rows)
    if stratum_counts != Counter({name: sources_per_stratum for name in STRATA}):
        raise ValueError("authoring queue is not balanced by stratum")
    per_cell = sources_per_stratum // 4
    cell_counts = Counter(
        (
            str(row.get("stratum", "")),
            str(row.get("split", "")),
            str(row.get("language_target", "")),
        )
        for row in rows
    )
    expected_cells = Counter(
        {
            (stratum, split, language): per_cell
            for stratum in STRATA
            for split in SPLITS
            for language in LANGUAGES
        }
    )
    if cell_counts != expected_cells:
        raise ValueError(
            "authoring queue is not balanced by stratum, split, and language"
        )


def validate_authored_pair(
    positive_query: str,
    hard_negative_query: str,
    *,
    language_target: str,
    changed_condition_kind: str,
    single_condition_confirmed: bool,
) -> None:
    positive = normalize_query(positive_query)
    negative = normalize_query(hard_negative_query)
    if len(positive) < 8 or len(negative) < 8:
        raise ValueError(
            "both queries must contain at least eight normalized characters"
        )
    if positive == negative:
        raise ValueError("positive and hard-negative queries must differ")
    if changed_condition_kind not in CHANGED_CONDITION_KINDS:
        raise ValueError("changed_condition_kind is invalid")
    if not single_condition_confirmed:
        raise ValueError("the author must confirm that exactly one condition changed")
    has_cjk_positive = bool(re.search(r"[\u3400-\u9fff]", positive))
    has_cjk_negative = bool(re.search(r"[\u3400-\u9fff]", negative))
    if language_target == "zh" and not (has_cjk_positive and has_cjk_negative):
        raise ValueError("Chinese-target queries must both contain Chinese text")
    if language_target == "en" and (has_cjk_positive or has_cjk_negative):
        raise ValueError("English-target queries must not contain Chinese text")
    if language_target not in LANGUAGES:
        raise ValueError("language_target is invalid")


def freeze_authored_queries(
    queue: Sequence[Mapping[str, Any]],
    submissions: Iterable[Mapping[str, Any]],
    *,
    study_id: str,
) -> list[dict[str, Any]]:
    """Expand approved source-level query pairs into a frozen query set."""

    submission_by_id: dict[str, Mapping[str, Any]] = {}
    for submission in submissions:
        authoring_id = str(submission.get("authoring_id", "")).strip()
        if not authoring_id or authoring_id in submission_by_id:
            raise ValueError("submissions need unique authoring IDs")
        submission_by_id[authoring_id] = submission
    queue_ids = {str(row["authoring_id"]) for row in queue}
    if set(submission_by_id) != queue_ids:
        missing = sorted(queue_ids - set(submission_by_id))
        unknown = sorted(set(submission_by_id) - queue_ids)
        raise ValueError(
            f"authoring coverage mismatch; missing={missing}, unknown={unknown}"
        )

    frozen: list[dict[str, Any]] = []
    for source in queue:
        submission = submission_by_id[str(source["authoring_id"])]
        if str(submission.get("review_action", "")) != "approve":
            raise ValueError(f"source pair is not approved: {source['authoring_id']}")
        reviewer_id = str(submission.get("reviewer_id", "")).strip()
        if not reviewer_id:
            raise ValueError(f"reviewer ID is missing: {source['authoring_id']}")
        positive = str(submission.get("positive_query", "")).strip()
        negative = str(submission.get("hard_negative_query", "")).strip()
        changed_kind = str(submission.get("changed_condition_kind", "")).strip()
        confirmed = bool(submission.get("single_condition_confirmed", False))
        stratum = str(source["stratum"])
        if changed_kind not in ALLOWED_CHANGED_KINDS_BY_STRATUM[stratum]:
            raise ValueError(
                f"changed condition {changed_kind!r} is invalid for {stratum}"
            )
        validate_authored_pair(
            positive,
            negative,
            language_target=str(source["language_target"]),
            changed_condition_kind=changed_kind,
            single_condition_confirmed=confirmed,
        )
        base = {
            "study_id": study_id,
            "split": source["split"],
            "group_id": source["group_id"],
            "stratum": stratum,
            "language": source["language_target"],
            "source_item_id": source["source_item_id"],
            "query_reviewer_id": reviewer_id,
            "review_status": "human_query_approved",
        }
        frozen.append({**base, "query": positive, "query_role": QUERY_ROLES[0]})
        frozen.append(
            {
                **base,
                "query": negative,
                "query_role": QUERY_ROLES[1],
                "changed_condition_kind": changed_kind,
            }
        )
    normalized = [normalize_query(str(row["query"])) for row in frozen]
    if len(normalized) != len(set(normalized)):
        raise ValueError("frozen queries must be globally unique")
    frozen.sort(
        key=lambda row: (
            SPLITS.index(str(row["split"])),
            str(row["group_id"]),
            QUERY_ROLES.index(str(row["query_role"])),
        )
    )
    for index, row in enumerate(frozen, start=1):
        row["query_id"] = f"v18_query_{index:03d}"
    validate_frozen_queries(frozen, expected_count=len(queue) * 2)
    return frozen


def validate_frozen_queries(
    rows: Sequence[Mapping[str, Any]], *, expected_count: int
) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} frozen queries, got {len(rows)}")
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group_id"])].append(row)
    for group_id, grouped in by_group.items():
        if len(grouped) != 2:
            raise ValueError(f"group {group_id} must contain exactly two queries")
        if {str(row["query_role"]) for row in grouped} != set(QUERY_ROLES):
            raise ValueError(f"group {group_id} must contain both query roles")
        if len({str(row["split"]) for row in grouped}) != 1:
            raise ValueError(f"group {group_id} crosses splits")


def query_set_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    material = [
        {
            "query_id": str(row["query_id"]),
            "query": normalize_query(str(row["query"])),
            "split": str(row["split"]),
            "group_id": str(row["group_id"]),
            "query_role": str(row["query_role"]),
        }
        for row in sorted(rows, key=lambda value: str(value["query_id"]))
    ]
    return hashlib.sha256(
        json.dumps(
            material, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
