from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "data/evaluation/v19_selective_intervention/protocol_draft.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "records/private/v19/selective_intervention/source_families_draft.jsonl"
)

STRATUM_CATEGORIES = {
    "ocr_literal_lookup": {
        "table_form_ticket",
        "general_text_document",
        "complex_academic",
    },
    "pure_visual": {"natural_no_text", "scene_text"},
    "layout_table": {
        "table_form_ticket",
        "ppt_poster_slide",
        "software_web_code",
    },
    "entity_identifier": {
        "table_form_ticket",
        "general_text_document",
        "scene_text",
    },
    "visual_compositional": {
        "natural_no_text",
        "scene_text",
        "ppt_poster_slide",
        "software_web_code",
        "complex_academic",
    },
    "text_visual_compositional": {
        "scene_text",
        "ppt_poster_slide",
        "software_web_code",
    },
    "topic_discovery": {
        "complex_academic",
        "general_text_document",
        "ppt_poster_slide",
    },
    "relationship_scene": {
        "natural_no_text",
        "scene_text",
        "ppt_poster_slide",
        "software_web_code",
    },
}

OCR_REQUIRED_STRATA = {
    "ocr_literal_lookup",
    "entity_identifier",
    "text_visual_compositional",
    "topic_discovery",
}

CATEGORY_PRIORITY = {
    "pure_visual": (
        "natural_no_text",
        "scene_text",
    ),
    "visual_compositional": (
        "natural_no_text",
        "scene_text",
        "software_web_code",
        "ppt_poster_slide",
        "complex_academic",
    ),
    "relationship_scene": (
        "software_web_code",
        "ppt_poster_slide",
        "scene_text",
        "natural_no_text",
    ),
}

SELECTION_ORDER = (
    "pure_visual",
    "relationship_scene",
    "text_visual_compositional",
    "visual_compositional",
    "ocr_literal_lookup",
    "layout_table",
    "entity_identifier",
    "topic_discovery",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _stable_key(seed: str, *parts: str) -> str:
    material = "\x1f".join((seed, *parts))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
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


def load_excluded_item_ids(evaluation_root: Path) -> set[str]:
    paths = (
        evaluation_root / "frozen_queries.jsonl",
        evaluation_root / "calibration/frozen_calibration_queries.jsonl",
        evaluation_root / "holdout/frozen_holdout_queries.jsonl",
    )
    excluded: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        for row in _load_jsonl(path):
            item_id = str(row.get("source_item_id", "")).strip()
            if item_id:
                excluded.add(item_id)
    return excluded


def _eligible_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    stratum: str,
    ocr_character_counts: Mapping[str, int],
) -> bool:
    if len(rows) < 2:
        return False
    allowed = STRATUM_CATEGORIES[stratum]
    candidates = [row for row in rows if str(row.get("category", "")) in allowed]
    if len(candidates) < 2:
        return False
    if stratum not in OCR_REQUIRED_STRATA:
        return True
    minimum = 20 if stratum == "text_visual_compositional" else 80
    return sum(
        ocr_character_counts.get(str(row.get("item_id", "")), 0) >= minimum
        for row in candidates
    ) >= 2


def _group_category_priority(
    rows: Sequence[Mapping[str, Any]], stratum: str
) -> int:
    preference = CATEGORY_PRIORITY.get(stratum)
    if preference is None:
        return 0
    categories = {str(row.get("category", "")) for row in rows}
    return min(
        (preference.index(category) for category in categories if category in preference),
        default=len(preference),
    )


def _source_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "item_id": str(row.get("item_id", "")),
        "image_path": str(row.get("source_path", "")),
        "display_name": str(row.get("display_name_zh", "")),
        "category": str(row.get("category", "")),
        "language": str(row.get("language", "")),
        "source_dataset": str(row.get("public_source_name", "")),
        "source_file": str(row.get("public_source_file", "")),
        "source_url": str(row.get("public_source_url", "")),
        "license": str(row.get("license", "")),
        "license_url": str(row.get("license_url", "")),
        "hard_negative_group": str(row.get("hard_negative_group", "")),
    }


def _split_slots(stratum_counts: Mapping[str, int]) -> dict[str, list[str]]:
    slots: dict[str, list[str]] = {}
    odd_index = 0
    for stratum, count in stratum_counts.items():
        development = count // 2
        if count % 2:
            development += int(odd_index % 2 == 0)
            odd_index += 1
        slots[stratum] = ["development"] * development + [
            "holdout"
        ] * (count - development)
    return slots


def build_source_families(
    manifest: Sequence[Mapping[str, Any]],
    *,
    stratum_counts: Mapping[str, int],
    excluded_item_ids: Iterable[str],
    ocr_character_counts: Mapping[str, int],
    seed: str,
) -> list[dict[str, Any]]:
    if set(stratum_counts) != set(STRATUM_CATEGORIES):
        raise ValueError("content strata do not match the selective-intervention schema")
    excluded = {str(value) for value in excluded_item_ids if value}
    excluded_groups = {
        str(row.get("hard_negative_group", ""))
        for row in manifest
        if str(row.get("item_id", "")) in excluded
    }
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in manifest:
        item_id = str(row.get("item_id", ""))
        group = str(row.get("hard_negative_group", ""))
        if (
            not item_id
            or not group
            or item_id in excluded
            or group in excluded_groups
            or bool(row.get("privacy_review_required", False))
            or str(row.get("dedup_role", "")) == "duplicate_variant"
        ):
            continue
        grouped[group].append(row)

    selected_groups: set[str] = set()
    families: list[dict[str, Any]] = []
    split_slots = _split_slots(stratum_counts)
    for stratum in SELECTION_ORDER:
        count = stratum_counts[stratum]
        candidates = [
            (group, rows)
            for group, rows in grouped.items()
            if group not in selected_groups
            and _eligible_group(
                rows, stratum=stratum, ocr_character_counts=ocr_character_counts
            )
        ]
        candidates.sort(
            key=lambda pair: (
                _group_category_priority(pair[1], stratum),
                _stable_key(seed, stratum, pair[0]),
            )
        )
        if len(candidates) < count:
            raise ValueError(
                f"Need {count} unused source groups for {stratum}; "
                f"found {len(candidates)}"
            )
        for index, (group, group_rows) in enumerate(candidates[:count], start=1):
            allowed = STRATUM_CATEGORIES[stratum]
            usable = [
                row for row in group_rows if str(row.get("category", "")) in allowed
            ]
            if stratum in OCR_REQUIRED_STRATA:
                minimum = 20 if stratum == "text_visual_compositional" else 80
                usable.sort(
                    key=lambda row: (
                        -ocr_character_counts.get(str(row.get("item_id", "")), 0),
                        _stable_key(seed, stratum, str(row.get("item_id", ""))),
                    )
                )
                usable = [
                    row
                    for row in usable
                    if ocr_character_counts.get(str(row.get("item_id", "")), 0)
                    >= minimum
                ]
            else:
                usable.sort(
                    key=lambda row: _stable_key(
                        seed, stratum, str(row.get("item_id", ""))
                    )
                )
            target, neighbor = usable[:2]
            selected_groups.add(group)
            families.append(
                {
                    "family_id": f"v19si_{stratum}_{index:02d}",
                    "split": split_slots[stratum][index - 1],
                    "content_stratum": stratum,
                    "language_target": "zh",
                    "target": _source_snapshot(target),
                    "neighbor": _source_snapshot(neighbor),
                    "query_drafts": {
                        "answerable_positive": "",
                        "paraphrase_positive": "",
                        "single_condition_hard_negative": "",
                        "unanswerable_neighbor": "",
                    },
                    "changed_condition_kind": "",
                    "selection_status": "source_bound_pending_query_draft",
                    "review_status": "pending_human_review_not_frozen",
                }
            )
    if len(families) != sum(stratum_counts.values()):
        raise AssertionError("unexpected family count")
    if len(selected_groups) != len(families):
        raise AssertionError("source groups must be unique")
    source_ids = [
        str(family[role]["item_id"])
        for family in families
        for role in ("target", "neighbor")
    ]
    if len(source_ids) != len(set(source_ids)):
        raise AssertionError("source items must be unique across families")
    if excluded & set(source_ids):
        raise AssertionError("source queue overlaps the predecessor evaluation")
    stratum_order = {name: index for index, name in enumerate(stratum_counts)}
    families.sort(
        key=lambda row: (
            stratum_order[str(row["content_stratum"])],
            str(row["family_id"]),
        )
    )
    return families


def load_ocr_character_counts(summary_path: Path) -> dict[str, int]:
    import csv

    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            str(row["item_id"]): int(row.get("character_count", "0") or 0)
            for row in csv.DictReader(handle)
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a fresh, predecessor-disjoint V19 source review queue."
    )
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", default="v19-selective-intervention-source-v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    library_root = args.library_root.resolve()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8-sig"))
    manifest = _load_jsonl(library_root / "outputs/user_library/manifest.jsonl")
    excluded = load_excluded_item_ids(library_root / "data/evaluation/v18")
    ocr_counts = load_ocr_character_counts(
        library_root / "outputs/user_library/ocr/summary.csv"
    )
    families = build_source_families(
        manifest,
        stratum_counts=protocol["content_strata"],
        excluded_item_ids=excluded,
        ocr_character_counts=ocr_counts,
        seed=args.seed,
    )
    _atomic_jsonl(args.output, families)
    split_counts = {
        split: sum(family["split"] == split for family in families)
        for split in ("development", "holdout")
    }
    print(f"Prepared {len(families)} source-bound families: {split_counts}")
    print(f"Excluded {len(excluded)} predecessor source items and their groups")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
