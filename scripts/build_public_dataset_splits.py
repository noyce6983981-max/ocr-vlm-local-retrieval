"""Build leakage-safe 120/40/40 splits for the 200-page candidate set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
CATEGORY_TARGETS = {
    "clear_document": (18, 6, 6),
    "complex_academic": (18, 6, 6),
    "table_form_ticket": (18, 6, 6),
    "ppt_poster_slide": (15, 5, 5),
    "software_web_code": (12, 4, 4),
    "scene_text": (15, 5, 5),
    "degraded_document": (12, 4, 4),
    "natural_no_text": (12, 4, 4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=Path("data/evaluation/public_dataset_200_sources.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/public_dataset_200_splits.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "data/evaluation/public_dataset_200_splits_summary.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def stable_random_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def normalized_group(row: dict[str, str]) -> str:
    return row["perceptual_group"] or f"unique:{row['filename']}"


def choose_multi_group_assignment(
    groups: list[tuple[str, list[dict[str, str]]]],
    seed: int,
) -> dict[str, str]:
    targets = {
        category: dict(zip(SPLITS, counts))
        for category, counts in CATEGORY_TARGETS.items()
    }
    group_counts = {
        group_id: Counter(row["category"] for row in rows)
        for group_id, rows in groups
    }
    ordered = sorted(
        groups,
        key=lambda item: (
            -len(item[1]),
            stable_random_key(item[0], seed),
        ),
    )
    used = {
        category: {split: 0 for split in SPLITS}
        for category in CATEGORY_TARGETS
    }
    assignment: dict[str, str] = {}
    best: tuple[float, dict[str, str]] | None = None
    total_multi = Counter(
        row["category"] for _, rows in groups for row in rows
    )
    desired_ratios = dict(zip(SPLITS, (0.6, 0.2, 0.2)))

    def search(index: int) -> None:
        nonlocal best
        if index == len(ordered):
            score = 0.0
            for category, total in total_multi.items():
                if not total:
                    continue
                for split in SPLITS:
                    actual = used[category][split] / total
                    score += (
                        actual - desired_ratios[split]
                    ) ** 2
            signature = "|".join(
                f"{key}:{assignment[key]}" for key in sorted(assignment)
            )
            score += int(
                stable_random_key(signature, seed)[:8], 16
            ) / 16**8 * 1e-9
            if best is None or score < best[0]:
                best = (score, dict(assignment))
            return

        group_id, _ = ordered[index]
        counts = group_counts[group_id]
        split_order = sorted(
            SPLITS,
            key=lambda split: stable_random_key(
                f"{group_id}:{split}", seed
            ),
        )
        for split in split_order:
            if any(
                used[category][split] + count
                > targets[category][split]
                for category, count in counts.items()
            ):
                continue
            assignment[group_id] = split
            for category, count in counts.items():
                used[category][split] += count
            search(index + 1)
            for category, count in counts.items():
                used[category][split] -= count
            del assignment[group_id]

    search(0)
    if best is None:
        raise ValueError("No leakage-safe split assignment is feasible.")
    return best[1]


def build_split_rows(
    source_rows: list[dict[str, str]],
    seed: int = 20260729,
) -> list[dict[str, Any]]:
    expected_counts = {
        category: sum(counts)
        for category, counts in CATEGORY_TARGETS.items()
    }
    actual_counts = Counter(row["category"] for row in source_rows)
    if actual_counts != Counter(expected_counts):
        raise ValueError(
            f"Unexpected category counts: {dict(actual_counts)}"
        )

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        grouped[normalized_group(row)].append(row)
    multi_groups = [
        (group_id, rows)
        for group_id, rows in grouped.items()
        if len(rows) > 1
    ]
    assignment = choose_multi_group_assignment(multi_groups, seed)
    row_split: dict[str, str] = {}
    used = {
        category: Counter()
        for category in CATEGORY_TARGETS
    }
    for group_id, split in assignment.items():
        for row in grouped[group_id]:
            row_split[row["filename"]] = split
            used[row["category"]][split] += 1

    randomizer = random.Random(seed)
    for category, counts in CATEGORY_TARGETS.items():
        unique_rows = [
            row
            for row in source_rows
            if row["category"] == category
            and len(grouped[normalized_group(row)]) == 1
        ]
        randomizer.shuffle(unique_rows)
        cursor = 0
        for split, target in zip(SPLITS, counts):
            needed = target - used[category][split]
            if needed < 0:
                raise ValueError(
                    f"Grouped rows exceed {category}/{split} target."
                )
            for row in unique_rows[cursor : cursor + needed]:
                row_split[row["filename"]] = split
            cursor += needed
        if cursor != len(unique_rows):
            raise ValueError(f"Unassigned unique rows in {category}.")

    output: list[dict[str, Any]] = []
    for row in source_rows:
        output.append(
            {
                "filename": row["filename"],
                "library_item_id": f"user_{row['item_id'][:12]}",
                "category": row["category"],
                "split": row_split[row["filename"]],
                "perceptual_group": normalized_group(row),
                "hard_negative_group": row["hard_negative_group"],
                "has_text": row["has_text"],
                "review_status": row["review_status"],
                "privacy_review_required": row[
                    "privacy_review_required"
                ],
                "category_review_required": row[
                    "category_review_required"
                ],
            }
        )
    return sorted(output, key=lambda row: row["filename"])


def validate_splits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts = Counter(row["split"] for row in rows)
    if split_counts != Counter(
        {"train": 120, "validation": 40, "test": 40}
    ):
        raise ValueError(f"Unexpected split counts: {dict(split_counts)}")
    category_split_counts = {
        category: Counter(
            row["split"] for row in rows if row["category"] == category
        )
        for category in CATEGORY_TARGETS
    }
    for category, targets in CATEGORY_TARGETS.items():
        expected = Counter(dict(zip(SPLITS, targets)))
        if category_split_counts[category] != expected:
            raise ValueError(
                f"Unexpected {category} split counts: "
                f"{dict(category_split_counts[category])}"
            )
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[row["perceptual_group"]].add(row["split"])
    leaked = sorted(
        group for group, splits in group_splits.items() if len(splits) > 1
    )
    if leaked:
        raise ValueError(
            "Perceptual groups leaked across splits: " + ", ".join(leaked)
        )
    return {
        "status": "success",
        "total_pages": len(rows),
        "split_counts": dict(split_counts),
        "category_split_counts": {
            category: dict(counts)
            for category, counts in category_split_counts.items()
        },
        "perceptual_group_leakage_count": 0,
        "pending_review_pages": sum(
            row["review_status"] == "待人工审核" for row in rows
        ),
        "privacy_review_pages": sum(
            str(row["privacy_review_required"]).lower() == "true"
            for row in rows
        ),
    }


def main() -> None:
    args = parse_args()
    source_path = project_path(args.source_csv)
    output_path = project_path(args.output)
    summary_path = project_path(args.summary)
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = build_split_rows(source_rows, args.seed)
    summary = validate_splits(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
