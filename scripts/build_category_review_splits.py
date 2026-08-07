"""Build a frozen, group-isolated split for category human review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY_DIR = PROJECT_ROOT / "outputs/user_library"
DEFAULT_QUALITY_GATE = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_1500_quality_gate.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_1500_category_review_splits.csv"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "records/experiments/category_review_split_001.json"
)
SPLIT_RATIOS = {
    "train": 0.65,
    "validation": 0.175,
    "test": 0.175,
}
SPLIT_LABELS = {
    "train": "训练集",
    "validation": "验证集",
    "test": "测试集",
}
OUTPUT_FIELDS = [
    "item_id",
    "split",
    "split_label",
    "review_required",
    "review_sequence",
    "category",
    "group_id",
    "source_file_name",
    "source_name",
]


class DisjointSet:
    """Small union-find used to keep related pages in one split."""

    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def review_required_ids(
    manifest: list[dict[str, Any]],
    quality_rows: list[dict[str, str]],
) -> set[str]:
    required = {
        row["item_id"]
        for row in quality_rows
        if row.get("item_id")
        and row.get("quality_route", "pass") != "pass"
    }
    required.update(
        row["item_id"]
        for row in manifest
        if row.get("taxonomy_review_status") == "pending"
    )
    return required


def relationship_tokens(row: dict[str, Any]) -> list[str]:
    """Return tokens whose pages must never cross split boundaries."""
    tokens = []
    for field in (
        "perceptual_group",
        "source_group_id",
        "hard_negative_group",
    ):
        value = str(row.get(field, "")).strip()
        if value:
            tokens.append(f"{field}:{value}")
    checksum = str(
        row.get("sha256")
        or row.get("checksum")
        or row.get("content_hash")
        or ""
    ).strip()
    if checksum:
        tokens.append(f"checksum:{checksum}")
    return tokens


def build_groups(
    manifest: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]]]:
    item_ids = [str(row["item_id"]) for row in manifest]
    dsu = DisjointSet(item_ids)
    token_owner: dict[str, str] = {}
    for row in manifest:
        item_id = str(row["item_id"])
        for token in relationship_tokens(row):
            owner = token_owner.setdefault(token, item_id)
            dsu.union(item_id, owner)

    members_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        members_by_root[dsu.find(str(row["item_id"]))].append(row)

    group_by_item: dict[str, str] = {}
    groups: dict[str, list[dict[str, Any]]] = {}
    for members in members_by_root.values():
        sorted_ids = sorted(str(row["item_id"]) for row in members)
        group_id = "grp_" + hashlib.sha256(
            "|".join(sorted_ids).encode("utf-8")
        ).hexdigest()[:12]
        groups[group_id] = sorted(
            members, key=lambda row: str(row["item_id"])
        )
        for item_id in sorted_ids:
            group_by_item[item_id] = group_id
    return group_by_item, groups


def integer_targets(total: int) -> dict[str, int]:
    raw = {
        split: total * ratio for split, ratio in SPLIT_RATIOS.items()
    }
    targets = {split: int(value) for split, value in raw.items()}
    remainder = total - sum(targets.values())
    order = sorted(
        SPLIT_RATIOS,
        key=lambda split: (
            -(raw[split] - targets[split]),
            split,
        ),
    )
    for split in order[:remainder]:
        targets[split] += 1
    return targets


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(
        f"{seed}|{value}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def assign_review_groups(
    groups: dict[str, list[dict[str, Any]]],
    required_ids: set[str],
    *,
    seed: int,
) -> dict[str, str]:
    group_stats = []
    categories = sorted(
        {
            str(row.get("category", ""))
            for members in groups.values()
            for row in members
            if str(row.get("category", ""))
        }
    )
    for group_id, members in groups.items():
        required_members = [
            row
            for row in members
            if str(row["item_id"]) in required_ids
        ]
        if not required_members:
            continue
        group_stats.append(
            {
                "group_id": group_id,
                "size": len(required_members),
                "category_counts": Counter(
                    str(row.get("category", ""))
                    for row in required_members
                ),
                "tie": stable_fraction(group_id, seed),
            }
        )
    group_stats.sort(
        key=lambda row: (-int(row["size"]), float(row["tie"]))
    )

    total_targets = integer_targets(len(required_ids))
    category_totals = Counter(
        str(row.get("category", ""))
        for members in groups.values()
        for row in members
        if str(row["item_id"]) in required_ids
    )
    category_targets = {
        category: integer_targets(category_totals[category])
        for category in categories
    }
    total_counts: Counter[str] = Counter()
    category_counts: dict[str, Counter[str]] = {
        split: Counter() for split in SPLIT_RATIOS
    }
    assignments: dict[str, str] = {}

    for group in group_stats:
        scores: list[tuple[float, float, str]] = []
        for split in SPLIT_RATIOS:
            new_total = total_counts[split] + int(group["size"])
            total_target = max(1, total_targets[split])
            total_fill = new_total / total_target
            total_overflow = max(0, new_total - total_target)
            category_fill = 0.0
            represented = 0
            for category, increment in group["category_counts"].items():
                target = category_targets[category][split]
                if target <= 0:
                    continue
                category_fill += (
                    category_counts[split][category] + int(increment)
                ) / target
                represented += 1
            if represented:
                category_fill /= represented
            score = (
                total_fill * 0.62
                + category_fill * 0.38
                + total_overflow * 4.0
            )
            scores.append(
                (
                    score,
                    stable_fraction(
                        f"{group['group_id']}|{split}", seed
                    ),
                    split,
                )
            )
        split = min(scores)[2]
        assignments[str(group["group_id"])] = split
        total_counts[split] += int(group["size"])
        category_counts[split].update(group["category_counts"])

    return assignments


def build_split_rows(
    manifest: list[dict[str, Any]],
    quality_rows: list[dict[str, str]],
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    required_ids = review_required_ids(manifest, quality_rows)
    known_ids = {str(row["item_id"]) for row in manifest}
    unknown = sorted(required_ids - known_ids)
    if unknown:
        raise ValueError(
            "Quality gate contains unknown item IDs: "
            + ", ".join(unknown[:5])
        )
    group_by_item, groups = build_groups(manifest)
    group_assignments = assign_review_groups(
        groups, required_ids, seed=seed
    )

    rows = []
    for item in manifest:
        item_id = str(item["item_id"])
        group_id = group_by_item[item_id]
        split = group_assignments.get(group_id, "train")
        rows.append(
            {
                "item_id": item_id,
                "split": split,
                "split_label": SPLIT_LABELS[split],
                "review_required": str(
                    item_id in required_ids
                ).lower(),
                "review_sequence": "",
                "category": str(item.get("category", "")),
                "group_id": group_id,
                "source_file_name": str(
                    item.get("source_file_name", "")
                ),
                "source_name": str(
                    item.get("public_source_name")
                    or item.get("source", "")
                ),
            }
        )

    for split in SPLIT_RATIOS:
        selected = [
            row
            for row in rows
            if row["split"] == split
            and parse_bool(row["review_required"])
        ]
        selected.sort(
            key=lambda row: (
                stable_fraction(
                    f"{row['category']}|{row['group_id']}|"
                    f"{row['item_id']}",
                    seed,
                ),
                row["item_id"],
            )
        )
        for sequence, row in enumerate(selected, start=1):
            row["review_sequence"] = sequence
    return rows


def validate_split_rows(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    valid_splits = set(SPLIT_RATIOS)
    if any(row["split"] not in valid_splits for row in rows):
        raise ValueError("Unknown split name.")
    item_ids = [str(row["item_id"]) for row in rows]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Duplicate item IDs in split file.")

    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[str(row["group_id"])].add(str(row["split"]))
    leaking_groups = sorted(
        group_id
        for group_id, splits in group_splits.items()
        if len(splits) > 1
    )
    if leaking_groups:
        raise ValueError(
            "Related pages cross split boundaries: "
            + ", ".join(leaking_groups[:5])
        )

    required = [
        row for row in rows if parse_bool(row["review_required"])
    ]
    return {
        "total_pages": len(rows),
        "review_required_pages": len(required),
        "review_split_counts": dict(
            Counter(str(row["split"]) for row in required)
        ),
        "all_page_split_counts": dict(
            Counter(str(row["split"]) for row in rows)
        ),
        "category_by_split": {
            split: dict(
                sorted(
                    Counter(
                        str(row["category"])
                        for row in required
                        if row["split"] == split
                    ).items()
                )
            )
            for split in SPLIT_RATIOS
        },
        "group_count": len(group_splits),
        "group_leakage_count": 0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library-dir", type=Path, default=DEFAULT_LIBRARY_DIR
    )
    parser.add_argument(
        "--quality-gate", type=Path, default=DEFAULT_QUALITY_GATE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--summary", type=Path, default=DEFAULT_SUMMARY
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing frozen split assignment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.is_file() and not args.overwrite:
        raise FileExistsError(
            f"Frozen split already exists: {output}. "
            "Use --overwrite only when intentionally starting a new experiment."
        )
    manifest = read_jsonl(
        args.library_dir.resolve() / "manifest.jsonl"
    )
    quality_rows = read_csv(args.quality_gate.resolve())
    rows = build_split_rows(manifest, quality_rows, seed=args.seed)
    summary = {
        "status": "frozen",
        "seed": args.seed,
        "ratios": SPLIT_RATIOS,
        **validate_split_rows(rows),
        "rules": [
            "同一原文档、感知近重复组和困难负例组不得跨集合",
            "验证集与测试集人工标签不得参与分类器训练",
            "无须人工审核且不关联留出页的样本仅进入训练集",
        ],
    }
    write_csv(output, rows)
    summary_path = args.summary.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
