"""Build two blind full-review queues from the frozen human-review cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs/user_library/manifest.jsonl"
DEFAULT_REVIEWS = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_1500_quality_human_reviews.csv"
)
DEFAULT_SPLITS = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_1500_category_review_splits.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/evaluation"
EXPECTED_COUNTS = {"train": 410, "validation": 95, "test": 95}
BLIND_SPLIT_CODES = {
    "train": "tr",
    "validation": "va",
    "test": "te",
}
QUEUE_FIELDS = [
    "blind_id",
    "item_id",
    "split",
    "split_sequence",
    "source_path",
    "ocr_json_path",
]
LABEL_FIELDS = [
    "blind_id",
    "item_id",
    "split",
    "assistant_category",
    "assistant_notes",
    "assistant_confidence",
    "status",
    "reviewed_at",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def blind_order(item_id: str, round_number: int) -> str:
    return hashlib.sha256(
        f"assistant-round-{round_number}:{item_id}".encode("utf-8")
    ).hexdigest()


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_queue(
    *,
    manifest_path: Path,
    reviews_path: Path,
    splits_path: Path,
    round_number: int,
) -> list[dict[str, Any]]:
    manifest = {
        row["item_id"]: row for row in read_jsonl(manifest_path)
    }
    usable_ids = {
        row["item_id"]
        for row in read_csv(reviews_path)
        if row.get("decision") in {"accepted", "reclassified"}
    }
    split_rows = {
        row["item_id"]: row for row in read_csv(splits_path)
    }
    missing = sorted(
        item_id
        for item_id in usable_ids
        if item_id not in manifest or item_id not in split_rows
    )
    if missing:
        raise ValueError(
            "Reviewed pages missing manifest or split rows: "
            + ", ".join(missing[:5])
        )

    selected: list[tuple[str, str]] = [
        (item_id, split_rows[item_id]["split"])
        for item_id in usable_ids
    ]
    counts = Counter(split for _, split in selected)
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(
            f"Unexpected frozen cohort counts: {dict(counts)}"
        )

    selected.sort(
        key=lambda pair: (
            {"train": 0, "validation": 1, "test": 2}[pair[1]],
            blind_order(pair[0], round_number),
        )
    )
    sequence_by_split: Counter[str] = Counter()
    queue: list[dict[str, Any]] = []
    for item_id, split in selected:
        sequence_by_split[split] += 1
        item = manifest[item_id]
        queue.append(
            {
                "blind_id": (
                    f"r{round_number}_{BLIND_SPLIT_CODES[split]}_"
                    f"{sequence_by_split[split]:03d}"
                ),
                "item_id": item_id,
                "split": split,
                "split_sequence": sequence_by_split[split],
                "source_path": item["source_path"],
                "ocr_json_path": (
                    f"outputs/user_library/ocr/json/{item_id}.json"
                ),
            }
        )
    return queue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    args = parser.parse_args()

    report: dict[str, Any] = {"status": "success", "rounds": {}}
    for round_number in (4, 5):
        queue = build_queue(
            manifest_path=args.manifest.resolve(),
            reviews_path=args.reviews.resolve(),
            splits_path=args.splits.resolve(),
            round_number=round_number,
        )
        queue_path = (
            args.output_dir
            / f"assistant_full_review_round{round_number}_queue.csv"
        )
        labels_path = (
            args.output_dir
            / f"assistant_full_review_round{round_number}_labels.csv"
        )
        write_csv(queue_path, queue, QUEUE_FIELDS)
        if not labels_path.exists():
            write_csv(labels_path, [], LABEL_FIELDS)
        report["rounds"][str(round_number)] = {
            "queue": str(queue_path.relative_to(PROJECT_ROOT)),
            "labels": str(labels_path.relative_to(PROJECT_ROOT)),
            "count": len(queue),
            "split_counts": dict(
                Counter(row["split"] for row in queue)
            ),
        }

    report_path = (
        args.output_dir / "assistant_full_review_protocol.json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
