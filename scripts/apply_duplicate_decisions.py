"""Quarantine visually verified duplicate variants from an active library."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "outputs/user_library/manifest.jsonl"
DEFAULT_DECISIONS_PATH = (
    PROJECT_ROOT
    / "data/evaluation/public_dataset_1500_duplicate_decisions.csv"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "outputs/user_library/duplicate_quarantine_report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=DEFAULT_DECISIONS_PATH,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_decisions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def apply_duplicate_decisions(
    manifest: list[dict[str, Any]],
    decisions: list[dict[str, str]],
    *,
    applied_at: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {row["item_id"]: row for row in manifest}
    if len(by_id) != len(manifest):
        raise ValueError("Manifest contains duplicate item IDs.")

    remove_to_decision: dict[str, dict[str, str]] = {}
    representative_counts: dict[str, int] = {}
    for decision in decisions:
        keep_id = decision["keep_item_id"].strip()
        remove_id = decision["remove_item_id"].strip()
        expected_group = decision["perceptual_group"].strip()
        if keep_id == remove_id:
            raise ValueError(f"Representative cannot remove itself: {keep_id}")
        if keep_id not in by_id or remove_id not in by_id:
            raise ValueError(
                f"Unknown duplicate decision IDs: {keep_id}, {remove_id}"
            )
        if remove_id in remove_to_decision:
            raise ValueError(f"Duplicate remove decision: {remove_id}")
        keep_group = str(by_id[keep_id].get("perceptual_group", ""))
        remove_group = str(by_id[remove_id].get("perceptual_group", ""))
        if (
            not expected_group
            or keep_group != expected_group
            or remove_group != expected_group
        ):
            raise ValueError(
                f"Perceptual group mismatch for {keep_id} and {remove_id}"
            )
        remove_to_decision[remove_id] = decision
        representative_counts[keep_id] = (
            representative_counts.get(keep_id, 0) + 1
        )

    active_before = sum(
        bool(row.get("search_enabled", True)) for row in manifest
    )
    newly_quarantined = 0
    updated: list[dict[str, Any]] = []
    for source_row in manifest:
        row = dict(source_row)
        item_id = row["item_id"]
        if item_id in representative_counts:
            row["dedup_role"] = "representative"
            row["dedup_variant_count"] = representative_counts[item_id]
        decision = remove_to_decision.get(item_id)
        if decision is not None:
            if bool(row.get("search_enabled", True)):
                newly_quarantined += 1
            row["search_enabled"] = False
            row["review_status"] = "近重复隔离"
            row["dedup_role"] = "duplicate_variant"
            row["duplicate_of"] = decision["keep_item_id"].strip()
            row["dedup_reason"] = decision["reason"].strip()
            row["dedup_applied_at"] = applied_at
        updated.append(row)

    active_after = sum(
        bool(row.get("search_enabled", True)) for row in updated
    )
    report = {
        "status": "success",
        "applied_at": applied_at,
        "decision_count": len(decisions),
        "newly_quarantined": newly_quarantined,
        "active_before": active_before,
        "active_after": active_after,
        "total_pages": len(updated),
        "quarantined_duplicate_ids": sorted(remove_to_decision),
    }
    return updated, report


def publish_duplicate_decisions(
    manifest_path: Path,
    decisions_path: Path,
    report_path: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    timestamp = now or datetime.now(timezone.utc)
    applied_at = timestamp.isoformat(timespec="seconds")
    manifest = read_jsonl(manifest_path)
    decisions = read_decisions(decisions_path)
    updated, report = apply_duplicate_decisions(
        manifest,
        decisions,
        applied_at=applied_at,
    )

    backup_dir = manifest_path.parent / "manifest_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = (
        backup_dir
        / f"manifest_before_dedup_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jsonl"
    )
    shutil.copy2(manifest_path, backup_path)

    temporary_path = manifest_path.with_suffix(".jsonl.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for row in updated:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary_path, manifest_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report["backup_path"] = str(backup_path)
    return report


def main() -> None:
    args = parse_args()
    report = publish_duplicate_decisions(
        args.manifest.resolve(),
        args.decisions.resolve(),
        args.report.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
