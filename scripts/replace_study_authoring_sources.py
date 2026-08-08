"""Replace a pre-freeze authoring batch while retaining an audit trail."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.studies.protocol import (  # noqa: E402
    load_study_protocol,
    protocol_fingerprint,
)
from ocr_vlm_retrieval.studies.query_split import (  # noqa: E402
    replace_authoring_sources,
)
from scripts.prepare_v18_query_authoring import (  # noqa: E402
    load_dominant_colors,
    ocr_excerpt,
    predecessor_exclusions,
    read_jsonl,
    read_ocr_summary,
    resolve_under,
    write_jsonl_atomic,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl_new(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "config/studies/v18.json",
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("outputs/user_library/manifest.jsonl")
    )
    parser.add_argument(
        "--ocr-summary",
        type=Path,
        default=Path("outputs/user_library/ocr/summary.csv"),
    )
    parser.add_argument(
        "--color-index",
        type=Path,
        default=Path("outputs/user_library/color_index/features_v1.npz"),
    )
    parser.add_argument(
        "--v17-proposals",
        type=Path,
        default=Path(
            "data/evaluation/v17/query_proposals/compositional_visual_proposals.jsonl"
        ),
    )
    parser.add_argument(
        "--authoring-dir",
        type=Path,
        default=Path("data/evaluation/v18/query_authoring"),
    )
    parser.add_argument(
        "--discard-ids",
        default=",".join(f"v18_source_{index:03d}" for index in range(1, 11)),
    )
    parser.add_argument("--first-replacement-index", type=int, default=81)
    parser.add_argument("--amendment-id", default="v18_a002")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_root = args.runtime_root.resolve()
    protocol = load_study_protocol(args.protocol.resolve())
    manifest_path = resolve_under(runtime_root, args.manifest)
    ocr_path = resolve_under(runtime_root, args.ocr_summary)
    color_path = resolve_under(runtime_root, args.color_index)
    proposals_path = resolve_under(runtime_root, args.v17_proposals)
    authoring_dir = resolve_under(runtime_root, args.authoring_dir)
    queue_path = authoring_dir / "authoring_queue.jsonl"
    submissions_path = authoring_dir / "authoring_submissions.jsonl"
    archive_dir = authoring_dir / "discarded"
    audit_path = archive_dir / f"{args.amendment_id}_replacement_audit.jsonl"
    queue_archive_path = archive_dir / f"{args.amendment_id}_discarded_queue.jsonl"
    submission_archive_path = (
        archive_dir / f"{args.amendment_id}_discarded_submissions.jsonl"
    )
    receipt_path = archive_dir / f"{args.amendment_id}_replacement_receipt.json"
    outputs = (
        audit_path,
        queue_archive_path,
        submission_archive_path,
        receipt_path,
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("replacement audit artifacts already exist")

    target_ids = {
        value.strip() for value in str(args.discard_ids).split(",") if value.strip()
    }
    manifest = read_jsonl(manifest_path)
    queue = read_jsonl(queue_path)
    submissions = read_jsonl(submissions_path)
    ocr_by_item = read_ocr_summary(ocr_path)
    colors = load_dominant_colors(color_path)
    v17_proposals = read_jsonl(proposals_path)
    exclusions = predecessor_exclusions(manifest, v17_proposals)
    queue_before_sha256 = sha256(queue_path)
    submissions_before_sha256 = sha256(submissions_path)

    replacement_queue, audit = replace_authoring_sources(
        queue,
        manifest,
        ocr_by_item,
        authoring_ids=target_ids,
        excluded_identity_keys=exclusions,
        dominant_colors_by_item=colors,
        seed=f"{protocol.study_id}:{args.amendment_id}",
        first_replacement_index=args.first_replacement_index,
    )
    replacement_ids = {str(row["replacement_authoring_id"]) for row in audit}
    for row in replacement_queue:
        if str(row["authoring_id"]) not in replacement_ids:
            continue
        summary = ocr_by_item.get(str(row["source_item_id"]), {})
        row["ocr_excerpt"] = ocr_excerpt(runtime_root, summary)

    discarded_queue = [
        dict(row) for row in queue if str(row["authoring_id"]) in target_ids
    ]
    discarded_submissions = [
        dict(row)
        for row in submissions
        if str(row.get("authoring_id", "")) in target_ids
    ]
    active_submissions = [
        dict(row)
        for row in submissions
        if str(row.get("authoring_id", "")) not in target_ids
    ]
    recorded_at = datetime.now(UTC).isoformat(timespec="seconds")
    for row in audit:
        row["amendment_id"] = args.amendment_id
        row["reason"] = "onboarding_batch_excluded_before_query_freeze"
        row["recorded_at"] = recorded_at

    write_jsonl_new(queue_archive_path, discarded_queue)
    write_jsonl_new(submission_archive_path, discarded_submissions)
    write_jsonl_new(audit_path, audit)
    write_jsonl_atomic(queue_path, replacement_queue)
    write_jsonl_atomic(submissions_path, active_submissions)
    receipt = {
        "study_id": protocol.study_id,
        "amendment_id": args.amendment_id,
        "protocol_sha256": protocol_fingerprint(protocol),
        "discarded_authoring_ids": sorted(target_ids),
        "replacement_authoring_ids": sorted(replacement_ids),
        "discarded_submission_count": len(discarded_submissions),
        "active_submission_count": len(active_submissions),
        "queue_count": len(replacement_queue),
        "queue_before_sha256": queue_before_sha256,
        "queue_after_sha256": sha256(queue_path),
        "submissions_before_sha256": submissions_before_sha256,
        "submissions_after_sha256": sha256(submissions_path),
        "retrieval_executed": False,
        "query_set_frozen": False,
        "v17_artifacts_modified": False,
        "recorded_at": recorded_at,
    }
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
