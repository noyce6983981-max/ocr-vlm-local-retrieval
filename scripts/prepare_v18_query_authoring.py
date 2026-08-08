"""Prepare the V18 paired-query authoring queue without running retrieval."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.studies.protocol import (  # noqa: E402
    load_study_protocol,
    protocol_fingerprint,
)
from ocr_vlm_retrieval.studies.query_split import (  # noqa: E402
    build_authoring_queue,
    source_identity_keys,
)

DEFAULT_OUTPUT_DIR = Path("data/evaluation/v18/query_authoring")


def resolve_under(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def read_ocr_summary(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            str(row["item_id"]): dict(row)
            for row in csv.DictReader(handle)
            if str(row.get("item_id", "")).strip()
        }


def ocr_excerpt(runtime_root: Path, summary: dict[str, Any]) -> str:
    raw_path = str(summary.get("json_path", "")).strip()
    if not raw_path:
        return ""
    path = resolve_under(runtime_root, Path(raw_path))
    if not path.is_file():
        return ""
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    texts = [" ".join(str(value).split()) for value in payload.get("rec_texts", [])]
    return "\n".join(value for value in texts if value)[:600]


def load_dominant_colors(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        return {}
    import numpy as np

    archive = np.load(path, allow_pickle=False)
    item_ids = [str(value) for value in archive["item_ids"]]
    color_names = [str(value) for value in archive["color_names"]]
    features = archive["features"]
    result: dict[str, list[str]] = {}
    for item_id, feature in zip(item_ids, features, strict=True):
        ranked = sorted(
            enumerate(feature), key=lambda pair: float(pair[1]), reverse=True
        )
        result[item_id] = [
            color_names[index] for index, score in ranked[:3] if float(score) >= 0.05
        ]
    return result


def predecessor_exclusions(
    manifest: list[dict[str, Any]], proposals: list[dict[str, Any]]
) -> set[str]:
    by_item = {str(row["item_id"]): row for row in manifest}
    excluded: set[str] = set()
    for proposal in proposals:
        group_id = str(proposal.get("group_id", "")).strip()
        item_id = str(proposal.get("source_item_id", "")).strip()
        if group_id:
            excluded.add(group_id)
        if item_id:
            excluded.add(f"item:{item_id}")
            source = by_item.get(item_id)
            if source:
                excluded.update(source_identity_keys(source))
    if not excluded:
        raise ValueError("V17 proposal sources are required for leakage exclusion")
    return excluded


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preserve_authoring_ids(
    queue: list[dict[str, Any]], previous: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep source-bound IDs stable across a pre-freeze protocol amendment."""

    if not previous:
        return queue
    previous_by_item = {
        str(row["source_item_id"]): str(row["authoring_id"]) for row in previous
    }
    current_items = {str(row["source_item_id"]) for row in queue}
    if current_items != set(previous_by_item):
        raise ValueError("cannot preserve authoring IDs because the source set changed")
    for row in queue:
        row["authoring_id"] = previous_by_item[str(row["source_item_id"])]
    return sorted(queue, key=lambda row: str(row["authoring_id"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path(os.environ.get("OCR_VLM_RUNTIME_ROOT", PROJECT_ROOT)),
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
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "config/studies/v18.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_root = args.runtime_root.resolve()
    manifest_path = resolve_under(runtime_root, args.manifest)
    ocr_summary_path = resolve_under(runtime_root, args.ocr_summary)
    color_index_path = resolve_under(runtime_root, args.color_index)
    proposals_path = resolve_under(runtime_root, args.v17_proposals)
    output_dir = resolve_under(runtime_root, args.output_dir)
    protocol_path = args.protocol.resolve()

    protocol = load_study_protocol(protocol_path)
    manifest = read_jsonl(manifest_path)
    manifest = [
        row
        for row in manifest
        if resolve_under(runtime_root, Path(str(row.get("source_path", "")))).is_file()
    ]
    ocr_by_item = read_ocr_summary(ocr_summary_path)
    v17_proposals = read_jsonl(proposals_path)
    exclusions = predecessor_exclusions(manifest, v17_proposals)
    colors = load_dominant_colors(color_index_path)
    source_count = protocol.query_design.source_group_count // len(
        protocol.query_design.strata
    )
    existing_queue_path = output_dir / "authoring_queue.jsonl"
    previous_queue = (
        read_jsonl(existing_queue_path) if existing_queue_path.is_file() else []
    )
    queue = build_authoring_queue(
        manifest,
        ocr_by_item,
        excluded_identity_keys=exclusions,
        dominant_colors_by_item=colors,
        seed=protocol.study_id,
        sources_per_stratum=source_count,
        languages=tuple(protocol.query_design.languages),
    )
    queue = preserve_authoring_ids(queue, previous_queue)
    for row in queue:
        summary = ocr_by_item.get(str(row["source_item_id"]), {})
        row["ocr_excerpt"] = ocr_excerpt(runtime_root, summary)
    write_jsonl_atomic(output_dir / "authoring_queue.jsonl", queue)
    receipt = {
        "study_id": protocol.study_id,
        "protocol_sha256": protocol_fingerprint(protocol),
        "manifest_sha256": file_sha256(manifest_path),
        "v17_proposals_sha256": file_sha256(proposals_path),
        "source_count": len(queue),
        "planned_query_count": len(queue) * 2,
        "split_counts": dict(sorted(Counter(row["split"] for row in queue).items())),
        "stratum_counts": dict(
            sorted(Counter(row["stratum"] for row in queue).items())
        ),
        "language_counts": dict(
            sorted(Counter(row["language_target"] for row in queue).items())
        ),
        "excluded_v17_identity_count": len(exclusions),
        "retrieval_executed": False,
        "v17_artifacts_modified": False,
        "human_authoring_complete": False,
    }
    (output_dir / "authoring_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
