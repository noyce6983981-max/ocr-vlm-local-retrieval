"""Export only frozen V18 calibration queries for downstream retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.studies.protocol import (  # noqa: E402
    load_study_protocol,
    protocol_fingerprint,
)
from ocr_vlm_retrieval.studies.query_split import (  # noqa: E402
    query_set_fingerprint,
    validate_frozen_queries,
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl_new(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def isolate_calibration_rows(
    rows: Sequence[Mapping[str, Any]],
    freeze_receipt: Mapping[str, Any],
    *,
    study_id: str,
    protocol_sha256: str,
    expected_total: int,
    expected_calibration: int,
    expected_holdout: int,
) -> list[dict[str, Any]]:
    """Validate the complete freeze while returning calibration rows only."""

    validate_frozen_queries(rows, expected_count=expected_total)
    if str(freeze_receipt.get("study_id", "")) != study_id:
        raise ValueError("freeze receipt study ID does not match the protocol")
    if str(freeze_receipt.get("protocol_sha256", "")) != protocol_sha256:
        raise ValueError("freeze receipt protocol hash does not match")
    if int(freeze_receipt.get("query_count", 0)) != expected_total:
        raise ValueError("freeze receipt query count does not match")
    if str(freeze_receipt.get("query_set_sha256", "")) != query_set_fingerprint(
        rows
    ):
        raise ValueError("frozen query-set fingerprint does not match its receipt")
    if freeze_receipt.get("retrieval_executed") is not False:
        raise ValueError("V18 retrieval was already marked as executed")
    if freeze_receipt.get("holdout_results_opened") is not False:
        raise ValueError("V18 holdout was already marked as opened")
    if freeze_receipt.get("v17_artifacts_modified") is not False:
        raise ValueError("V17 read-only invariant is not satisfied")

    split_counts = Counter(str(row.get("split", "")) for row in rows)
    expected_splits = Counter(
        {"calibration": expected_calibration, "holdout": expected_holdout}
    )
    if split_counts != expected_splits:
        raise ValueError("frozen V18 split counts do not match the protocol")
    if any(row.get("review_status") != "human_query_approved" for row in rows):
        raise ValueError("every V18 query must be human_query_approved")
    calibration = [dict(row) for row in rows if row.get("split") == "calibration"]
    calibration.sort(key=lambda row: str(row["query_id"]))
    return calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=PROJECT_ROOT / "config/studies/v18.json"
    )
    parser.add_argument(
        "--methods",
        type=Path,
        default=PROJECT_ROOT / "config/studies/v18_methods.json",
    )
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    methods_path = args.methods.resolve()
    queries_path = args.queries.resolve()
    freeze_receipt_path = args.freeze_receipt.resolve()
    output_path = args.output.resolve()
    receipt_path = args.receipt.resolve()

    protocol = load_study_protocol(protocol_path)
    methods = read_json(methods_path)
    if methods.get("study_id") != protocol.study_id:
        raise ValueError("V18 method specification targets a different study")
    if methods.get("scope") != "calibration_only":
        raise ValueError("V18 method specification is not calibration-only")
    holdout_access = methods.get("holdout_access")
    if not isinstance(holdout_access, dict) or holdout_access.get(
        "permitted_during_calibration"
    ) is not False:
        raise ValueError("V18 method specification does not seal the holdout")

    frozen = read_jsonl(queries_path)
    freeze_receipt = read_json(freeze_receipt_path)
    calibration = isolate_calibration_rows(
        frozen,
        freeze_receipt,
        study_id=protocol.study_id,
        protocol_sha256=protocol_fingerprint(protocol),
        expected_total=protocol.query_design.total_queries,
        expected_calibration=protocol.query_design.splits["calibration"],
        expected_holdout=protocol.query_design.splits["holdout"],
    )
    scope = {
        "status": "validated_no_retrieval" if args.dry_run else "prepared",
        "study_id": protocol.study_id,
        "exported_split": "calibration",
        "calibration_query_count": len(calibration),
        "holdout_query_count_not_exported": protocol.query_design.splits["holdout"],
        "frozen_query_set_sha256": freeze_receipt["query_set_sha256"],
        "calibration_query_set_sha256": query_set_fingerprint(calibration),
        "methods_sha256": sha256_file(methods_path),
        "retrieval_executed": False,
        "holdout_results_opened": False,
        "v17_artifacts_modified": False,
    }
    if args.dry_run:
        print(json.dumps(scope, ensure_ascii=False, sort_keys=True))
        return
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("calibration scope output or receipt already exists")
    write_jsonl_new(output_path, calibration)
    write_json_new(
        receipt_path,
        {
            **scope,
            "calibration_query_file_sha256": sha256_file(output_path),
            "freeze_receipt_sha256": sha256_file(freeze_receipt_path),
            "protocol_sha256": protocol_fingerprint(protocol),
        },
    )
    print(json.dumps(scope, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
