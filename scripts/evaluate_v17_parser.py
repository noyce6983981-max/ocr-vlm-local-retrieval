"""Evaluate parser v3 on a calibration-only requirement reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.evaluation.parser_metrics import (
    REFERENCE_FIELDS,
    evaluate_parser_predictions,
)
from ocr_vlm_retrieval.gating.attribute_coverage import (
    decompose_visual_query,
    load_attribute_policy,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parser_predictions(
    references: Iterable[Mapping[str, Any]], policy: Mapping[str, Any]
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for reference in references:
        if reference.get("split") != "calibration":
            raise ValueError("Parser evaluation is calibration-only")
        query_id = str(reference.get("query_id", "")).strip()
        query = str(reference.get("query", "")).strip()
        if not query_id or not query:
            raise ValueError("Every parser reference needs query_id and query")
        plan = decompose_visual_query(query, policy)
        row: dict[str, Any] = {
            "query_id": query_id,
            "query": query,
            "parser_version": plan.parser_version,
            "fingerprint": plan.fingerprint,
        }
        for field in REFERENCE_FIELDS.values():
            row[field] = []
        for requirement in plan.requirements:
            row[REFERENCE_FIELDS[requirement.kind]].append(requirement.value)
        predictions.append(row)
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(
            "data/evaluation/v17/parser/calibration_parser_reference.jsonl"
        ),
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("config/v17_attribute_coverage.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/evaluation/v17/calibration/parser_v3/"
            "parser_requirement_metrics.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_path = PROJECT_ROOT / args.reference
    policy_path = PROJECT_ROOT / args.policy
    references = read_jsonl(reference_path)
    provenance = {str(row.get("provenance", "")) for row in references}
    if provenance != {"codex_assisted_calibration_reference"}:
        raise ValueError("Parser reference provenance is missing or mixed")
    predictions = parser_predictions(references, load_attribute_policy(policy_path))
    metrics = evaluate_parser_predictions(references, predictions)
    report = {
        "status": "calibration_parser_evaluation_complete_holdout_not_read",
        "human_gold": False,
        "reference_provenance": "codex_assisted_calibration_reference",
        "reference_limitations": (
            "The calibration reference was reviewed with AI assistance and must "
            "not be described as an independent human gold set."
        ),
        "reference_sha256": file_sha256(reference_path),
        "policy_path": args.policy.as_posix(),
        "policy_sha256": file_sha256(policy_path),
        "metrics": metrics,
        "predictions": predictions,
    }
    output = PROJECT_ROOT / args.output
    write_json_atomic(output, report)
    print(output)


if __name__ == "__main__":
    main()
