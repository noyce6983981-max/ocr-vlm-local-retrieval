"""Validate V17 annotations and report agreement and paired bootstrap CIs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.evaluation.human_evaluation import (
    agreement_report,
    grouped_paired_bootstrap,
    validate_annotation_coverage,
)
from scripts.build_v17_human_pool import read_rows


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument(
        "--paired-records",
        type=Path,
        help=(
            "Optional JSONL rows with group_id and paired per-query metrics. "
            "When supplied, both --baseline-field and --contender-field are required."
        ),
    )
    parser.add_argument("--baseline-field")
    parser.add_argument("--contender-field")
    parser.add_argument("--group-field", default="group_id")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/v17/human_study/evaluation_report.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = read_rows(project_path(args.queries))
    judgments = read_rows(project_path(args.judgments))
    coverage = validate_annotation_coverage(queries, judgments)
    report: dict[str, Any] = {
        "annotation_coverage": coverage,
        "agreement": agreement_report(judgments),
    }
    if args.paired_records:
        if not args.baseline_field or not args.contender_field:
            raise ValueError(
                "Paired bootstrap requires --baseline-field and --contender-field"
            )
        records = read_rows(project_path(args.paired_records))
        report["paired_bootstrap"] = grouped_paired_bootstrap(
            records,
            baseline_field=args.baseline_field,
            contender_field=args.contender_field,
            group_field=args.group_field,
            repetitions=args.bootstrap_repetitions,
            seed=args.seed,
        )
    output = project_path(args.output)
    write_json_atomic(output, report)
    print(output)
    if not coverage["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
