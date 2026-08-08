"""Publish a report-only, de-identified analysis of sealed V17 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ocr_vlm_retrieval.evaluation.protocol_lock import file_sha256
from ocr_vlm_retrieval.evaluation.public_holdout import (
    analyze_public_paired_records,
    build_public_paired_records,
    summarize_verification_runtime,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    )
    path.write_text(content + "\n", encoding="utf-8", newline="\n")


def build_public_audit(
    *,
    final_report_path: Path,
    verification_path: Path,
    v16_ranking_path: Path,
    judgments_path: Path,
    paired_output: Path,
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    final_report = read_json(final_report_path)
    verification = read_json(verification_path)
    public_rows = build_public_paired_records(
        final_report=final_report,
        v16_rankings=read_jsonl(v16_ranking_path),
        judgments=read_jsonl(judgments_path),
    )
    write_jsonl(paired_output, public_rows)
    return {
        "schema_version": 1,
        "status": "report_only_analysis_of_sealed_v17_holdout",
        "study_name": "V17-Compositional-80 Pilot",
        "method_changed": False,
        "model_rerun": False,
        "holdout_rerun": False,
        "analysis": analyze_public_paired_records(
            public_rows,
            bootstrap_repetitions=bootstrap_repetitions,
            seed=seed,
        ),
        "runtime": summarize_verification_runtime(verification),
        "source_artifact_sha256": {
            "final_private_report": file_sha256(final_report_path),
            "label_blind_verification": file_sha256(verification_path),
            "v16_frozen_ranking": file_sha256(v16_ranking_path),
            "final_human_adjudication": file_sha256(judgments_path),
        },
        "public_artifact": {
            "path": "data/evaluation/v17/public_holdout_paired_records.jsonl",
            "sha256": file_sha256(paired_output),
            "contains_query_text": False,
            "contains_candidate_ids": False,
            "contains_source_metadata": False,
            "contains_reviewer_identity": False,
        },
        "limitations": [
            "All metrics are conditional on the frozen 20-candidate pools.",
            (
                "The eight no-relevant-in-pool queries do not establish "
                "corpus-level answerability."
            ),
            (
                "Runtime is verifier-only descriptive timing, not production "
                "search latency."
            ),
            (
                "Fine-grained parser, relation, and binding error types need "
                "separate human diagnosis."
            ),
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-report", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--v16-ranking", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--paired-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_public_audit(
        final_report_path=args.final_report,
        verification_path=args.verification,
        v16_ranking_path=args.v16_ranking,
        judgments_path=args.judgments,
        paired_output=args.paired_output,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    write_json(args.audit_output, audit)
    print(args.paired_output)
    print(args.audit_output)


if __name__ == "__main__":
    main()
