"""Materialize frozen V18 retrieval pools for V19.1 machine-draft development."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_v19_downstream_retrieval_pilot import (  # noqa: E402
    read_json,
    run_baseline,
)

DEFAULT_ASSIGNMENTS = (
    ROOT
    / "outputs/evaluation/v19_1/condition_completeness"
    / "development_assignments_machine.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs/evaluation/v19_1/condition_completeness/retrieval"
    / "development_v18_frozen"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--library-dir", type=Path, default=ROOT / "outputs/user_library"
    )
    parser.add_argument(
        "--attribute-policy",
        type=Path,
        default=ROOT / "config/v17_attribute_coverage.json",
    )
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = read_json(args.assignments)
    if payload.get("split") != "v19_1_machine_draft_development_only":
        raise ValueError("retrieval materialization may read V19.1 development only")
    assignments = list(payload.get("assignments", []))
    if len(assignments) != 48:
        raise ValueError("expected exactly 48 V19.1 development assignments")
    for index, assignment in enumerate(assignments, start=1):
        query_id = str(assignment["query_id"])
        print(f"[{index:02d}/{len(assignments)}] {query_id}", flush=True)
        run_baseline(
            dict(assignment),
            output_path=args.output_dir / f"{query_id}_v18_frozen.json",
            library_dir=args.library_dir,
            attribute_policy=args.attribute_policy,
            timeout=args.timeout,
        )
    print(
        json.dumps(
            {
                "status": "complete_development_only",
                "query_count": len(assignments),
                "holdout_opened": False,
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
