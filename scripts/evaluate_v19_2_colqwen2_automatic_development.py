"""Evaluate ColQwen2 on the machine-only V19.2 development contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402
from ocr_vlm_retrieval.runtime.late_interaction import (  # noqa: E402
    exclusive_process_lock,
)
from scripts import evaluate_v19_1_colqwen2_development as shared  # noqa: E402
from scripts.run_v19_downstream_retrieval_pilot import read_json  # noqa: E402

EVALUATION_ROOT = ROOT / "outputs/evaluation/v19_2/automatic_optimization"
DEFAULT_ASSIGNMENTS = EVALUATION_ROOT / "development_assignments_machine.json"
DEFAULT_OUTPUT = EVALUATION_ROOT / "development_colqwen2_machine.json"
SPLIT = "v19_2_automatic_development_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--model", type=Path, default=shared.DEFAULT_MODEL)
    parser.add_argument("--index-dir", type=Path, default=shared.DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--query-batch-size", type=int, default=4)
    parser.add_argument("--passage-batch-size", type=int, default=16)
    parser.add_argument("--stored-top-k", type=int, default=50)
    parser.add_argument("--min-free-gib", type=float, default=5.0)
    parser.add_argument("--throttle-ms", type=int, default=0)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    payload = read_json(args.assignments)
    if payload.get("split") != SPLIT:
        raise ValueError("ColQwen2 may read V19.2 automatic development only")
    if payload.get("human_review_used") is not False:
        raise ValueError("V19.2 automatic development must not use human review")
    shared.DEVELOPMENT_SPLITS = frozenset({SPLIT})
    status = int(shared.run(args))
    result = read_json(args.output)
    result.update(
        {
            "status": "machine_evidence_contract_development_diagnostic_only",
            "eligible_for_promotion": False,
            "human_review_used": False,
            "future_holdout_opened": False,
        }
    )
    write_json_atomic(args.output, result)
    return status


def main() -> int:
    args = parse_args()
    lock_path = ROOT / "outputs/user_library/.v19_2_automatic_gpu_job.lock"
    with exclusive_process_lock(lock_path):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
