"""Generate four syntax-shift rounds from the machine evidence contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_vlm_retrieval.gating.ocr_literals_v19_2 import (  # noqa: E402
    extract_v19_2_literal_groups,
)
from ocr_vlm_retrieval.runtime.cache import write_json_atomic  # noqa: E402
from scripts.run_v19_downstream_retrieval_pilot import read_json  # noqa: E402

EVALUATION_ROOT = ROOT / "outputs/evaluation/v19_2/automatic_optimization"
DEFAULT_ASSIGNMENTS = EVALUATION_ROOT / "development_assignments_machine.json"
DEFAULT_OUTPUT_DIR = EVALUATION_ROOT / "stress"
SPLIT = "v19_2_automatic_development_only"
TEMPLATES = (
    "帮我找同时写有“{first}”和“{second}”的那一页。",
    "请定位一份资料，其中“{first}”与“{second}”必须在同页出现。",
    "页面需要同时出现“{first}”；另一个不可缺少的内容是“{second}”。",
    "我只要同时能看到“{first}”以及“{second}”的资料，缺一项都不要。",
    "请在同一页核对「{first}」和「{second}」，两项都出现在同一页才算命中。",
    "只有【{first}】与【{second}】均可在该页找到时才返回。",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def stress_query(template: str, phrases: list[str]) -> str:
    if len(phrases) != 2:
        raise ValueError("stress query requires exactly two evidence phrases")
    return template.format(first=phrases[0], second=phrases[1])


def main() -> int:
    args = parse_args()
    source = read_json(args.assignments)
    if source.get("split") != SPLIT or source.get("human_review_used") is not False:
        raise ValueError("stress generation requires machine-only V19.2 development")
    assignments = [dict(row) for row in source.get("assignments", [])]
    contracts = {
        str(row["query_id"]): list(row["phrases"])
        for row in source.get("evidence_contract_rows", [])
    }
    if len(assignments) != 48 or assignments[0].get("query_id") not in contracts:
        raise ValueError("expected 48 assignments with evidence contracts")
    source_sha = hashlib.sha256(args.assignments.read_bytes()).hexdigest()
    outputs: list[dict[str, Any]] = []
    for round_index, template in enumerate(TEMPLATES, start=1):
        rows: list[dict[str, Any]] = []
        for assignment in assignments:
            source_query_id = str(assignment["query_id"])
            phrases = contracts[source_query_id]
            query = stress_query(template, phrases)
            parsed = [
                group.label
                for group in extract_v19_2_literal_groups(query)
                if group.source == "explicit_required"
            ]
            if parsed != phrases:
                raise ValueError(f"stress parser mismatch for {source_query_id}")
            rows.append(
                {
                    **assignment,
                    "query_id": f"stress_r{round_index}_{source_query_id}",
                    "source_query_id": source_query_id,
                    "query": query,
                    "stress_round": round_index,
                    "stress_template": template,
                    "review_status": "machine_syntax_stress_contract_passed",
                }
            )
        payload = {
            "schema_version": 1,
            "study_id": "v19-2-automatic-syntax-stress",
            "status": "machine_authored_syntax_stress_development",
            "split": SPLIT,
            "eligible_for_final_claim": False,
            "human_review_used": False,
            "future_holdout_opened": False,
            "stress_round": round_index,
            "source_assignment_sha256": source_sha,
            "query_count": len(rows),
            "assignments": rows,
        }
        output = args.output_dir / f"stress_round_{round_index}_assignments.json"
        write_json_atomic(output, payload)
        outputs.append(
            {
                "round": round_index,
                "path": str(output),
                "query_count": len(rows),
            }
        )
    print(json.dumps({"status": "complete", "rounds": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
