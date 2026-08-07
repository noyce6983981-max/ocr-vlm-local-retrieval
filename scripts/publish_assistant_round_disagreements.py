from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.assistant_category_proposals import write_assistant_proposals
from scripts.taxonomy import CATEGORY_LABELS


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disagreements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.disagreements)
    if not rows:
        raise ValueError("disagreement queue is empty")
    if len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError("disagreement queue contains duplicate item_id values")

    if args.output.is_file():
        args.backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.output, args.backup)

    proposed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    proposals: dict[str, dict[str, str]] = {}
    for row in rows:
        round4 = row["round4_category"]
        round5 = row["round5_category"]
        if round4 not in CATEGORY_LABELS or round5 not in CATEGORY_LABELS:
            raise ValueError(f"unsupported category for {row['item_id']}")
        proposals[row["item_id"]] = {
            "item_id": row["item_id"],
            "proposed_category": round4,
            "assistant_notes": (
                "两次完整独立盲审出现分歧："
                f"第4轮为“{CATEGORY_LABELS[round4]}”，"
                f"第5轮为“{CATEGORY_LABELS[round5]}”。"
                "当前预选第4轮结果（其候选模型验证集Macro-F1更高），"
                "请只依据图片内容作最终裁决。"
            ),
            "proposal_round": "4/5分歧复核",
            "model_current_category": "",
            "model_predicted_category": "",
            "model_confidence": "",
            "status": "pending",
            "proposed_at": proposed_at,
            "reviewed_at": "",
        }

    write_assistant_proposals(args.output, proposals)
    print(
        f"published={len(proposals)} output={args.output} "
        f"backup={args.backup}"
    )


if __name__ == "__main__":
    main()
