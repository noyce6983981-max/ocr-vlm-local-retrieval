from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.assistant_category_proposals import (
    read_assistant_proposals,
    write_assistant_proposals,
)
from scripts.quality_review import read_quality_reviews


def read_manifest(path: Path) -> dict[str, dict]:
    return {
        row["item_id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()

    proposals = read_assistant_proposals(args.proposals)
    reviews = read_quality_reviews(args.reviews)
    manifest = read_manifest(args.manifest)
    args.backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.proposals, args.backup)

    counts = {"confirmed": 0, "revised": 0, "rejected": 0, "pending": 0}
    for item_id, proposal in proposals.items():
        if proposal.get("status") != "pending":
            counts[proposal["status"]] += 1
            continue
        review = reviews.get(item_id)
        item = manifest.get(item_id)
        if review is None or item is None:
            counts["pending"] += 1
            continue
        decision = review.get("decision")
        if decision == "accepted":
            human_category = str(item.get("category", ""))
        elif decision == "reclassified":
            human_category = review.get("revised_category", "")
        else:
            human_category = ""
        if not human_category:
            proposal["status"] = "rejected"
        elif human_category == proposal["proposed_category"]:
            proposal["status"] = "confirmed"
        else:
            proposal["status"] = "revised"
        proposal["reviewed_at"] = review.get("reviewed_at", "")
        counts[proposal["status"]] += 1

    write_assistant_proposals(args.proposals, proposals)
    print(json.dumps(counts, ensure_ascii=False))
    if counts["pending"]:
        raise SystemExit(f"{counts['pending']} proposals still lack human review")


if __name__ == "__main__":
    main()
