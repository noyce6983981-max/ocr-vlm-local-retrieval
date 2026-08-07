from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"blind_id", "item_id", "split", "assistant_category", "assistant_confidence"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"invalid review label file: {path}")
    return rows


def index_unique(rows: list[dict[str, str]], name: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        item_id = row["item_id"]
        if item_id in indexed:
            raise ValueError(f"duplicate item_id in {name}: {item_id}")
        indexed[item_id] = row
    return indexed


def agreement_stats(
    round4: dict[str, dict[str, str]],
    round5: dict[str, dict[str, str]],
    item_ids: list[str],
) -> dict[str, float | int]:
    pairs = [
        (round4[item_id]["assistant_category"], round5[item_id]["assistant_category"])
        for item_id in item_ids
    ]
    observed = sum(left == right for left, right in pairs) / len(pairs)
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    labels = set(left_counts) | set(right_counts)
    expected = sum(
        (left_counts[label] / len(pairs)) * (right_counts[label] / len(pairs))
        for label in labels
    )
    kappa = (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0
    return {
        "count": len(pairs),
        "agreement_count": sum(left == right for left, right in pairs),
        "disagreement_count": sum(left != right for left, right in pairs),
        "agreement": round(observed, 6),
        "cohen_kappa": round(kappa, 6),
    }


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round4-labels", type=Path, required=True)
    parser.add_argument("--round5-labels", type=Path, required=True)
    parser.add_argument("--round4-report", type=Path, required=True)
    parser.add_argument("--round5-report", type=Path, required=True)
    parser.add_argument("--disagreements", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    rows4 = read_rows(args.round4_labels)
    rows5 = read_rows(args.round5_labels)
    if len(rows4) != 600 or len(rows5) != 600:
        raise ValueError(f"expected 600 rows per round, got {len(rows4)} and {len(rows5)}")

    by4 = index_unique(rows4, "round4")
    by5 = index_unique(rows5, "round5")
    if set(by4) != set(by5):
        raise ValueError("rounds do not contain the same item_id cohort")

    item_ids = sorted(by4)
    if any(by4[item_id]["split"] != by5[item_id]["split"] for item_id in item_ids):
        raise ValueError("an item changed train/validation/test split between rounds")

    disagreements: list[dict[str, str]] = []
    for item_id in item_ids:
        row4 = by4[item_id]
        row5 = by5[item_id]
        if row4["assistant_category"] == row5["assistant_category"]:
            continue
        disagreements.append(
            {
                "item_id": item_id,
                "split": row4["split"],
                "round4_blind_id": row4["blind_id"],
                "round5_blind_id": row5["blind_id"],
                "round4_category": row4["assistant_category"],
                "round5_category": row5["assistant_category"],
                "round4_confidence": row4["assistant_confidence"],
                "round5_confidence": row5["assistant_confidence"],
                "round4_notes": row4.get("assistant_notes", ""),
                "round5_notes": row5.get("assistant_notes", ""),
                "human_category": "",
                "human_notes": "",
                "status": "pending_user_confirmation",
            }
        )

    args.disagreements.parent.mkdir(parents=True, exist_ok=True)
    with args.disagreements.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(disagreements[0]) if disagreements else [
            "item_id", "split", "round4_blind_id", "round5_blind_id",
            "round4_category", "round5_category", "round4_confidence",
            "round5_confidence", "round4_notes", "round5_notes",
            "human_category", "human_notes", "status",
        ])
        writer.writeheader()
        writer.writerows(disagreements)

    report4 = load_metrics(args.round4_report)
    report5 = load_metrics(args.round5_report)
    metrics4 = report4["heldout_metrics"]
    metrics5 = report5["heldout_metrics"]
    selected_round = 4 if metrics4["validation"]["macro_f1"] >= metrics5["validation"]["macro_f1"] else 5

    summary = {
        "protocol": {
            "cohort_size": 600,
            "train": 410,
            "validation": 95,
            "test": 95,
            "selection_rule": "select by validation macro_f1; report test once",
            "test_labels_used_for_fitting": False,
        },
        "overall": agreement_stats(by4, by5, item_ids),
        "by_split": {
            split: agreement_stats(
                by4, by5, [item_id for item_id in item_ids if by4[item_id]["split"] == split]
            )
            for split in ("train", "validation", "test")
        },
        "round4_label_counts": dict(sorted(Counter(row["assistant_category"] for row in rows4).items())),
        "round5_label_counts": dict(sorted(Counter(row["assistant_category"] for row in rows5).items())),
        "disagreement_pairs": [
            {"round4": left, "round5": right, "count": count}
            for (left, right), count in Counter(
                (row["round4_category"], row["round5_category"]) for row in disagreements
            ).most_common()
        ],
        "model_metrics": {
            "round4": metrics4,
            "round5": metrics5,
        },
        "selected_candidate_round": selected_round,
        "selection_reason": "higher validation macro_f1",
        "deployment_status": "not_deployed_pending_user_confirmation",
        "disagreement_csv": str(args.disagreements),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
