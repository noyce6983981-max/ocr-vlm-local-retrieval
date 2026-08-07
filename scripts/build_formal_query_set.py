"""Build the formal query set from completed human review decisions."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.query_review import read_reviews, split_item_ids  # noqa: E402


DEFAULT_QUEUE = (
    PROJECT_ROOT
    / "data/evaluation/"
    "public_dataset_1500_retrieval_query_queue_100.csv"
)
DEFAULT_REVIEWS = (
    PROJECT_ROOT
    / "data/evaluation/"
    "public_dataset_1500_retrieval_query_human_reviews.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/evaluation/"
    "public_dataset_1500_retrieval_queries_formal.csv"
)
PRIVACY_PATTERN = re.compile(
    r"(住址|身份证|公民身份号码|手机号|电话|"
    r"\d{6,}|[\w.+-]+@[\w.-]+)"
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_formal_rows(
    queue: list[dict[str, str]],
    reviews: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    missing = [
        row["query_id"]
        for row in queue
        if row["query_id"] not in reviews
    ]
    if missing:
        raise ValueError(
            f"还有 {len(missing)} 条查询未人工审核："
            + "、".join(missing[:5])
        )

    formal_rows: list[dict[str, str]] = []
    unsafe_privacy_ids: list[str] = []
    for source in queue:
        review = reviews[source["query_id"]]
        if review["decision"] == "excluded":
            continue
        review_reasons = set(
            source.get("diagnostic_review_reasons", "").split(",")
        )
        if (
            "privacy" in review_reasons
            or PRIVACY_PATTERN.search(review["reviewed_query"])
        ):
            unsafe_privacy_ids.append(source["query_id"])
        relevant_ids = split_item_ids(review["relevant_item_ids"])
        is_no_answer = review["decision"] == "no_answer"
        if not review["reviewed_query"].strip():
            raise ValueError(
                f"{source['query_id']} 已通过，但查询为空。"
            )
        if not is_no_answer and not relevant_ids:
            raise ValueError(
                f"{source['query_id']} 是有答案查询，但正确图片为空。"
            )
        if is_no_answer and relevant_ids:
            raise ValueError(
                f"{source['query_id']} 标记为无答案，但仍填写了正确图片。"
            )
        expected_item_id = ""
        if relevant_ids:
            expected_item_id = (
                source["expected_item_id"]
                if source["expected_item_id"] in relevant_ids
                else relevant_ids[0]
            )
        formal_rows.append(
            {
                "query_id": source["query_id"],
                "query": review["reviewed_query"].strip(),
                "expected_item_id": expected_item_id,
                "relevant_item_ids": ";".join(relevant_ids),
                "query_type": source["query_type"],
                "review_status": "已确认",
                "split": source["split"],
                "category": source["category"],
                "review_decision": review["decision"],
                "human_notes": review["human_notes"],
                "is_no_answer": str(is_no_answer),
            }
        )
    if unsafe_privacy_ids:
        raise ValueError(
            "以下查询仍涉及隐私风险，必须排除："
            + "、".join(unsafe_privacy_ids)
        )
    if not formal_rows:
        raise ValueError("所有查询都被排除，无法建立正式评测集。")
    query_to_ids: dict[str, list[str]] = {}
    for row in formal_rows:
        query_to_ids.setdefault(row["query"], []).append(row["query_id"])
    duplicate_groups = [
        ids for ids in query_to_ids.values() if len(ids) > 1
    ]
    if duplicate_groups:
        raise ValueError(
            "存在重复查询："
            + "；".join("、".join(ids) for ids in duplicate_groups)
        )
    return formal_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queue = read_rows(args.queue)
    reviews = read_reviews(args.reviews)
    formal_rows = build_formal_rows(queue, reviews)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(formal_rows[0])
        )
        writer.writeheader()
        writer.writerows(formal_rows)
    print(
        f"formal_queries={len(formal_rows)} "
        f"excluded={len(queue) - len(formal_rows)}"
    )
    print(args.output)


if __name__ == "__main__":
    main()
