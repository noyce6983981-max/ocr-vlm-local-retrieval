"""Small local UI for reviewing V17 query wording before retrieval runs."""

from __future__ import annotations

import csv
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = (
    PROJECT_ROOT / "data/evaluation/v17/query_proposals/query_review_queue.csv"
)
REVIEWS_PATH = (
    PROJECT_ROOT / "data/evaluation/v17/query_proposals/query_reviews.csv"
)
REVIEW_FIELDS = (
    "query_id",
    "review_action",
    "reviewed_query",
    "reviewer_id",
    "human_notes",
    "reviewed_at",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_reviews(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in REVIEW_FIELDS})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_review(path: Path, review: dict[str, Any]) -> None:
    existing = {row["query_id"]: row for row in read_csv(path)}
    existing[str(review["query_id"])] = review
    write_reviews(path, [existing[key] for key in sorted(existing)])


def main() -> None:
    st.set_page_config(page_title="V17 查询审核", layout="wide")
    st.title("V17 查询措辞审核")
    st.caption(
        "这里只审核问题是否自然、是否仍是复合视觉查询；不会运行检索，"
        "也不会显示候选、分数或预期相关页面。"
    )
    queue = read_csv(QUEUE_PATH)
    if not queue:
        st.error("尚未生成 query_review_queue.csv。")
        return
    reviews = {row["query_id"]: row for row in read_csv(REVIEWS_PATH)}
    reviewer_id = st.sidebar.text_input("审核者 ID")
    pending = [row for row in queue if row["query_id"] not in reviews]
    st.sidebar.metric("总查询", len(queue))
    st.sidebar.metric("已审核", len(reviews))
    st.sidebar.metric("待审核", len(pending))
    show_completed = st.sidebar.checkbox("显示已审核查询", value=False)
    visible = queue if show_completed else pending
    if not visible:
        st.success("所有查询措辞均已审核，可以运行冻结命令。")
        return
    labels = {
        row["query_id"]: (
            f"{row['query_id']} · {row['split']} · "
            f"{row['transformation_type']}"
        )
        for row in visible
    }
    selected_id = st.selectbox(
        "选择查询",
        [row["query_id"] for row in visible],
        format_func=lambda value: labels[value],
    )
    row = next(source for source in visible if source["query_id"] == selected_id)
    current = reviews.get(selected_id, {})
    st.code(row["query"], language=None)
    left, right = st.columns(2)
    left.write(f"来源类型：`{row['query_origin']}`")
    left.write(f"计划用途：`{row['expected_answerability']}`")
    right.write(f"数据划分：`{row['split']}`")
    right.write(f"变换类型：`{row['transformation_type']}`")
    with st.form(f"query_review_{selected_id}"):
        action = st.radio(
            "审核结论",
            ("approve", "rewrite", "reject"),
            index=("approve", "rewrite", "reject").index(
                current.get("review_action", "approve")
            ),
            horizontal=True,
        )
        reviewed_query = st.text_area(
            "确认或改写后的查询",
            value=current.get("reviewed_query") or row["query"],
        )
        notes = st.text_area("备注", value=current.get("human_notes", ""))
        submitted = st.form_submit_button("保存审核", type="primary")
    if submitted:
        if not reviewer_id.strip():
            st.error("请先填写审核者 ID。")
            return
        if action == "rewrite" and not reviewed_query.strip():
            st.error("选择 rewrite 时必须填写改写后的查询。")
            return
        save_review(
            REVIEWS_PATH,
            {
                "query_id": selected_id,
                "review_action": action,
                "reviewed_query": reviewed_query.strip(),
                "reviewer_id": reviewer_id.strip(),
                "human_notes": notes.strip(),
                "reviewed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        )
        st.success("已保存。")
        st.rerun()


if __name__ == "__main__":
    main()
