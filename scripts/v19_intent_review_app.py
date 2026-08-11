from __future__ import annotations

import csv
import os
import re
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
QUERY_PATH = ROOT / "data/evaluation/v19/pilot/queries_draft.csv"
PRIVATE_REVIEW_DIR = ROOT / "records/private/v19"
ROUTE_LABELS = {
    "text_evidence": "文字证据",
    "visual_discovery": "视觉内容",
    "visual_metadata": "版面/结构",
    "entity_exact": "精确实体",
    "topic_discovery": "主题发现",
    "mixed": "复合证据",
}
REVIEW_FIELDS = (
    "query_id",
    "query_text",
    "proposed_route",
    "final_route",
    "reviewer_id",
    "reviewed_at_unix",
    "accepted_proposal",
    "notes",
)


def load_queries(path: Path = QUERY_PATH) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("V19 pilot query file is empty")
    query_ids = [row["query_id"] for row in rows]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("V19 pilot query IDs are not unique")
    for row in rows:
        if row["gold_route"] not in ROUTE_LABELS:
            raise ValueError(f"Unsupported route for {row['query_id']}")
    return rows


def load_reviews(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["query_id"]: row for row in rows if row.get("query_id")}


def write_reviews_atomic(path: Path, reviews: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            for query_id in sorted(reviews):
                writer.writerow(
                    {
                        field: reviews[query_id].get(field, "")
                        for field in REVIEW_FIELDS
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_review(
    path: Path,
    *,
    query: dict[str, str],
    final_route: str,
    reviewer_id: str,
    notes: str,
) -> dict[str, dict[str, str]]:
    if final_route not in ROUTE_LABELS:
        raise ValueError(f"Unsupported final route: {final_route}")
    reviews = load_reviews(path)
    existing = reviews.get(query["query_id"])
    if existing and existing["query_text"] != query["query_text"]:
        raise ValueError("Saved query text does not match the current query ID")
    proposed = query["gold_route"]
    reviews[query["query_id"]] = {
        "query_id": query["query_id"],
        "query_text": query["query_text"],
        "proposed_route": proposed,
        "final_route": final_route,
        "reviewer_id": reviewer_id,
        "reviewed_at_unix": f"{time.time():.6f}",
        "accepted_proposal": str(final_route == proposed).lower(),
        "notes": notes.strip(),
    }
    write_reviews_atomic(path, reviews)
    return reviews


def review_path_for(reviewer_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", reviewer_id):
        raise ValueError("审核者ID只能使用1到32位字母、数字、下划线或连字符")
    return PRIVATE_REVIEW_DIR / f"{reviewer_id}_pilot_reviews.csv"


def next_unreviewed_index(
    queries: list[dict[str, str]],
    reviews: dict[str, dict[str, str]],
    current_index: int,
) -> int:
    for offset in range(1, len(queries) + 1):
        index = (current_index + offset) % len(queries)
        if queries[index]["query_id"] not in reviews:
            return index
    return min(current_index + 1, len(queries) - 1)


def render() -> None:
    st.set_page_config(page_title="V19意图标签审核", layout="wide")
    st.title("V19 自然查询意图标签审核")
    st.caption(
        "本页只审核72条开发先导查询的路由标签；保存即写入本地文件，换页或退出不会丢失。"
    )

    queries = load_queries()
    reviewer_id = st.sidebar.text_input("审核者ID", value="reviewer_01").strip()
    try:
        review_path = review_path_for(reviewer_id)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    reviews = load_reviews(review_path)
    completed = len(set(reviews) & {query["query_id"] for query in queries})
    st.sidebar.metric("已完成", f"{completed}/{len(queries)}")
    st.sidebar.progress(completed / len(queries))
    st.sidebar.caption(f"进度文件：{review_path}")

    if "v19_review_index" not in st.session_state:
        st.session_state.v19_review_index = next(
            (
                index
                for index, query in enumerate(queries)
                if query["query_id"] not in reviews
            ),
            0,
        )
    index = max(0, min(int(st.session_state.v19_review_index), len(queries) - 1))
    query = queries[index]
    saved = reviews.get(query["query_id"])

    left, middle, right = st.columns([1, 3, 1])
    with left:
        if st.button("← 上一条", disabled=index == 0, use_container_width=True):
            st.session_state.v19_review_index = index - 1
            st.rerun()
    with middle:
        st.markdown(f"### {index + 1}/{len(queries)} · `{query['query_id']}`")
    with right:
        if st.button(
            "下一条 →",
            disabled=index == len(queries) - 1,
            use_container_width=True,
        ):
            st.session_state.v19_review_index = index + 1
            st.rerun()

    st.info(query["query_text"])
    proposed = query["gold_route"]
    st.write(
        "建议标签：",
        f"**{ROUTE_LABELS[proposed]}** (`{proposed}`)",
    )
    st.write("判断依据：", query["label_basis"])
    st.write("歧义重点：", query["ambiguity_focus"])

    route_options = list(ROUTE_LABELS)
    default_route = saved["final_route"] if saved else proposed
    with st.form(key=f"review_form_{query['query_id']}"):
        final_route = st.selectbox(
            "你的最终标签",
            route_options,
            index=route_options.index(default_route),
            format_func=lambda route: f"{ROUTE_LABELS[route]} · {route}",
        )
        checked = st.checkbox(
            "我已根据查询所需的候选页证据核对该标签",
            value=saved is not None,
        )
        notes = st.text_area("备注（可选）", value=saved["notes"] if saved else "")
        submitted = st.form_submit_button(
            "保存并进入下一条",
            type="primary",
        )
    if submitted and not checked:
        st.error("请先勾选确认框，再保存本条审核结果。")
    elif submitted:
        reviews = save_review(
            review_path,
            query=query,
            final_route=final_route,
            reviewer_id=reviewer_id,
            notes=notes,
        )
        st.session_state.v19_review_index = next_unreviewed_index(
            queries, reviews, index
        )
        st.rerun()

    if saved:
        st.success(
            "本条已保存："
            f"{ROUTE_LABELS[saved['final_route']]} · {saved['final_route']}"
        )


if __name__ == "__main__":
    render()
