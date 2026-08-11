from __future__ import annotations

import csv
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    ROOT / "data/evaluation/v19/formal/second_review_manifest.json"
)
PRIVATE_DIR = ROOT / "records/private/v19/formal"
ROUTE_LABELS = {
    "text_evidence": "文字证据",
    "visual_discovery": "视觉场景",
    "visual_metadata": "页面版式",
    "entity_exact": "精确实体",
    "topic_discovery": "主题发现",
    "mixed": "复合证据",
}
FIELDS = (
    "family_id",
    "family_snapshot_sha256",
    "split",
    "final_route",
    "reviewer_id",
    "reviewed_at_unix",
    "notes",
)


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    payload = cast(
        dict[str, Any], json.loads(path.read_text(encoding="utf-8"))
    )
    if payload.get("primary_labels_exposed") is not False:
        raise ValueError("second review manifest must not expose primary labels")
    families = payload.get("families")
    if not isinstance(families, list) or len(families) != 18:
        raise ValueError("second review manifest must contain 18 families")
    if any("proposed_route" in row or "final_route" in row for row in families):
        raise ValueError("second review family rows must be label blind")
    return payload


def review_path_for(reviewer_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", reviewer_id):
        raise ValueError("审核者ID只能包含字母、数字、下划线或连字符")
    if reviewer_id == "reviewer_01":
        raise ValueError("第二审核必须使用不同于主审核者的ID")
    return PRIVATE_DIR / f"{reviewer_id}_second_family_reviews.csv"


def load_reviews(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    family_ids = [row["family_id"] for row in rows]
    if len(family_ids) != len(set(family_ids)):
        raise ValueError("second review contains duplicate family IDs")
    return {row["family_id"]: row for row in rows}


def save_review(
    path: Path,
    reviews: dict[str, dict[str, str]],
    *,
    family: dict[str, Any],
    final_route: str,
    reviewer_id: str,
    notes: str,
) -> dict[str, dict[str, str]]:
    if final_route not in ROUTE_LABELS:
        raise ValueError(f"unsupported route: {final_route}")
    family_id = str(family["family_id"])
    reviews = dict(reviews)
    reviews[family_id] = {
        "family_id": family_id,
        "family_snapshot_sha256": str(family["family_snapshot_sha256"]),
        "split": str(family["split"]),
        "final_route": final_route,
        "reviewer_id": reviewer_id,
        "reviewed_at_unix": f"{time.time():.6f}",
        "notes": notes.strip(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for saved_id in sorted(reviews):
                writer.writerow(reviews[saved_id])
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return reviews


def render() -> None:
    st.set_page_config(page_title="V19独立第二审核", layout="wide")
    st.title("V19 独立第二审核（18个家族）")
    st.caption(
        "本页不展示主审核标签或建议标签。请独立判断每个查询家族需要的候选页证据类型。"
    )
    manifest = load_manifest()
    families = list(manifest["families"])
    reviewer_id = st.sidebar.text_input("第二审核者ID", value="reviewer_02").strip()
    try:
        review_path = review_path_for(reviewer_id)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    reviews = load_reviews(review_path)
    completed = len(set(reviews) & {str(row["family_id"]) for row in families})
    st.sidebar.metric("已完成", f"{completed}/18")
    st.sidebar.progress(completed / 18)
    st.sidebar.markdown(
        "**标签说明**\n\n"
        "- 文字证据：读取原文、数字或表格\n"
        "- 视觉场景：对象、颜色、动作或真实空间关系\n"
        "- 页面版式：表单、图表、界面或页面排列\n"
        "- 精确实体：短人名或明确实体的精确查找\n"
        "- 主题发现：浏览宽泛主题的一组资料\n"
        "- 复合证据：至少两类必要证据同时满足"
    )
    if "v19_second_index" not in st.session_state:
        st.session_state.v19_second_index = next(
            (
                index
                for index, family in enumerate(families)
                if str(family["family_id"]) not in reviews
            ),
            0,
        )
    index = max(0, min(int(st.session_state.v19_second_index), 17))
    family = families[index]
    family_id = str(family["family_id"])
    saved = reviews.get(family_id)

    left, middle, right = st.columns([1, 3, 1])
    with left:
        if st.button("← 上一家族", disabled=index == 0, use_container_width=True):
            st.session_state.v19_second_index = index - 1
            st.rerun()
    with middle:
        st.markdown(f"### {index + 1}/18 · `{family_id}`")
    with right:
        if st.button("下一家族 →", disabled=index == 17, use_container_width=True):
            st.session_state.v19_second_index = index + 1
            st.rerun()

    for query_index, query in enumerate(family["queries"], start=1):
        st.info(f"{query_index}. {query}")

    route_key = f"v19_second_route_{family_id}"
    notes_key = f"v19_second_notes_{family_id}"
    if route_key not in st.session_state:
        st.session_state[route_key] = saved["final_route"] if saved else None
    if notes_key not in st.session_state:
        st.session_state[notes_key] = saved["notes"] if saved else ""

    def persist() -> None:
        selected = st.session_state[route_key]
        if selected is None:
            return
        save_review(
            review_path,
            load_reviews(review_path),
            family=family,
            final_route=str(selected),
            reviewer_id=reviewer_id,
            notes=str(st.session_state[notes_key]),
        )

    st.selectbox(
        "独立判断标签",
        list(ROUTE_LABELS),
        index=None,
        placeholder="请选择一个标签",
        format_func=lambda route: f"{ROUTE_LABELS[route]} · {route}",
        key=route_key,
        on_change=persist,
    )
    st.text_area("备注（可选）", key=notes_key, on_change=persist)
    if saved:
        st.success(f"已自动保存：{ROUTE_LABELS[saved['final_route']]}")
    else:
        st.warning("本家族尚未选择标签。")


if __name__ == "__main__":
    render()
