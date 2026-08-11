from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
FAMILY_PATH = ROOT / "data/evaluation/v19/formal/query_families_draft.jsonl"
PRIVATE_REVIEW_DIR = ROOT / "records/private/v19/formal"
ROUTE_LABELS = {
    "text_evidence": "文字证据",
    "visual_discovery": "视觉场景",
    "visual_metadata": "页面版式",
    "entity_exact": "精确实体",
    "topic_discovery": "主题发现",
    "mixed": "复合证据",
}
REVIEW_FIELDS = (
    "family_id",
    "family_snapshot_sha256",
    "split",
    "proposed_route",
    "final_route",
    "reviewer_id",
    "reviewed_at_unix",
    "accepted_proposal",
    "notes",
)


def load_families(path: Path = FAMILY_PATH) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("family row must be an object")
                rows.append(payload)
    if len(rows) != 60:
        raise ValueError(f"expected 60 formal families, got {len(rows)}")
    family_ids = [str(row["family_id"]) for row in rows]
    if len(family_ids) != len(set(family_ids)):
        raise ValueError("formal family IDs must be unique")
    for row in rows:
        if len(row.get("queries", [])) != 4:
            raise ValueError(f"family {row['family_id']} must contain four queries")
    return rows


def family_snapshot_sha256(family: dict[str, Any]) -> str:
    snapshot = {
        "family_id": family["family_id"],
        "split": family["split"],
        "proposed_route": family["proposed_route"],
        "query_ids": family["query_ids"],
        "queries": family["queries"],
    }
    material = json.dumps(
        snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def review_path_for(reviewer_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", reviewer_id):
        raise ValueError("审核者ID只能包含字母、数字、下划线或连字符")
    return PRIVATE_REVIEW_DIR / f"{reviewer_id}_family_reviews.csv"


def load_reviews(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    family_ids = [row["family_id"] for row in rows]
    if len(family_ids) != len(set(family_ids)):
        raise ValueError("review file contains duplicate family IDs")
    return {row["family_id"]: row for row in rows}


def write_reviews_atomic(
    path: Path, reviews: dict[str, dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            for family_id in sorted(reviews):
                writer.writerow(
                    {
                        field: reviews[family_id].get(field, "")
                        for field in REVIEW_FIELDS
                    }
                )
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def save_review(
    path: Path,
    *,
    family: dict[str, Any],
    final_route: str,
    reviewer_id: str,
    notes: str,
) -> dict[str, dict[str, str]]:
    if final_route not in ROUTE_LABELS:
        raise ValueError(f"unsupported route: {final_route}")
    reviews = load_reviews(path)
    snapshot = family_snapshot_sha256(family)
    existing = reviews.get(str(family["family_id"]))
    if existing and existing["family_snapshot_sha256"] != snapshot:
        raise ValueError("family content changed after an earlier review")
    proposed = str(family["proposed_route"])
    family_id = str(family["family_id"])
    reviews[family_id] = {
        "family_id": family_id,
        "family_snapshot_sha256": snapshot,
        "split": str(family["split"]),
        "proposed_route": proposed,
        "final_route": final_route,
        "reviewer_id": reviewer_id,
        "reviewed_at_unix": f"{time.time():.6f}",
        "accepted_proposal": str(final_route == proposed).lower(),
        "notes": notes.strip(),
    }
    write_reviews_atomic(path, reviews)
    return reviews


def next_unreviewed_index(
    families: list[dict[str, Any]],
    reviews: dict[str, dict[str, str]],
    current_index: int,
) -> int:
    for offset in range(1, len(families) + 1):
        index = (current_index + offset) % len(families)
        if str(families[index]["family_id"]) not in reviews:
            return index
    return min(current_index + 1, len(families) - 1)


def render() -> None:
    st.set_page_config(page_title="V19正式查询家族审核", layout="wide")
    st.title("V19 正式查询家族审核")
    st.caption(
        "每个家族含4条同义改写；建议标签默认视为通过。"
        "你只需浏览，发现错误时修改下拉框，修改会立即自动保存。"
    )

    families = load_families()
    reviewer_id = st.sidebar.text_input("审核者ID", value="reviewer_01").strip()
    try:
        review_path = review_path_for(reviewer_id)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    reviews = load_reviews(review_path)
    family_by_id = {str(row["family_id"]): row for row in families}
    corrected = sum(
        1
        for family_id, review in reviews.items()
        if family_id in family_by_id
        and review["final_route"] != str(family_by_id[family_id]["proposed_route"])
    )
    st.sidebar.metric("已自动保存修正", corrected)
    st.sidebar.caption("建议标签默认通过；未修改的家族无需逐条保存。")
    st.sidebar.caption(f"修正记录：{review_path}")
    st.sidebar.markdown(
        "**判断提示**\n\n"
        "- 文字证据：必须读取候选中的原文、数字或表格\n"
        "- 视觉场景：对象、颜色、动作或真实空间关系\n"
        "- 页面版式：表单、图表、界面或页面排列\n"
        "- 精确实体：短人名或明确实体的精确查找\n"
        "- 主题发现：浏览宽泛主题的一组资料\n"
        "- 复合证据：至少两类必要证据必须同时满足"
    )

    if "v19_formal_family_index" not in st.session_state:
        st.session_state.v19_formal_family_index = next(
            (
                index
                for index, family in enumerate(families)
                if str(family["family_id"]) not in reviews
            ),
            0,
        )
    index = max(
        0,
        min(int(st.session_state.v19_formal_family_index), len(families) - 1),
    )
    st.sidebar.metric("当前浏览位置", f"{index + 1}/{len(families)}")
    st.sidebar.progress((index + 1) / len(families))
    family = families[index]
    family_id = str(family["family_id"])
    saved = reviews.get(family_id)

    left, middle, right = st.columns([1, 3, 1])
    with left:
        if st.button("← 上一家族", disabled=index == 0, use_container_width=True):
            st.session_state.v19_formal_family_index = index - 1
            st.rerun()
    with middle:
        st.markdown(
            f"### {index + 1}/{len(families)} · `{family_id}` · "
            f"{family['split']}"
        )
    with right:
        if st.button(
            "下一家族 →",
            disabled=index == len(families) - 1,
            use_container_width=True,
        ):
            st.session_state.v19_formal_family_index = index + 1
            st.rerun()

    st.write("同一意图的4条自然改写：")
    for query_index, query in enumerate(family["queries"], start=1):
        st.info(f"{query_index}. {query}")
    proposed = str(family["proposed_route"])
    st.write(
        "建议标签：",
        f"**{ROUTE_LABELS[proposed]}** (`{proposed}`)",
    )

    route_options = list(ROUTE_LABELS)
    default_route = saved["final_route"] if saved else proposed
    route_key = f"formal_family_route_{family_id}"
    notes_key = f"formal_family_notes_{family_id}"
    if route_key not in st.session_state:
        st.session_state[route_key] = default_route
    if notes_key not in st.session_state:
        st.session_state[notes_key] = saved["notes"] if saved else ""

    def persist_current_review() -> None:
        save_review(
            review_path,
            family=family,
            final_route=str(st.session_state[route_key]),
            reviewer_id=reviewer_id,
            notes=str(st.session_state[notes_key]),
        )

    st.selectbox(
        "这个家族的最终标签",
        route_options,
        format_func=lambda route: f"{ROUTE_LABELS[route]} · {route}",
        key=route_key,
        on_change=persist_current_review,
    )
    st.text_area(
        "备注（可选；失去焦点后自动保存）",
        key=notes_key,
        on_change=persist_current_review,
    )

    if saved:
        st.success(
            "本家族的人工调整已自动保存："
            f"{ROUTE_LABELS[saved['final_route']]} · {saved['final_route']}"
        )
    else:
        st.info("当前沿用建议标签；如判断正确，直接点击“下一家族”即可。")


if __name__ == "__main__":
    render()
