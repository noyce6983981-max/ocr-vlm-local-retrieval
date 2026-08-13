"""V19 development Top-20多相关性复核网页。"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

PACKETS_PATH = Path(
    os.environ.get(
        "V19_TOP20_PACKETS",
        str(
            ROOT
            / "records/private/v19/selective_intervention"
            / "development_top20_review_packets.jsonl"
        ),
    )
).resolve()
JUDGMENTS_PATH = Path(
    os.environ.get(
        "V19_TOP20_JUDGMENTS",
        str(
            ROOT
            / "records/private/v19/selective_intervention"
            / "development_top20_judgments.jsonl"
        ),
    )
).resolve()
REVIEWER_ID = os.environ.get("V19_TOP20_REVIEWER_ID", "primary_reviewer").strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_judgment(row: dict[str, Any]) -> None:
    indexed = {
        (str(source["query_id"]), str(source["reviewer_id"])): source
        for source in read_jsonl(JUDGMENTS_PATH)
    }
    key = (str(row["query_id"]), str(row["reviewer_id"]))
    indexed[key] = row
    write_jsonl_atomic(JUDGMENTS_PATH, [indexed[key] for key in sorted(indexed)])


def install_translation_guard() -> None:
    st.html(
        """
        <script>
        (() => {
          const doc = document;
          doc.documentElement.lang = "zh-CN";
          doc.documentElement.setAttribute("translate", "no");
          doc.documentElement.classList.add("notranslate");
          if (doc.body) {
            doc.body.setAttribute("translate", "no");
            doc.body.classList.add("notranslate");
          }
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def candidate_path(candidate: dict[str, Any]) -> Path | None:
    path = ROOT / str(candidate.get("review_asset", {}).get("image_path", ""))
    return path if path.is_file() else None


def move(cursor_key: str, delta: int, total: int) -> None:
    current = int(st.session_state.get(cursor_key, 1))
    st.session_state[cursor_key] = min(max(current + delta, 1), total)


def main() -> None:
    st.set_page_config(page_title="V19开发集 Top-20 复核", layout="wide")
    install_translation_guard()
    st.title("V19 开发集 Top-20 多相关性复核")
    st.caption(
        "只审核 development；页面隐藏方法、分数、原始排名和系统答案。当前结果不会解封 holdout。"
    )
    packets = read_jsonl(PACKETS_PATH)
    if not packets:
        st.error("尚未生成 development Top-20 候选池。")
        return
    if {str(row.get("split")) for row in packets} != {"development"}:
        st.error("候选池混入非 development 数据，已停止。")
        return
    judgments = {
        str(row["query_id"]): row
        for row in read_jsonl(JUDGMENTS_PATH)
        if row.get("reviewer_id") == REVIEWER_ID
    }
    show_completed = st.sidebar.checkbox("显示已完成查询", value=False)
    visible = (
        packets
        if show_completed
        else [row for row in packets if row["query_id"] not in judgments]
    )
    st.sidebar.metric("development 查询", len(packets))
    st.sidebar.metric("已完成", len(judgments))
    st.sidebar.metric("待完成", len(packets) - len(judgments))
    if not visible:
        st.success("development Top-20 复核已全部完成。")
        return

    cursor_key = f"v19_top20_cursor::{REVIEWER_ID}::{show_completed}"
    if not 1 <= int(st.session_state.get(cursor_key, 1)) <= len(visible):
        st.session_state[cursor_key] = 1
    left, middle, right = st.columns([1, 2, 1])
    position = int(
        middle.number_input(
            "查询序号",
            min_value=1,
            max_value=len(visible),
            step=1,
            key=cursor_key,
        )
    )
    left.button(
        "← 上一条",
        disabled=position <= 1,
        on_click=move,
        args=(cursor_key, -1, len(visible)),
        width="stretch",
    )
    right.button(
        "下一条 →",
        disabled=position >= len(visible),
        on_click=move,
        args=(cursor_key, 1, len(visible)),
        width="stretch",
    )
    packet = visible[position - 1]
    query_id = str(packet["query_id"])
    existing = judgments.get(query_id, {})
    st.info(f"{query_id} · {packet['query']}")
    st.write("请勾选所有同时满足查询全部必要条件的候选；可以多选，也可以全部不选。")
    previous = dict(existing.get("candidate_relevance", {}))
    candidate_relevance: dict[str, bool] = {}
    columns = st.columns(5)
    for offset, candidate in enumerate(packet["candidates"]):
        item_id = str(candidate["item_id"])
        with columns[offset % 5]:
            path = candidate_path(candidate)
            if path is None:
                st.warning("图片不可用")
            else:
                st.image(str(path), width="stretch")
            candidate_relevance[item_id] = st.checkbox(
                "完整相关",
                value=bool(previous.get(item_id, False)),
                key=f"v19_top20::{REVIEWER_ID}::{query_id}::{item_id}",
            )
    notes = st.text_area("备注（可选）", value=str(existing.get("notes", "")))
    if st.button("保存并完成本条", type="primary"):
        save_judgment(
            {
                "query_id": query_id,
                "reviewer_id": REVIEWER_ID,
                "candidate_relevance": candidate_relevance,
                "notes": notes.strip(),
                "task_id": packet["task_id"],
                "split": "development",
                "pool_sha256": packet["pool_sha256"],
                "eligible_for_final_claim": False,
                "reviewed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        st.success("已保存，换页或退出不会丢失。")
        st.rerun()


if __name__ == "__main__":
    main()
