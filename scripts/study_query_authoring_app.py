"""Local Streamlit UI for source-grounded paired-query authoring."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import streamlit as st

from ocr_vlm_retrieval.studies.query_split import (  # noqa: E402
    ALLOWED_CHANGED_KINDS_BY_STRATUM,
    validate_authored_pair,
)

RUNTIME_ROOT = Path(os.environ.get("STUDY_RUNTIME_ROOT", str(PROJECT_ROOT))).resolve()
QUEUE_PATH = Path(
    os.environ.get(
        "STUDY_AUTHORING_QUEUE",
        str(RUNTIME_ROOT / "data/evaluation/v18/query_authoring/authoring_queue.jsonl"),
    )
).resolve()
SUBMISSIONS_PATH = Path(
    os.environ.get(
        "STUDY_AUTHORING_SUBMISSIONS",
        str(
            RUNTIME_ROOT
            / "data/evaluation/v18/query_authoring/authoring_submissions.jsonl"
        ),
    )
).resolve()
FIXED_AUTHOR_ID = os.environ.get("STUDY_AUTHOR_ID", "").strip()

STRATUM_LABELS = {
    "attribute_object_binding": "属性—对象绑定",
    "spatial_relation": "明确空间关系",
    "ocr_visual_condition": "OCR 文字 + 视觉条件",
    "multi_condition_scene": "多条件场景",
}
CONDITION_LABELS = {
    "object": "对象",
    "scene": "场景",
    "color": "颜色",
    "relation": "关系",
    "attribute_binding": "属性绑定",
    "ocr_text": "OCR 文字",
}


def install_translation_guard() -> None:
    """Keep browser translation from mutating React-owned nodes and queries."""

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
          let meta = doc.head.querySelector('meta[name="google"]');
          if (!meta) {
            meta = doc.createElement("meta");
            meta.name = "google";
            doc.head.appendChild(meta);
          }
          meta.content = "notranslate";
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


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


def save_submission(submission: dict[str, Any]) -> None:
    indexed = {str(row["authoring_id"]): row for row in read_jsonl(SUBMISSIONS_PATH)}
    indexed[str(submission["authoring_id"])] = submission
    write_jsonl_atomic(SUBMISSIONS_PATH, [indexed[key] for key in sorted(indexed)])


def submission_status_ids(
    submissions: dict[str, dict[str, Any]],
) -> tuple[set[str], set[str]]:
    approved = {
        key for key, row in submissions.items() if row.get("review_action") == "approve"
    }
    codex_drafts = {
        key
        for key, row in submissions.items()
        if row.get("review_action") == "codex_draft"
    }
    return approved, codex_drafts


def source_image_path(row: dict[str, Any]) -> Path | None:
    raw = str(row.get("image_path", "")).strip()
    if not raw:
        return None
    path = Path(raw)
    resolved = path.resolve() if path.is_absolute() else (RUNTIME_ROOT / path).resolve()
    return resolved if resolved.is_file() else None


def main() -> None:
    st.set_page_config(page_title="V18 成对查询创作", layout="wide")
    install_translation_guard()
    st.title("V18 成对查询创作与审核")
    st.caption(
        "每张来源图创建两条自然查询：一条由该图完整满足；另一条只改变一个必要"
        "条件，使该图不再满足。此阶段不会运行检索，也不会打开 V18 留出结果。"
    )
    st.warning(
        "请关闭浏览器翻译扩展对本页的自动翻译。页面已主动设置 translate=no，"
        "用于避免换页状态丢失和 React removeChild 错误。"
    )

    queue = read_jsonl(QUEUE_PATH)
    if not queue:
        st.error(f"未找到查询创作队列：{QUEUE_PATH}")
        return
    submissions = {
        str(row["authoring_id"]): row for row in read_jsonl(SUBMISSIONS_PATH)
    }
    author_id = st.sidebar.text_input(
        "创作审核者 ID",
        value=FIXED_AUTHOR_ID,
        disabled=bool(FIXED_AUTHOR_ID),
    ).strip()
    completed_ids, draft_ids = submission_status_ids(submissions)
    pending = [row for row in queue if row["authoring_id"] not in completed_ids]
    st.sidebar.metric("来源图总数", len(queue))
    st.sidebar.metric("已完成", len(completed_ids))
    st.sidebar.metric("Codex 草稿", len(draft_ids))
    st.sidebar.metric("待完成", len(pending))
    show_completed = st.sidebar.checkbox("显示已完成来源图", value=False)
    visible = queue if show_completed else pending
    if not author_id:
        st.info("请先在左侧填写创作审核者 ID。")
        return
    if not visible:
        st.success("全部 80 个来源图的成对查询均已完成。")
        return

    labels = {
        str(row["authoring_id"]): (
            f"{row['authoring_id']} · {STRATUM_LABELS[str(row['stratum'])]} · "
            f"{row['language_target']}"
        )
        + (" · Codex 草稿" if str(row["authoring_id"]) in draft_ids else "")
        for row in visible
    }
    selected_id = st.selectbox(
        "选择来源图",
        [str(row["authoring_id"]) for row in visible],
        format_func=lambda value: labels[value],
    )
    row = next(item for item in visible if item["authoring_id"] == selected_id)
    existing = submissions.get(selected_id, {})

    if existing.get("review_action") == "codex_draft":
        st.warning(
            "这是 Codex 生成的待审核草稿，不是人工标签。请逐字核对图片，按需修改，"
            "并在勾选确认后保存。"
        )

    st.markdown("#### 当前绑定对象")
    st.info(
        f"{selected_id} · {STRATUM_LABELS[str(row['stratum'])]} · "
        f"目标语言 {row['language_target']}"
    )
    left, right = st.columns([3, 2])
    with left:
        image_path = source_image_path(row)
        if image_path is None:
            st.error("来源图不存在，已停止本条创作。")
            return
        st.image(str(image_path), width="stretch")
    with right:
        st.markdown("**机械证据提示（可能有误，以原图为准）**")
        colors = ", ".join(str(value) for value in row.get("dominant_colors", []))
        st.write(f"主色候选：{colors or '无'}")
        ocr = str(row.get("ocr_excerpt", "")).strip()
        st.text_area("OCR 摘录", value=ocr or "无", height=240, disabled=True)
        st.write(
            "正例必须同时包含至少两个必要条件。强负例只能改一个条件；对象、"
            "场景、颜色、关系或文字中的其余条件保持不变。"
        )

    stratum = str(row["stratum"])
    allowed_kinds = sorted(ALLOWED_CHANGED_KINDS_BY_STRATUM[stratum])
    with st.form(f"authoring::{selected_id}"):
        positive_query = st.text_area(
            "正例查询（该图完整满足）",
            value=str(existing.get("positive_query", "")),
            help="使用自然表达，不要照抄固定模板。",
        )
        hard_negative_query = st.text_area(
            "单条件强负例（该图不满足）",
            value=str(existing.get("hard_negative_query", "")),
        )
        previous_kind = str(existing.get("changed_condition_kind", ""))
        default_index = (
            allowed_kinds.index(previous_kind) if previous_kind in allowed_kinds else 0
        )
        changed_kind = st.selectbox(
            "唯一被改变的必要条件",
            allowed_kinds,
            index=default_index,
            format_func=lambda value: CONDITION_LABELS[value],
        )
        confirmed = st.checkbox(
            "我已逐字核对：两条查询只有一个必要条件不同，且正例满足、强负例不满足",
            value=bool(existing.get("single_condition_confirmed", False)),
        )
        notes = st.text_area("备注（可选）", value=str(existing.get("notes", "")))
        submitted = st.form_submit_button("保存并完成本条", type="primary")
    if submitted:
        try:
            validate_authored_pair(
                positive_query,
                hard_negative_query,
                language_target=str(row["language_target"]),
                changed_condition_kind=changed_kind,
                single_condition_confirmed=confirmed,
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        save_submission(
            {
                "authoring_id": selected_id,
                "review_action": "approve",
                "positive_query": " ".join(positive_query.split()),
                "hard_negative_query": " ".join(hard_negative_query.split()),
                "changed_condition_kind": changed_kind,
                "single_condition_confirmed": confirmed,
                "reviewer_id": author_id,
                "notes": notes.strip(),
                "reviewed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        st.success(f"已保存 {selected_id}；换页后仍会从文件恢复。")
        st.rerun()


if __name__ == "__main__":
    main()
