"""Internal Streamlit site for image classification and OCR correction."""

from __future__ import annotations

import html

import streamlit as st

from scripts.library_manager import (
    library_by_id,
    load_registry,
    resolve_library_dir,
)


PRIVACY_LABELS = {
    "public": "公开研究",
    "study": "学习科研",
    "private": "私人敏感",
}


def inject_admin_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --admin-navy: #10253B;
            --admin-blue: #2667A8;
            --admin-amber: #E6A23C;
            --admin-ink: #16283B;
            --admin-muted: #607286;
            --admin-line: #D9E2EA;
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 90% 0%, rgba(38, 103, 168, 0.10), transparent 27rem),
                #F3F6F8;
        }
        [data-testid="stHeader"] {
            background: rgba(243, 246, 248, 0.84);
            backdrop-filter: blur(12px);
        }
        .block-container {
            max-width: 1480px;
            padding-top: 1.3rem;
            padding-bottom: 4rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #10253B 0%, #183650 100%);
            border-right: 0;
        }
        [data-testid="stSidebar"] * {
            color: #E9F0F5;
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] > div {
            color: #16283B;
            background: #F8FAFB;
            border-color: rgba(255, 255, 255, 0.18);
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] * {
            color: #16283B;
        }
        .admin-brand {
            display: flex;
            align-items: center;
            gap: 0.72rem;
            margin: 0.3rem 0 1.5rem;
            font-weight: 720;
        }
        .admin-brand-icon {
            display: grid;
            place-items: center;
            width: 2.2rem;
            height: 2.2rem;
            border-radius: 10px;
            color: #10253B;
            background: #F2B85B;
            font-size: 1.05rem;
        }
        .admin-side-note {
            margin-top: 1.2rem;
            padding: 0.85rem;
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            color: rgba(233, 240, 245, 0.78);
            background: rgba(255, 255, 255, 0.05);
            font-size: 0.78rem;
            line-height: 1.65;
        }
        .admin-hero {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 1.5rem;
            margin: 0.25rem 0 1.6rem;
            padding: 2rem 2.2rem;
            border-radius: 22px;
            color: white;
            background:
                radial-gradient(circle at 88% 25%, rgba(242, 184, 91, 0.22), transparent 14rem),
                linear-gradient(120deg, #10253B 0%, #1A4664 72%, #2667A8 100%);
            box-shadow: 0 18px 45px rgba(16, 37, 59, 0.16);
        }
        .admin-eyebrow {
            margin-bottom: 0.65rem;
            color: #F5C97C;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.13em;
        }
        .admin-hero h1 {
            margin: 0;
            color: white;
            font-size: clamp(1.9rem, 3vw, 3rem);
            letter-spacing: -0.035em;
        }
        .admin-hero p {
            max-width: 760px;
            margin: 0.8rem 0 0;
            color: rgba(235, 244, 248, 0.78);
            line-height: 1.7;
        }
        .admin-local-badge {
            flex: 0 0 auto;
            padding: 0.5rem 0.75rem;
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 999px;
            color: #F8D699;
            background: rgba(16, 37, 59, 0.32);
            font-size: 0.76rem;
            font-weight: 650;
        }
        h1, h2, h3, h4 {
            color: var(--admin-ink);
            letter-spacing: -0.025em;
        }
        [data-testid="stMetric"] {
            padding: 0.95rem 1rem;
            border: 1px solid var(--admin-line);
            border-radius: 15px;
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 8px 24px rgba(16, 37, 59, 0.05);
        }
        [data-testid="stExpander"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            border-color: var(--admin-line);
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.90);
        }
        [data-testid="stImage"] img {
            border-radius: 16px;
        }
        div.stButton > button {
            min-height: 2.7rem;
            border-radius: 10px;
            font-weight: 680;
        }
        div.stButton > button[kind="primary"] {
            color: white;
            border: 0;
            background: linear-gradient(110deg, #2667A8, #194B72);
        }
        [data-testid="stAlert"] {
            border-radius: 12px;
        }
        @media (max-width: 760px) {
            .admin-hero {
                align-items: flex-start;
                flex-direction: column;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_admin_header(library_name: str) -> None:
    st.markdown(
        f"""
        <section class="admin-hero">
          <div>
            <div class="admin-eyebrow">INTERNAL DATA GOVERNANCE</div>
            <h1>类别、质量与OCR纠错中心</h1>
            <p>
              审核自动质量路由与模型分类冲突，修正类别、OCR状态和隔离决定；
              所有裁决保留审计记录，并可反哺后续反馈学习。
            </p>
          </div>
          <div class="admin-local-badge">仅本机 · {html.escape(library_name)}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="OCR-VLM 内部纠错中心",
        page_icon="🛠️",
        layout="wide",
    )
    inject_admin_style()

    registry = load_registry()
    st.sidebar.markdown(
        """
        <div class="admin-brand">
          <span class="admin-brand-icon">✓</span>
          <span>内部质量治理</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    library_ids = [row["id"] for row in registry["libraries"]]
    default_id = (
        "public_research_200"
        if "public_research_200" in library_ids
        else registry["active_library_id"]
    )
    selected_id = st.sidebar.selectbox(
        "待审核资料库",
        options=library_ids,
        index=library_ids.index(default_id),
        format_func=lambda library_id: (
            f"{library_by_id(registry, library_id)['name']} · "
            f"{PRIVACY_LABELS[library_by_id(registry, library_id)['privacy']]}"
        ),
    )
    selected_library = library_by_id(registry, selected_id)
    selected_library_dir = resolve_library_dir(selected_library)
    workflow = st.sidebar.radio(
        "审核任务",
        options=(
            "图片分类与OCR纠错",
            "检索真值审核",
            "独立盲测审核",
        ),
    )
    st.sidebar.markdown(
        """
        <div class="admin-side-note">
          此站点不属于对外产品页面。发布决定前需要再次勾选确认；
          原图不会被删除，旧manifest会自动备份。
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_admin_header(selected_library["name"])
    from app import (
        render_blind_study_review,
        render_quality_review,
        render_query_review,
    )

    requested_item_id = str(st.query_params.get("item_id", "")).strip()
    if requested_item_id:
        st.info(
            f"已从检索结果定位页面：{requested_item_id}。"
            "保存后可返回主站继续检索。"
        )
    if workflow == "独立盲测审核":
        render_blind_study_review(selected_library, selected_library_dir)
    elif workflow == "检索真值审核":
        render_query_review()
    else:
        render_quality_review(
            selected_library,
            selected_library_dir,
            requested_item_id=requested_item_id,
        )
    st.caption(
        "内部治理站点默认仅监听127.0.0.1；请勿把包含私人资料的页面截图公开上传。"
    )


if __name__ == "__main__":
    main()
