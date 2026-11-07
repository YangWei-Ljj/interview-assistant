"""Streamlit 复用组件：初始化、首次自动播种、侧边栏状态、引用来源与反馈控件。"""
from __future__ import annotations

import streamlit as st

from . import config, knowledge
from .db import init_db


def bootstrap() -> None:
    """每个页面入口调用：建表 + 首次自动导入示例题库（幂等）。"""
    init_db()
    flag = config.DATA_DIR / ".seeded"
    if not flag.exists():
        from . import graph
        if knowledge.count_documents(config.DEFAULT_USER_ID) == 0 and any(
            config.SEED_DOC_DIR.glob("*")
        ):
            knowledge.seed_documents(config.DEFAULT_USER_ID)
            graph.build_graph(config.DEFAULT_USER_ID)
        flag.write_text("seeded", encoding="utf-8")


def _light_health() -> tuple[bool, str]:
    """轻量探测：访问 Ollama /api/tags，不触发 LLM 生成，避免阻塞首页。"""
    import requests
    base = config.OLLAMA_BASE_URL.replace("/v1", "")
    try:
        resp = requests.get(f"{base}/api/tags", timeout=5)
        names = [m["name"] for m in resp.json().get("models", [])]
    except Exception as exc:  # noqa: BLE001
        return False, f"Ollama 未连接（请先启动 Ollama）：{type(exc).__name__}"
    have_chat = config.OLLAMA_MODEL in names
    have_embed = config.OLLAMA_EMBED_MODEL in names
    if have_chat and have_embed:
        return True, "本地 Ollama 在线，对话与嵌入模型均就绪"
    missing = []
    if not have_chat:
        missing.append(config.OLLAMA_MODEL)
    if not have_embed:
        missing.append(config.OLLAMA_EMBED_MODEL)
    return False, f"模型缺失：{', '.join(missing)}（请 ollama pull）"


def sidebar() -> None:
    with st.sidebar:
        st.markdown("### 模型与服务状态")
        if "health" not in st.session_state:
            st.session_state["health"] = _light_health()
        ok, msg = st.session_state["health"]
        st.markdown(f"{'🟢' if ok else '🔴'} {msg}")
        st.caption(f"对话模型：{config.MODEL_PROVIDER}")
        if st.button("重新检测模型", use_container_width=True):
            st.session_state["health"] = _light_health()
            st.rerun()
        st.divider()
        st.caption("数据全部保存在本机，不会上传云端")


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    st.markdown("#### 引用来源（答案依据）")
    for i, d in enumerate(sources, 1):
        with st.expander(
            f"来源 {i}：{d.get('source')} ｜ 相关度 {d.get('final_score','-')}"
        ):
            st.caption(
                f"类型：{d.get('meta_type','text')} ｜ 向量相似度：{d.get('vector_sim',0)} "
                f"｜ BM25：{d.get('bm25_norm',0)} ｜ 图谱：{d.get('graph_score',0)}"
            )
            st.markdown(d["content"])


def render_feedback(record_id: str) -> None:
    st.markdown("#### 这条回答有帮助吗？")
    c1, c2 = st.columns([3, 1])
    with c1:
        rating = st.radio(
            "评分（1-5 星）", options=[1, 2, 3, 4, 5], horizontal=True,
            key=f"rate_{record_id}", label_visibility="collapsed",
        )
    with c2:
        submit = st.button("提交评分", key=f"submit_{record_id}")
    if st.button("标记答案不准确 / 有错误", key=f"wrong_{record_id}"):
        from .qa import rate_record
        rate_record(record_id, rating=None, marked_wrong=True)
        st.warning("已标记为错误并加入薄弱点清单，可点击下方按钮重新检索生成")
    if submit:
        from .qa import rate_record
        rate_record(record_id, rating=rating, marked_wrong=False)
        st.success(f"已提交 {rating} 星评分，感谢反馈，系统会据此持续优化")
