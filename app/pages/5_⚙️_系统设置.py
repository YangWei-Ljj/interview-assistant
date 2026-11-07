"""系统设置页：运行参数调整、恢复默认、本地备份与恢复、模型与数据信息。"""
from __future__ import annotations

import streamlit as st

from app import config
from app.ui_components import bootstrap, sidebar

st.set_page_config(page_title="系统设置", page_icon="⚙️", layout="wide")
bootstrap()
sidebar()

st.title("⚙️ 系统设置")

# ---------------- 运行参数 ----------------
st.subheader("检索与分块参数")
with st.form("settings_form"):
    chunk_size = st.slider(
        "文本分块长度（字符）", config.CHUNK_SIZE_MIN, config.CHUNK_SIZE_MAX,
        value=int(config.get_setting("chunk_size")),
    )
    overlap = st.number_input("分块重叠字符数", 0, 300,
                              value=int(config.get_setting("chunk_overlap")))
    top_k = st.number_input("检索返回条数 Top-K", 1, 20,
                            value=int(config.get_setting("top_k")))
    c1, c2 = st.columns(2)
    vw = c1.number_input("向量语义权重", 0.0, 1.0, step=0.05,
                         value=float(config.get_setting("vector_weight")))
    bw = c2.number_input("BM25 关键词权重", 0.0, 1.0, step=0.05,
                         value=float(config.get_setting("bm25_weight")))
    boost = st.number_input("知识图谱扩展加分", 0.0, 1.0, step=0.05,
                            value=float(config.get_setting("graph_boost")))
    c3, c4, c5 = st.columns(3)
    cache_on = c3.toggle("启用 L1/L2 缓存", value=bool(config.get_setting("cache_enabled")))
    check_on = c4.toggle("启用答案三重校验", value=bool(config.get_setting("triple_check_enabled")))
    hybrid_on = c5.toggle("启用 BM25 混合检索", value=bool(config.get_setting("hybrid_search_enabled")))

    if st.form_submit_button("保存设置", use_container_width=True):
        try:
            config.save_user_settings({
                "chunk_size": chunk_size, "chunk_overlap": overlap, "top_k": top_k,
                "vector_weight": vw, "bm25_weight": bw, "graph_boost": boost,
                "cache_enabled": cache_on, "triple_check_enabled": check_on,
                "hybrid_search_enabled": hybrid_on,
            })
            st.success("设置已保存，刷新页面或重启系统后生效（权重会自动归一化）")
        except ValueError as exc:
            st.error(str(exc))

if st.button("恢复全部默认设置"):
    config.reset_user_settings()
    st.success("已恢复默认，重启后生效")

# ---------------- 模型信息 ----------------
st.subheader("模型与数据")
st.caption(
    f"模型提供方：{config.MODEL_PROVIDER} ｜ 对话模型：{config.OLLAMA_MODEL if config.MODEL_PROVIDER=='ollama' else config.DEEPSEEK_MODEL}"
    f" ｜ 嵌入模型：{config.OLLAMA_EMBED_MODEL}"
)
st.caption(f"数据目录（全部本地）：{config.DATA_DIR}")
st.caption(f"访问地址：http://localhost:{config.STREAMLIT_PORT}")

# ---------------- 备份与恢复 ----------------
st.subheader("本地备份 / 恢复（跨设备迁移）")
from app import backup

if st.button("立即创建备份"):
    p = backup.create_backup()
    st.success(f"备份已创建：{p.name}")

backups = backup.list_backups()
if backups:
    st.dataframe(backups, use_container_width=True, hide_index=True)
    bname = st.selectbox("选择备份文件", options=[b["name"] for b in backups])
    confirm = st.checkbox("我确认恢复会覆盖当前数据（系统会先自动备份一份）")
    c1, c2 = st.columns(2)
    if c1.button("恢复所选备份", disabled=not confirm, use_container_width=True):
        safety = backup.restore_backup(bname)
        st.success(f"已恢复，恢复前数据已自动备份为：{safety.name}，重启系统后生效")
    if c2.button("删除所选备份", use_container_width=True):
        backup.delete_backup(bname)
        st.rerun()
else:
    st.info("暂无备份文件")
