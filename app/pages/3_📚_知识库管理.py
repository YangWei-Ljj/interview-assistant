"""知识库管理页：多格式上传、网页导入、文件夹/标签、批量操作、索引与图谱维护。"""
from __future__ import annotations

import streamlit as st

from app import config, knowledge
from app.ui_components import bootstrap, sidebar

st.set_page_config(page_title="知识库管理", page_icon="📚", layout="wide")
bootstrap()
sidebar()

st.title("📚 知识库管理")
uid = config.DEFAULT_USER_ID

tab_up, tab_url, tab_folder = st.tabs(["⬆️ 上传文件", "🔗 网页链接导入", "🗂️ 文件夹管理"])

with tab_up:
    files = st.file_uploader(
        "选择文件（支持 PDF/Word/TXT/图片 OCR，单文件≤100MB，可多选）",
        accept_multiple_files=True,
        type=[ext.lstrip(".") for ext in config.ALLOWED_UPLOAD_EXT],
    )
    folders = knowledge.list_folders()
    fopts = {f["id"]: f["name"] for f in folders}
    target = st.selectbox("归入文件夹（可选）", options=[None, *fopts.keys()],
                          format_func=lambda x: "未分类" if x is None else fopts[x], key="up_folder")
    tags = st.text_input("标签（逗号分隔，如 Python,RAG,一面）", key="up_tags")
    if st.button("开始上传入库", disabled=not files):
        for f in files:
            with st.spinner(f"正在处理 {f.name} ..."):
                try:
                    r = knowledge.ingest_file(
                        uid, f.name, f.read(), folder_id=target, tags=tags
                    )
                    st.success(f"{f.name}：{r['message']}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"{f.name} 失败：{exc}")

with tab_url:
    url = st.text_input("粘贴面试资料/面经网页链接", placeholder="https://...")
    if st.button("抓取并导入", disabled=not url):
        from app.document_loader import fetch_url
        try:
            parsed = fetch_url(url)
            r = knowledge.ingest_text(
                uid, parsed.get("title", url) + ".txt", parsed["text"],
                title=parsed.get("title"), doc_type="webpage",
            )
            st.success(r["message"])
        except Exception as exc:  # noqa: BLE001
            st.error(f"导入失败：{exc}")

with tab_folder:
    with st.form("folder_form"):
        fname = st.text_input("新文件夹名称（如：后端岗 / Java技术栈 / 二面）")
        ftype = st.selectbox("分类维度", ["job", "tech", "round", "custom"])
        if st.form_submit_button("创建文件夹"):
            knowledge.create_folder(fname, folder_type=ftype)
            st.success("已创建")
    for f in knowledge.list_folders():
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"**{f['name']}**（{f['folder_type']}）")
        if c2.button("删除", key=f"del_{f['id']}"):
            knowledge.delete_folder(f["id"])
            st.rerun()

st.divider()
st.subheader("文档列表与批量操作")
docs = knowledge.list_documents(uid)
if docs:
    label_map = {d["id"]: f"{d['title']} ｜ {d['status']} ｜ v{d['version']} ｜ {d['chunk_count']}块"
                 for d in docs}
    selected = st.multiselect("选择要操作的文档", options=[d["id"] for d in docs],
                              format_func=lambda x: label_map[x])
    m1, m2, m3, m4 = st.columns(4)
    move_target = m1.selectbox("移动到", options=[None, *fopts.keys()],
                               format_func=lambda x: "未分类" if x is None else fopts[x], key="move")
    if m2.button("批量移动", use_container_width=True, disabled=not selected):
        knowledge.move_documents(uid, selected, move_target)
        st.success("已移动"); st.rerun()
    if m3.button("批量归档", use_container_width=True, disabled=not selected):
        knowledge.set_archive(uid, selected, True)
        st.success("已归档"); st.rerun()
    if m4.button("批量删除", use_container_width=True, disabled=not selected):
        for did in selected:
            knowledge.delete_document(uid, did)
        st.success("已删除"); st.rerun()

    st.dataframe(
        [{"标题": d["title"], "文件": d["filename"], "状态": d["status"], "标签": d["tags"],
          "版本": d["version"], "块数": d["chunk_count"], "更新时间": d["updated_at"]} for d in docs],
        use_container_width=True, hide_index=True,
    )
else:
    st.info("暂无文档，可先上传或使用内置示例题库")

st.divider()
st.subheader("索引与图谱维护")
c1, c2, c3 = st.columns(3)
if c1.button("重建向量索引", use_container_width=True):
    n = knowledge.rebuild_index()
    st.success(f"已重建，共 {n} 块")
if c2.button("重建知识图谱", use_container_width=True):
    from app import graph
    r = graph.build_graph(uid)
    st.success(f"图谱：{r['nodes']} 节点 / {r['edges']} 关系")
if c3.button("清空检索缓存", use_container_width=True):
    from app.retriever import clear_cache
    from app.qa import clear_answer_cache
    clear_cache()
    clear_answer_cache()
    st.success("检索缓存与答案缓存已清空")

ist = knowledge.index_stats()
st.caption(f"文档 {ist['documents']}（就绪 {ist['ready']} / 归档 {ist['archived']} / 失败 {ist['failed']}）"
           f"｜ SQLite 块 {ist['chunks_sqlite']} ｜ 向量块 {ist['chunks_vector']} ｜ 一致：{ist['consistent']}")
