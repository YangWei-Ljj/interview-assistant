"""模拟面试个人知识库问答系统 · 总览首页。运行：streamlit run app/Home.py"""
from __future__ import annotations

import sys
from pathlib import Path

# 把项目根目录加入 sys.path，保证 `from app import ...` 在直接运行本文件时可用
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app import config, knowledge, stats
from app.ui_components import bootstrap, sidebar

st.set_page_config(page_title="模拟面试知识库", page_icon="🎯", layout="wide")
bootstrap()
sidebar()

st.title("🎯 模拟面试 · 个人知识库问答系统")
st.markdown(
    "专为面试备考打造的**私有 RAG 知识库**：所有回答严格基于你上传的面试资料，"
    "内置混合检索、模拟面试、答案评分与学习统计，数据全部保存在本机。"
)

ov = stats.overview(config.DEFAULT_USER_ID)
m = stats.mastery(config.DEFAULT_USER_ID)
c1, c2, c3, c4 = st.columns(4)
c1.metric("知识文档数", knowledge.count_documents(config.DEFAULT_USER_ID, ready_only=True))
c2.metric("累计问答数", ov["total_qa"])
c3.metric("模拟面试场次", ov["interviews"])
c4.metric("累计练习时长(分钟)", ov["practice_minutes"])

st.divider()
left, right = st.columns([3, 2])
with left:
    st.subheader("新手使用步骤")
    st.markdown(
        """
        1. **📚 知识库管理**：上传 PDF / Word / TXT / 面试截图 / 网页链接，可分文件夹、打标签；
           系统已内置示例题库，打开即可直接体验；
        2. **💬 知识问答**：随时提问，回答基于知识库并标注引用来源，可评分纠错；
        3. **🎤 模拟面试**：选择岗位与难度，AI 面试官随机抽题、连续追问，自动评分并指出遗漏点；
        4. **📊 学习统计**：查看高频问题、知识盲区、掌握度与练习趋势，结束后面经自动沉淀导出。
        """
    )
with right:
    st.subheader("当前掌握度")
    if m["total"]:
        st.metric("已掌握知识点占比", f"{m['mastery_pct']}%")
        st.caption(f"掌握 {m['mastered']} ｜ 一般 {m['average']} ｜ 薄弱 {m['weak']}（共 {m['total']} 题）")
    else:
        st.info("还没有模拟面试评分，去「🎤 模拟面试」完成一场练习吧")
    st.caption(f"分块长度：{config.CHUNK_SIZE} 字符 ｜ 检索 Top-K：{config.RETRIEVER_TOP_K} ｜ 融合权重 向量:BM25 = {config.VECTOR_WEIGHT}:{config.BM25_WEIGHT}")
