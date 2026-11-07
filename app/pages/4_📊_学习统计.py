"""学习统计页：高频问题、知识盲区、掌握度、练习趋势（Plotly）与面经文件。"""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from app import config, stats
from app.interview import EXPORT_DIR
from app.ui_components import bootstrap, sidebar

st.set_page_config(page_title="学习统计", page_icon="📊", layout="wide")
bootstrap()
sidebar()

st.title("📊 学习统计与数据分析")
uid = config.DEFAULT_USER_ID
ov = stats.overview(uid)

c1, c2, c3, c4 = st.columns(4)
c1.metric("累计问答", ov["total_qa"])
c2.metric("面试场次", ov["interviews"])
c3.metric("错题数", ov["wrong_total"])
c4.metric("练习时长(分钟)", ov["practice_minutes"])

# ---------- 每日练习时长 & 错题率趋势 ----------
def _zero_range(vals: list[float]) -> list[float]:
    """Y 轴固定从 0 开始；全为 0 时给一个上限，避免出现负值刻度。"""
    top = max(vals) if vals else 0
    return [0, top if top > 0 else 1]


trend = stats.daily_trend(uid, 14)
if trend:
    dates = [d["log_date"] for d in trend]
    mins = [d["practice_minutes"] for d in trend]
    fig_bar = go.Figure(go.Bar(
        x=dates, y=mins,
        text=mins, marker_color="#4C78A8", name="练习分钟",
    ))
    fig_bar.update_layout(
        title="每日练习时长（分钟）", height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(type="category"), yaxis=dict(range=_zero_range(mins)),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    rates = [d["wrong_rate"] for d in trend]
    fig_line = go.Figure(go.Scatter(
        x=dates, y=rates, mode="lines+markers+text",
        text=rates, line=dict(color="#E45756", width=3),
        name="错题率%",
    ))
    fig_line.update_layout(
        title="错题率变化趋势（%）", height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(type="category"), yaxis=dict(range=_zero_range(rates), title="错题率 %"),
    )
    st.plotly_chart(fig_line, use_container_width=True)
else:
    st.info("还没有学习日志，完成问答或模拟面试后这里会生成趋势图")

# ---------- 掌握度 ----------
m = stats.mastery(uid)
left, right = st.columns(2)
if m["total"]:
    with left:
        fig_pie = go.Figure(go.Pie(
            labels=["掌握(≥80)", "一般(60-79)", "薄弱(<60)"],
            values=[m["mastered"], m["average"], m["weak"]], hole=0.45,
            marker=dict(colors=["#54A24B", "#F58518", "#E45756"]),
        ))
        fig_pie.update_layout(title=f"知识点掌握度（已掌握 {m['mastery_pct']}%）", height=340)
        st.plotly_chart(fig_pie, use_container_width=True)
else:
    left.info("完成模拟面试后展示掌握度分布")

# ---------- 高频问题 ----------
top = stats.top_questions(uid, 10)
if top:
    with right:
        fig_top = go.Figure(go.Bar(
            y=[t["question"][:18] for t in top][::-1], x=[t["times"] for t in top][::-1],
            orientation="h", text=[t["times"] for t in top][::-1], marker_color="#72B7B2",
        ))
        fig_top.update_layout(title="高频问题 Top10（询问次数）", height=340,
                              margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_top, use_container_width=True)

# ---------- 知识盲区 ----------
weak = stats.weak_points(uid)
st.subheader("🎯 知识盲区与薄弱点复习清单")
if weak["missing_points"]:
    st.markdown("**面试中反复遗漏的得分点：**")
    for p in weak["missing_points"]:
        st.markdown(f"- {p['point']}（遗漏 {p['times']} 次）")
if weak["wrong_qa"]:
    st.markdown("**被标记为错误的问题：**")
    for w in weak["wrong_qa"][:10]:
        st.markdown(f"- {w['question']}（{w['created_at']}）")
if not weak["missing_points"] and not weak["wrong_qa"]:
    st.success("暂无明显薄弱点，继续保持")

# ---------- 面经文件 ----------
st.subheader("📄 已沉淀面经")
if EXPORT_DIR.exists():
    reports = sorted(EXPORT_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    for rpt in reports:
        with st.expander(rpt.name):
            st.markdown(rpt.read_text(encoding="utf-8"))
            st.download_button("下载", data=rpt.read_bytes(), file_name=rpt.name,
                               key=f"dl_{rpt.name}")
else:
    st.caption("还没有面经文件")
