"""模拟面试页：配置岗位/难度/压力模式，AI 出题、答题评分、追问、结束并导出面经。"""
from __future__ import annotations

import streamlit as st

from app import config, interview
from app.ui_components import bootstrap, sidebar

st.set_page_config(page_title="模拟面试", page_icon="🎤", layout="wide")
bootstrap()
sidebar()

st.title("🎤 模拟面试")
uid = config.DEFAULT_USER_ID
sid = st.session_state.get("interview_sid")

# ---------------- 没有进行中的会话：配置并开始 ----------------
if not sid or (interview.get_session(sid) and interview.get_session(sid)["status"] == "finished"):
    st.markdown("选择面试配置，开始一场完全基于你知识库的模拟面试。")
    with st.form("start_form"):
        job = st.selectbox("目标岗位", config.JOB_TYPES)
        diff = st.selectbox("难度等级", config.DIFFICULTY_LEVELS)
        stress = st.toggle("开启压力面试（不断深挖、尖锐反问）", value=False)
        if st.form_submit_button("开始面试", use_container_width=True):
            new_sid = interview.create_session(uid, job, diff, stress)
            interview.next_question(uid, new_sid)  # 先出第一题
            st.session_state["interview_sid"] = new_sid
            st.rerun()

# ---------------- 进行中 ----------------
if sid:
    session = interview.get_session(sid)
    if session and session["status"] == "ongoing":
        msgs = interview.get_messages(sid)
        answered = interview.answered_count(sid)
        st.caption(
            f"岗位：{session['job_type']} ｜ 难度：{session['difficulty']} ｜ "
            f"压力模式：{'是' if session['stress_mode'] else '否'} ｜ 已答 {answered} 题"
        )
        st.progress(min(answered / config.INTERVIEW_QUESTION_COUNT, 1.0),
                    text=f"进度 {answered}/{config.INTERVIEW_QUESTION_COUNT}（可随时提前结束）")

        pending = next((m for m in reversed(msgs) if m["user_answer"] == ""), None)
        if pending:
            st.markdown("### 面试官提问")
            st.markdown(f"**{pending['question']}**")
            if pending["star_hint"]:
                st.info("这是行为面试题，请按 STAR（情境-任务-行动-结果）结构组织回答")
            with st.form("answer_form"):
                user_answer = st.text_area("你的回答", height=180,
                                           placeholder="像真实面试一样口述你的答案...")
                if st.form_submit_button("提交回答并评分", use_container_width=True):
                    with st.spinner("面试官正在评分..."):
                        interview.submit_answer(sid, user_answer)
                    st.rerun()

        # 最近一轮反馈
        last_scored = next((m for m in reversed(msgs) if m["score"] is not None), None)
        if last_scored:
            import json
            fb = json.loads(last_scored["feedback"])
            st.divider()
            st.markdown("### 上一轮评分")
            st.metric("得分", f"{last_scored['score']} / 100")
            if fb.get("missing_points"):
                st.markdown("**遗漏的关键得分点：**")
                for p in fb["missing_points"]:
                    st.markdown(f"- {p}")
            st.markdown(f"**改进建议：** {fb.get('suggestions','')}")
            with st.expander("查看参考答案"):
                st.markdown(last_scored["reference_answer"])

        st.divider()
        c1, c2 = st.columns(2)
        if c1.button("下一题 / 接受追问", use_container_width=True):
            with st.spinner("面试官正在出题..."):
                interview.next_question(uid, sid)
            st.rerun()
        if c2.button("结束面试并导出面经", use_container_width=True):
            with st.spinner("正在生成结构化面经..."):
                result = interview.end_session(sid)
            st.session_state["finished_result"] = result
            st.rerun()

    # ---------------- 已结束：汇总与下载 ----------------
    finished = st.session_state.get("finished_result")
    if session and session["status"] == "finished":
        st.success("本场面试已结束，面经已自动沉淀")
        summary = interview.session_summary(sid)
        c1, c2, c3 = st.columns(3)
        c1.metric("答题数", summary["answered"])
        c2.metric("平均得分", summary["avg_score"])
        c3.metric("练习时长(秒)", finished["practice_seconds"] if finished else 0)
        ep = session["export_path"]
        if ep:
            with open(ep, "rb") as fh:
                st.download_button(
                    "下载面经 Markdown", data=fh.read(),
                    file_name=ep.split("\\")[-1], mime="text/markdown",
                    use_container_width=True,
                )
        st.caption(f"面经保存位置：{ep}")
        if st.button("开始一场新的面试", use_container_width=True):
            st.session_state.pop("interview_sid", None)
            st.session_state.pop("finished_result", None)
            st.rerun()

# ---------------- 历史会话 ----------------
st.divider()
with st.expander("📚 历史面试场次"):
    for s in interview.list_sessions(uid):
        st.markdown(
            f"**{s['job_type']}** ｜ {s['difficulty']} ｜ {s['status']} ｜ {s['started_at']}"
        )
