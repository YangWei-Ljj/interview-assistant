"""知识问答页：文字/截图/语音提问，基于知识库作答，展示来源、校验与反馈。"""
from __future__ import annotations

import io

import streamlit as st

from app import config, qa
from app.ui_components import bootstrap, render_feedback, render_sources, sidebar

st.set_page_config(page_title="知识问答", page_icon="💬", layout="wide")
bootstrap()
sidebar()

st.title("💬 知识问答")
st.caption("回答严格基于你的知识库，无相关资料时会明确告知，避免大模型幻觉")


@st.cache_resource(show_spinner="首次加载语音模型，请稍候...")
def _asr_model():
    from faster_whisper import WhisperModel
    return WhisperModel(config.ASR_MODEL, device="cpu", compute_type="int8")


def _transcribe(audio_bytes: bytes) -> str:
    try:
        model = _asr_model()
    except Exception as exc:  # 依赖缺失等
        st.error(f"语音功能未就绪：{exc}。可执行 python -m pip install faster-whisper 后重启，或直接用文字提问。")
        return ""
    segments, _ = model.transcribe(io.BytesIO(audio_bytes), language="zh", vad_filter=True)
    return "".join(seg.text for seg in segments).strip()


tab_text, tab_shot, tab_voice = st.tabs(["⌨️ 文字提问", "📷 截图识别题目", "🎤 语音提问"])

with tab_shot:
    shot = st.file_uploader("上传题目截图（自动 OCR 识别）", type=list(config.IMAGE_EXT), key="shot")
    if shot and st.button("识别截图文字", key="ocr_btn"):
        from app.llm import vision_ocr
        with st.spinner("正在识别图片文字..."):
            st.session_state["question_input"] = vision_ocr(shot.read(), shot.type or "image/png")
        st.success("已识别，可在文字提问页编辑后提交")

with tab_voice:
    audio = st.audio_input("点击录制语音问题")
    if audio is not None and st.button("转写语音", key="asr_btn"):
        text = _transcribe(audio.read())
        if text:
            st.session_state["question_input"] = text
            st.success(f"转写结果：{text}")

with tab_text:
    with st.form("qa_form", clear_on_submit=False):
        question = st.text_area(
            "输入你的问题", value=st.session_state.get("question_input", ""),
            placeholder="例如：请介绍一下 RAG 的混合检索为什么能提升准确率？", height=100,
        )
        col1, col2 = st.columns(2)
        submit = col1.form_submit_button("提交提问", use_container_width=True)
        force = col2.form_submit_button("强制重新检索生成", use_container_width=True)

if submit or force:
    if not question.strip():
        st.warning("请先输入问题")
    else:
        with st.spinner("正在检索知识库并生成答案..."):
            res = qa.answer_question(config.DEFAULT_USER_ID, question, force_refresh=force)
        st.session_state["last_result"] = res

res = st.session_state.get("last_result")
if res:
    st.divider()
    tag = f"问题类型：{res['question_type']}"
    if res["cache_hit"]:
        tag += f" ｜ 命中{res['cache_hit']}缓存"
    tag += f" ｜ 耗时 {res['latency_ms']} ms"
    st.caption(tag)
    if res["check"].get("regenerated"):
        st.info("答案经三重校验发现资料外内容，已按“仅依据知识库”重新生成")
    st.markdown("### 回答")
    st.markdown(res["answer"])
    if res["star_tip"]:
        with st.expander("💡 行为面试 STAR 框架引导"):
            st.markdown(res["star_tip"])
    render_sources(res["sources"])
    render_feedback(res["record_id"])

st.divider()
with st.expander("🕘 最近问答记录"):
    for r in qa.list_records(config.DEFAULT_USER_ID, limit=10):
        st.markdown(f"**{r['question']}** ｜ {r['question_type']} ｜ {r['created_at']}")
        st.caption(r["answer"][:150])
