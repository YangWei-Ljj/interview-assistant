"""模拟面试：AI 面试官出题/追问、STAR 引导、答案对比评分、压力面试与面经沉淀。"""
from __future__ import annotations

import json
import random
import uuid
from datetime import datetime
from pathlib import Path

from . import config, knowledge
from .db import db_cursor, get_conn
from .llm import invoke_json

EXPORT_DIR = config.DATA_DIR / "exports"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------- 会话管理 ----------------

def create_session(user_id: str, job_type: str, difficulty: str, stress_mode: bool) -> str:
    sid = str(uuid.uuid4())
    now = _now()
    with db_cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO interview_sessions (id,user_id,job_type,difficulty,stress_mode,status,"
            "started_at) VALUES (?,?,?,?,?,?,'ongoing',?)",
            (sid, user_id, job_type, difficulty, 1 if stress_mode else 0, now),
        )
    return sid


def get_session(session_id: str) -> dict | None:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM interview_sessions WHERE id=?", (session_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_sessions(user_id: str) -> list[dict]:
    conn = get_conn()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM interview_sessions WHERE user_id=? ORDER BY started_at DESC",
            (user_id,),
        ).fetchall()]
    finally:
        conn.close()


def get_messages(session_id: str) -> list[dict]:
    conn = get_conn()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM interview_messages WHERE session_id=? ORDER BY rowid",
            (session_id,),
        ).fetchall()]
    finally:
        conn.close()


def answered_count(session_id: str) -> int:
    conn = get_conn()
    try:
        return conn.execute(
            "SELECT COUNT(*) c FROM interview_messages WHERE session_id=? AND user_answer!=''",
            (session_id,),
        ).fetchone()["c"]
    finally:
        conn.close()


# ---------------- 出题 ----------------

def _random_seed_block(user_id: str) -> dict | None:
    chunks = knowledge.user_ready_chunks(user_id)
    if not chunks:
        return None
    return random.choice(chunks)


def _question_prompt(session: dict, block: dict, index: int) -> str:
    stress = bool(session["stress_mode"])
    tone = (
        "这是压力面试，请出更有挑战性、容易暴露一知半解的题，角度刁钻一些。"
        if stress else "题目贴近真实校招/社招面试，难度适中。"
    )
    return (
        f"你是{session['job_type']}岗位的面试官，难度等级：{session['difficulty']}。"
        f"这是第 {index} 题。请严格基于以下知识库内容出一道面试题并给出标准答案。\n"
        f"知识库内容：\n{block['content']}\n\n要求：{tone}\n"
        "只输出 JSON：{\"question\":\"面试题目\",\"reference_answer\":\"基于知识库的标准答案\","
        "\"question_type\":\"技术知识/项目经历/行为面试/算法编码\"}"
    )


def _followup_prompt(session: dict, prev: dict) -> str:
    return (
        "你正在进行压力面试。考生上一题的回答如下，请针对其中的细节提出一个尖锐的深挖追问"
        "（追问原理、权衡取舍、具体数据、边界情况或失败处理），并给出参考答案。\n"
        f"上一题：{prev['question']}\n考生回答：{prev['user_answer']}\n"
        "只输出 JSON：{\"question\":\"追问\",\"reference_answer\":\"参考答案\","
        "\"question_type\":\"技术知识\"}"
    )


def next_question(user_id: str, session_id: str) -> dict:
    """出下一题；压力模式下每隔一轮基于上一回答做深挖追问。"""
    session = get_session(session_id)
    if not session or session["status"] != "ongoing":
        raise RuntimeError("面试会话不存在或已结束")
    msgs = get_messages(session_id)
    index = len(msgs) + 1

    prev = msgs[-1] if msgs else None
    if session["stress_mode"] and prev and prev["user_answer"] and index % 2 == 0:
        data = invoke_json(_followup_prompt(session, prev))
    else:
        block = _random_seed_block(user_id)
        if block is None:
            raise RuntimeError("知识库为空，请先在「知识库管理」上传资料或等待示例题库入库")
        data = invoke_json(_question_prompt(session, block, index))

    if not data or not data.get("question"):
        raise RuntimeError("题目生成失败，请重试")
    qtype = data.get("question_type", "技术知识")
    star = 1 if qtype == "行为面试" else 0
    mid = str(uuid.uuid4())
    with db_cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO interview_messages (id,session_id,role,question,reference_answer,"
            "user_answer,score,feedback,star_hint,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (mid, session_id, "interviewer", data["question"],
             data.get("reference_answer", ""), "", None, None, star, _now()),
        )
    return {"message_id": mid, "question": data["question"],
            "reference_answer": data.get("reference_answer", ""),
            "question_type": qtype, "star_hint": bool(star)}


# ---------------- 答题评分 ----------------

def _score_prompt(question, reference, user_answer, star_hint) -> str:
    star_line = (
        "同时评估是否遵循 STAR 结构（情境-任务-行动-结果），在 star_followed 字段给出 true/false。"
        if star_hint else ""
    )
    return (
        "你是严格的面试官。请依据【标准答案】对【考生回答】打分，评分要能区分背诵与真正理解。\n"
        f"题目：{question}\n标准答案：{reference}\n考生回答：{user_answer}\n{star_line}\n"
        "只输出 JSON：{\"score\":0到100的整数,\"star_followed\":true或false,"
        "\"missing_points\":[\"遗漏的关键得分点\"],\"strengths\":[\"回答亮点\"],"
        "\"suggestions\":\"具体可执行的改进建议\"}"
    )


def submit_answer(session_id: str, user_answer: str) -> dict:
    user_answer = user_answer.strip()
    if not user_answer:
        raise ValueError("回答不能为空")
    conn = get_conn()
    target = conn.execute(
        "SELECT * FROM interview_messages WHERE session_id=? AND user_answer='' ORDER BY rowid DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    conn.close()
    if not target:
        raise RuntimeError("当前没有待回答的题目，请先出题")

    data = invoke_json(_score_prompt(
        target["question"], target["reference_answer"], user_answer, target["star_hint"]
    ))
    if not data or data.get("score") is None:
        data = {"score": 0, "missing_points": [], "strengths": [],
                "suggestions": "评分解析失败，请参考标准答案自行核对"}
    feedback = json.dumps(data, ensure_ascii=False)
    with db_cursor(commit=True) as cur:
        cur.execute(
            "UPDATE interview_messages SET user_answer=?, score=?, feedback=? WHERE id=?",
            (user_answer, float(data["score"]), feedback, target["id"]),
        )
    return {"score": float(data["score"]), "feedback": data}


# ---------------- 结束与面经导出 ----------------

def session_summary(session_id: str) -> dict:
    msgs = get_messages(session_id)
    scored = [m for m in msgs if m["score"] is not None]
    avg = round(sum(m["score"] for m in scored) / len(scored), 1) if scored else 0
    return {"rounds": len(msgs), "answered": len(scored), "avg_score": avg,
            "messages": msgs}


def end_session(session_id: str) -> dict:
    session = get_session(session_id)
    if not session:
        raise RuntimeError("会话不存在")
    now = _now()
    summary = session_summary(session_id)
    export_path = _export_report(session, summary)

    started = datetime.strptime(session["started_at"], "%Y-%m-%d %H:%M:%S")
    seconds = max(0, int((datetime.now() - started).total_seconds()))
    with db_cursor(commit=True) as cur:
        cur.execute(
            "UPDATE interview_sessions SET status='finished', ended_at=?, export_path=? WHERE id=?",
            (now, str(export_path), session_id),
        )
        cur.execute(
            "INSERT INTO study_log (id,user_id,log_date,practice_seconds,question_count,wrong_count)"
            " VALUES (?,?,date('now'),?,?,0) ON CONFLICT(user_id,log_date) DO UPDATE SET "
            "practice_seconds=practice_seconds+?, question_count=question_count+?",
            (str(uuid.uuid4()), config.DEFAULT_USER_ID, seconds, summary["answered"],
             seconds, summary["answered"]),
        )
    return {"avg_score": summary["avg_score"], "rounds": summary["rounds"],
            "export_path": str(export_path), "practice_seconds": seconds}


def _export_report(session: dict, summary: dict) -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = EXPORT_DIR / f"面经_{session['job_type']}_{stamp}.md"
    lines = [
        f"# 模拟面试面经（{session['job_type']}）",
        "",
        f"- 时间：{session['started_at']}",
        f"- 难度：{session['difficulty']} ｜ 压力模式：{'是' if session['stress_mode'] else '否'}",
        f"- 答题数：{summary['answered']} ｜ 平均得分：{summary['avg_score']}",
        "",
    ]
    weak: list[str] = []
    for i, m in enumerate(summary["messages"], 1):
        lines += [f"## 第 {i} 题", "", f"**题目：** {m['question']}", ""]
        if m["reference_answer"]:
            lines += ["**参考答案：**", m["reference_answer"], ""]
        if m["user_answer"]:
            lines += ["**我的回答：**", m["user_answer"], ""]
        if m["feedback"]:
            fb = json.loads(m["feedback"])
            lines += [f"**得分：** {m['score']} / 100",
                      f"**改进建议：** {fb.get('suggestions','')}", ""]
            weak += fb.get("missing_points", [])
    if weak:
        lines += ["## 薄弱点复习清单", ""] + [f"- {w}" for w in weak]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
