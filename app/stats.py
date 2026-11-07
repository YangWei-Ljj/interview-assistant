"""统计分析：高频问题榜、知识盲区、学习仪表盘数据（供 Plotly 可视化）。"""
from __future__ import annotations

import json
from collections import Counter

from . import config
from .db import get_conn


def top_questions(user_id: str, limit: int = 10) -> list[dict]:
    """高频问题排行榜：询问次数、最后访问时间、平均评分。"""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT question, COUNT(*) AS times, MAX(created_at) AS last_time, "
            "AVG(rating) AS avg_rating, SUM(marked_wrong) AS wrong_times "
            "FROM qa_records WHERE user_id=? GROUP BY question ORDER BY times DESC, last_time DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def weak_points(user_id: str) -> dict:
    """知识盲区：被标记错误的问题、面试遗漏的得分点、查询多且评分低的知识点。"""
    conn = get_conn()
    wrong_qa = [dict(r) for r in conn.execute(
        "SELECT question, created_at FROM qa_records WHERE user_id=? AND marked_wrong=1 "
        "ORDER BY created_at DESC", (user_id,),
    ).fetchall()]

    missing = Counter()
    rows = conn.execute(
        "SELECT m.feedback FROM interview_messages m JOIN interview_sessions s ON s.id=m.session_id "
        "WHERE s.user_id=? AND m.feedback IS NOT NULL", (user_id,),
    ).fetchall()
    for r in rows:
        try:
            fb = json.loads(r["feedback"])
            for p in fb.get("missing_points", []):
                missing[p] += 1
        except (json.JSONDecodeError, TypeError):
            continue
    conn.close()
    return {
        "wrong_qa": wrong_qa,
        "missing_points": [{"point": k, "times": v} for k, v in missing.most_common(15)],
        "review_list": [w["question"] for w in wrong_qa][:15]
        + [k for k, _ in missing.most_common(10)],
    }


def mastery(user_id: str) -> dict:
    """掌握度：按面试评分分档（>=80 掌握 / 60-79 一般 / <60 薄弱）。"""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT m.score FROM interview_messages m JOIN interview_sessions s ON s.id=m.session_id "
            "WHERE s.user_id=? AND m.score IS NOT NULL", (user_id,),
        ).fetchall()
    finally:
        conn.close()
    scores = [r["score"] for r in rows]
    mastered = sum(1 for s in scores if s >= 80)
    average = sum(1 for s in scores if 60 <= s < 80)
    weak = sum(1 for s in scores if s < 60)
    pct = round(mastered / len(scores) * 100, 1) if scores else 0.0
    return {"total": len(scores), "mastered": mastered, "average": average, "weak": weak,
            "mastery_pct": pct, "scores": scores}


def daily_trend(user_id: str, days: int = 14) -> list[dict]:
    """近 N 天学习日志：练习时长、题数、错题数与当日错题率。"""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT log_date, practice_seconds, question_count, wrong_count FROM study_log "
            "WHERE user_id=? ORDER BY log_date DESC LIMIT ?", (user_id, days),
        ).fetchall()
    finally:
        conn.close()
    out = []
    for r in reversed(rows):
        d = dict(r)
        total = d["question_count"] or 0
        d["wrong_rate"] = round(d["wrong_count"] / total * 100, 1) if total else 0.0
        d["practice_minutes"] = round(d["practice_seconds"] / 60, 1)
        out.append(d)
    return out


def overview(user_id: str) -> dict:
    conn = get_conn()
    try:
        total_qa = conn.execute(
            "SELECT COUNT(*) c FROM qa_records WHERE user_id=?", (user_id,)
        ).fetchone()["c"]
        total_practice = conn.execute(
            "SELECT COALESCE(SUM(practice_seconds),0) s FROM study_log WHERE user_id=?", (user_id,)
        ).fetchone()["s"]
        wrong_total = conn.execute(
            "SELECT COUNT(*) c FROM qa_records WHERE user_id=? AND marked_wrong=1", (user_id,)
        ).fetchone()["c"]
        interviews = conn.execute(
            "SELECT COUNT(*) c FROM interview_sessions WHERE user_id=?", (user_id,)
        ).fetchone()["c"]
    finally:
        conn.close()
    return {
        "total_qa": total_qa,
        "interviews": interviews,
        "wrong_total": wrong_total,
        "practice_minutes": round(total_practice / 60, 1),
    }
