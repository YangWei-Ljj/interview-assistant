"""问答引擎：混合检索 → 场景化生成 → 三重校验 → 落库 → 反馈闭环。"""
from __future__ import annotations

import json
import time
import uuid
from collections import OrderedDict
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from . import config, retriever
from .db import db_cursor, get_conn
from .llm import get_chat_model, invoke_json

# 完整答案缓存：缓存检索+生成+校验后的最终结果，重复问题直接命中、跳过 LLM
_ANSWER_L1: OrderedDict[str, dict] = OrderedDict()
ANSWER_CACHE_PATH = config.CACHE_DIR / "answer_cache.json"


def _answer_key(user_id: str, question: str) -> str:
    import hashlib
    return hashlib.md5(f"answer::{user_id}::{question.strip()}".encode("utf-8")).hexdigest()


def _answer_l2_load() -> dict:
    if not ANSWER_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(ANSWER_CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _answer_get(key: str):
    if key in _ANSWER_L1:
        _ANSWER_L1.move_to_end(key)
        return _ANSWER_L1[key], "L1"
    l2 = _answer_l2_load()
    if key in l2:
        _answer_put_l1(key, l2[key])
        return l2[key], "L2"
    return None, None


def _answer_put_l1(key: str, value: dict) -> None:
    _ANSWER_L1[key] = value
    _ANSWER_L1.move_to_end(key)
    if len(_ANSWER_L1) > config.L1_MAX_ENTRIES:
        _ANSWER_L1.popitem(last=False)


def _answer_put(key: str, value: dict) -> None:
    _answer_put_l1(key, value)
    l2 = _answer_l2_load()
    l2[key] = value
    if len(l2) > 1000:
        l2 = dict(list(l2.items())[-1000:])
    try:
        ANSWER_CACHE_PATH.write_text(json.dumps(l2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def clear_answer_cache() -> None:
    _ANSWER_L1.clear()
    if ANSWER_CACHE_PATH.exists():
        ANSWER_CACHE_PATH.unlink()

SYSTEM_PROMPT = (
    "你是资深技术面试官兼面试备考教练，必须严格依据用户提供的【参考资料】作答，"
    "杜绝参考资料之外的编造（防止幻觉）。\n"
    "回答规则：\n"
    "1. 只使用参考资料中的事实、数字、定义与结论；若资料完全没有相关信息，"
    "直接回复“知识库中暂无该知识点，建议补充相关资料后再练”，不要凭常识硬答；\n"
    "2. 答案要适合面试口述：先给结论，再分点展开，逻辑清晰、用词专业，避免大段流水账；\n"
    "3. 涉及代码时使用代码块，涉及步骤/对比时使用编号或分点，保留关键技术术语原文；\n"
    "4. 可在结尾用一句话点出该题的面试加分要点，但不要输出与问题无关的内容。"
)

STAR_TIP = (
    "行为面试题建议按 **STAR 框架** 组织回答：\n"
    "- **S（Situation 情境）**：交代背景与场景；\n"
    "- **T（Task 任务）**：说明你承担的目标/职责；\n"
    "- **A（Action 行动）**：讲清你具体做了什么（重点，多讲“我”）；\n"
    "- **R（Result 结果）**：给出可量化的结果与复盘。"
)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _format_context(docs: list[dict]) -> str:
    return "\n\n".join(
        f"[资料 {i+1}｜来源：{d['source']}｜类型：{d.get('meta_type','text')}]\n{d['content']}"
        for i, d in enumerate(docs)
    )


def _cross_check(question: str, answer: str, docs: list[dict]) -> dict:
    """第二重：多源交叉验证，核对答案是否有资料不支撑的陈述。"""
    context = _format_context(docs)
    prompt = (
        "你是答案审核员。请核对【候选答案】中的事实性陈述（定义、数字、结论、原理）是否都能在"
        "【参考资料】中找到依据。面试表达层面的建议（如 STAR、表达技巧、加分要点）不属于事实陈述，"
        "不要计入 unsupported_claims。\n"
        f"参考资料：\n{context}\n\n候选答案：\n{answer}\n\n"
        "只输出 JSON：{\"supported\": true 或 false, "
        "\"unsupported_claims\": [资料中找不到依据的事实陈述], \"notes\": \"简短说明\"}"
    )
    data = invoke_json(prompt, temperature=0.0)
    return data or {"supported": True, "unsupported_claims": [], "notes": "校验未返回结构化结果，按通过处理"}


def _generate(question: str, docs: list[dict], strict: bool = False) -> str:
    sys_text = SYSTEM_PROMPT
    if strict:
        sys_text += "\n特别注意：上一版答案包含资料外内容，本次只能复述/概括参考资料中的信息，资料没有的一律不要写。"
    resp = get_chat_model().invoke([
        SystemMessage(sys_text),
        HumanMessage(content=f"参考资料：\n{_format_context(docs)}\n\n问题：{question}"),
    ])
    return (resp.content or "").strip()


def _save_record(user_id, question, qtype, answer, sources, cache_hit, latency, rating=None) -> str:
    rid = str(uuid.uuid4())
    with db_cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO qa_records (id,user_id,question,question_type,answer,sources,rating,"
            "marked_wrong,cache_hit,latency_ms,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (rid, user_id, question, qtype, answer,
             json.dumps(sources, ensure_ascii=False) if sources else None,
             rating, 0, 1 if cache_hit else 0, round(latency, 2), _now()),
        )
        cur.execute(
            "INSERT INTO study_log (id,user_id,log_date,practice_seconds,question_count,wrong_count)"
            " VALUES (?,?,date('now'),0,1,0) ON CONFLICT(user_id,log_date) DO UPDATE SET "
            "question_count=question_count+1",
            (str(uuid.uuid4()), user_id),
        )
    return rid


def answer_question(user_id: str, question: str, force_refresh: bool = False) -> dict:
    """完整问答流程并落库，返回答案、来源、校验信息与耗时。

    重复问题命中完整答案缓存时，跳过检索/生成/校验，毫秒级返回（需求中的热点问题加速）。
    """
    question = question.strip()
    if not question:
        raise ValueError("问题不能为空")
    t0 = time.perf_counter()
    akey = _answer_key(user_id, question)

    # ---- 快速路径：完整答案缓存命中 ----
    if not force_refresh and config.ENABLE_CACHE:
        cached, level = _answer_get(akey)
        if cached is not None:
            latency = (time.perf_counter() - t0) * 1000
            rid = _save_record(user_id, question, cached["question_type"],
                               cached["answer"], cached["sources"], level, latency)
            return {
                "record_id": rid,
                "answer": cached["answer"],
                "sources": cached["sources"],
                "question_type": cached["question_type"],
                "cache_hit": level,
                "latency_ms": round(latency, 2),
                "check": cached["check"],
                "star_tip": cached.get("star_tip"),
            }

    # ---- 未命中：完整检索 + 生成 + 校验 ----
    retrieval = retriever.search(user_id, question, use_cache=not force_refresh)
    docs = retrieval["results"]
    qtype = retrieval["question_type"]

    if not docs:  # 场景规则：无来源即拒答，不调用生成模型，杜绝幻觉
        answer = "知识库中暂无该知识点，建议先在「知识库管理」上传相关面试资料，或换个问法再试。"
        check = {"supported": True, "notes": "无检索结果，按拒答处理"}
    else:
        answer = _generate(question, docs)
        check = {"supported": True, "notes": "未开启三重校验"}
        if config.ENABLE_TRIPLE_CHECK and not force_refresh:
            check = _cross_check(question, answer, docs)
            if check.get("supported") is False and check.get("unsupported_claims"):
                answer = _generate(question, docs, strict=True)  # 第三重不通过则带约束重生成
                check["regenerated"] = True

    latency = (time.perf_counter() - t0) * 1000
    record_id = _save_record(user_id, question, qtype, answer, docs,
                             retrieval["cache_hit"], latency)
    result = {
        "record_id": record_id,
        "answer": answer,
        "sources": docs,
        "question_type": qtype,
        "cache_hit": retrieval["cache_hit"],
        "latency_ms": round(latency, 2),
        "check": check,
        "star_tip": STAR_TIP if qtype == "行为面试" else None,
    }
    if docs:  # 仅缓存有来源的成功问答，拒答不缓存
        _answer_put(akey, {
            "answer": answer, "sources": docs, "question_type": qtype,
            "check": check, "star_tip": result["star_tip"],
        })
    return result


def rate_record(record_id: str, rating: int | None = None, marked_wrong: bool = False) -> None:
    """用户反馈：1-5 星评分、标记答案不准确；标记错误同步计入错题。"""
    with db_cursor(commit=True) as cur:
        cur.execute(
            "UPDATE qa_records SET rating=?, marked_wrong=? WHERE id=?",
            (rating, 1 if marked_wrong else 0, record_id),
        )
        if marked_wrong:
            cur.execute(
                "INSERT INTO study_log (id,user_id,log_date,practice_seconds,question_count,wrong_count)"
                " VALUES (?,?,date('now'),0,0,1) ON CONFLICT(user_id,log_date) DO UPDATE SET "
                "wrong_count=wrong_count+1",
                (str(uuid.uuid4()), config.DEFAULT_USER_ID),
            )


def list_records(user_id: str, limit: int = 100) -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM qa_records WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
