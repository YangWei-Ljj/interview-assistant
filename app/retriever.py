"""混合检索器：BM25 关键词 + 向量语义（归一化加权 6:4）+ 知识图谱扩展，含 L1/L2 缓存与问题分类。"""
from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict

from rank_bm25 import BM25Okapi

from . import config, graph, knowledge

# L1 进程内 LRU 缓存
_L1: OrderedDict[str, list[dict]] = OrderedDict()


# ---------------- 缓存 ----------------

def _cache_key(user_id: str, question: str) -> str:
    return hashlib.md5(f"{user_id}::{question.strip()}".encode("utf-8")).hexdigest()


def _load_l2() -> dict:
    if not config.L2_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(config.L2_CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_l2(key: str, value: list[dict]) -> None:
    l2 = _load_l2()
    l2[key] = value
    if len(l2) > 1000:  # 防止磁盘缓存无限增长，保留较新的 1000 条
        l2 = dict(list(l2.items())[-1000:])
    try:
        config.L2_CACHE_PATH.write_text(json.dumps(l2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def clear_cache() -> None:
    _L1.clear()
    if config.L2_CACHE_PATH.exists():
        config.L2_CACHE_PATH.unlink()


def _cache_get(key: str):
    if key in _L1:
        _L1.move_to_end(key)
        return _L1[key], "L1"
    l2 = _load_l2()
    if key in l2:
        val = l2[key]
        _cache_put_l1(key, val)
        return val, "L2"
    return None, None


def _cache_put_l1(key: str, value: list[dict]) -> None:
    _L1[key] = value
    _L1.move_to_end(key)
    if len(_L1) > config.L1_MAX_ENTRIES:
        _L1.popitem(last=False)


# ---------------- 分词与 BM25 ----------------

def tokenize(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9]+", text)
    han = re.findall(r"[一-鿿]", text)
    bigrams = [han[i] + han[i + 1] for i in range(len(han) - 1)]
    return words + bigrams


def _bm25_channel(chunks: list[dict], question: str) -> dict[str, float]:
    if not chunks:
        return {}
    corpus = [tokenize(c["content"]) for c in chunks]
    bm25 = BM25Okapi(corpus)
    q_tokens = tokenize(question)
    scores = bm25.get_scores(q_tokens)
    q_set = set(q_tokens)
    raw: dict[str, float] = {}
    for c, sc, toks in zip(chunks, scores, corpus):
        if sc < config.BM25_MIN_SCORE:
            continue
        if len(q_set & set(toks)) < config.BM25_MIN_MATCHED_TOKENS:
            continue
        raw[c["chunk_id"]] = float(sc)
    return raw


def _normalize_minmax(raw: dict[str, float]) -> dict[str, float]:
    if not raw:
        return {}
    if len(raw) == 1:
        return {k: 1.0 for k in raw}
    lo, hi = min(raw.values()), max(raw.values())
    if hi == lo:
        return {k: 1.0 for k in raw}
    return {k: (v - lo) / (hi - lo) for k, v in raw.items()}


# ---------------- 向量通道 ----------------

def _vector_channel(question: str, user_id: str) -> dict[str, float]:
    store = knowledge.get_vector_store()
    flt = {"user_id": user_id}
    pairs = store.similarity_search_with_score(question, k=config.RETRIEVER_TOP_K * 2, filter=flt)
    dist = {
        doc.metadata.get("chunk_id"): float(d)
        for doc, d in pairs
        if float(d) <= config.SCORE_DISTANCE_THRESHOLD
    }
    if dist and config.SCORE_RELATIVE_SPREAD > 0:
        best = min(dist.values())
        dist = {k: v for k, v in dist.items() if v <= best + config.SCORE_RELATIVE_SPREAD}
    # L2 距离 → 归一化相似度（归一化向量距离约 0-2）
    return {k: max(0.0, 1.0 - v / 2.0) for k, v in dist.items()}


# ---------------- 问题类型识别（规则）----------------

_PROJECT_HINTS = ("项目", "做过", "负责", "介绍一下你", "简历", "实习", "毕业设计", "作品")
_BEHAVIOR_HINTS = (
    "为什么", "冲突", "矛盾", "压力", "团队", "合作", "困难", "失败", "加班", "职业规划",
    "优点", "缺点", "说说你", "怎么处理", "分歧", "领导", "沟通", "情绪",
)
_ALGO_HINTS = (
    "算法", "复杂度", "排序", "写一个", "实现", "代码", "leetcode", "数据结构",
    "算法题", "编程题", "刷题", "题目", "做题",
)


def classify_question(question: str) -> str:
    q = question.strip()
    if any(h in q for h in _ALGO_HINTS):
        return "算法编码"
    if any(h in q for h in _PROJECT_HINTS):
        return "项目经历"
    if any(h in q for h in _BEHAVIOR_HINTS):
        return "行为面试"
    return "技术知识"


# ---------------- 融合检索 ----------------

def search(user_id: str, question: str, use_cache: bool = True) -> dict:
    """返回 {results, cache_hit, channels, question_type}。

    融合口径：向量语义与 BM25 各自归一化到 [0,1]，按 vector_weight:bm25_weight（默认 6:4）
    加权，图谱扩展对关联块额外加分；缺失通道按 0 计，双通道命中块自然靠前。
    """
    question = question.strip()
    key = _cache_key(user_id, question)
    if use_cache and config.ENABLE_CACHE:
        cached, level = _cache_get(key)
        if cached is not None:
            return {
                "results": cached, "cache_hit": level, "channels": "cache",
                "question_type": classify_question(question),
            }

    chunks = knowledge.user_ready_chunks(user_id)
    chunk_by_id = {c["chunk_id"]: c for c in chunks}

    vec_sim = _vector_channel(question, user_id)
    bm25_norm = _normalize_minmax(_bm25_channel(chunks, question)) \
        if config.get_setting("hybrid_search_enabled") else {}

    candidates = set(vec_sim) | set(bm25_norm)
    graph_scores: dict[str, float] = {}
    # 仅以高置信度块（向量相似度≥0.5，即距离≤1.0）作为图谱种子，避免边缘误召回被图扩展放大
    strong_ids = [cid for cid in candidates if vec_sim.get(cid, 0.0) >= 0.5]
    if config.get_setting("graph_search_enabled") and strong_ids:
        graph_scores = graph.graph_expand(user_id, strong_ids)

    fused: dict[str, float] = {}
    for cid in candidates | set(graph_scores):
        score = (
            config.VECTOR_WEIGHT * vec_sim.get(cid, 0.0)
            + config.BM25_WEIGHT * bm25_norm.get(cid, 0.0)
            + graph_scores.get(cid, 0.0)
        )
        fused[cid] = round(score, 5)

    ordered_ids = sorted(fused, key=lambda k: fused[k], reverse=True)[: config.RETRIEVER_TOP_K]
    results = []
    for cid in ordered_ids:
        base = chunk_by_id.get(cid)
        if not base:
            continue
        results.append(
            {
                "chunk_id": cid,
                "content": base["content"],
                "source": base["source"],
                "title": base.get("title", ""),
                "meta_type": base.get("meta_type", "text"),
                "doc_id": base["doc_id"],
                "vector_sim": round(vec_sim.get(cid, 0.0), 4),
                "bm25_norm": round(bm25_norm.get(cid, 0.0), 4),
                "graph_score": graph_scores.get(cid, 0.0),
                "final_score": fused[cid],
            }
        )

    if config.ENABLE_CACHE and results:
        _cache_put_l1(key, results)
        _write_l2(key, results)

    return {
        "results": results,
        "cache_hit": None,
        "channels": f"vector({len(vec_sim)})+bm25({len(bm25_norm)})+graph({len(graph_scores)})",
        "question_type": classify_question(question),
    }
