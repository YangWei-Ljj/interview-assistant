"""端到端自检：离题拒答、库内问答全链路、缓存命中。运行：python scripts/smoke_qa.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, qa, retriever  # noqa: E402

uid = config.DEFAULT_USER_ID

print("== 1. 离题问题：期望检索为空（拒答路径，不落库） ==")
off = retriever.search(uid, "今天杭州天气怎么样，适合出门吗", use_cache=False)
print(f"   命中块数：{len(off['results'])}（应为 0）")
assert len(off["results"]) == 0, "离题问题不应召回知识库内容"

print("== 2. 库内问题：完整问答（检索+生成+三重校验） ==")
r1 = qa.answer_question(uid, "什么是 RAG？它解决了什么问题？")
print(f"   类型：{r1['question_type']} ｜ 来源数：{len(r1['sources'])} ｜ "
      f"耗时：{r1['latency_ms']}ms ｜ 缓存：{r1['cache_hit']}")
print(f"   校验：{r1['check'].get('notes','')}")
print(f"   答案预览：{r1['answer'][:120]} ...")
assert r1["sources"], "库内问题应召回资料"

print("== 3. 重复同一问题：期望命中缓存 ==")
r2 = qa.answer_question(uid, "什么是 RAG？它解决了什么问题？")
print(f"   缓存命中：{r2['cache_hit']} ｜ 耗时：{r2['latency_ms']}ms")
assert r2["cache_hit"] in ("L1", "L2"), "重复问题应命中缓存"

print("\n全部端到端自检通过。")
