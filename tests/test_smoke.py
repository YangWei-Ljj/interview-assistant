"""冒烟测试：纯函数逻辑（不依赖网络、不污染生产数据库）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import knowledge, retriever  # noqa: E402
from app.llm import parse_json_block  # noqa: E402


def test_structured_split_detects_code_and_table():
    text = "普通说明文字。\n\n```\nprint('hi')\n```\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n"
    chunks = knowledge.structured_split(text, chunk_size=500, overlap=50)
    types = {mt for _, mt, _ in chunks}
    assert "code" in types
    assert "table" in types


def test_heading_is_tracked():
    text = "# 重要标题\n这是标题下的正文内容，需要继承标题。"
    chunks = knowledge.structured_split(text, 500, 50)
    assert any(h == "重要标题" for _, _, h in chunks)


def test_classify_question():
    assert retriever.classify_question("讲一下快速排序的时间复杂度") == "算法编码"
    assert retriever.classify_question("介绍一下你做过的项目") == "项目经历"
    assert retriever.classify_question("你怎么处理团队冲突") == "行为面试"
    assert retriever.classify_question("什么是依赖注入") == "技术知识"


def test_tokenize_chinese_bigram():
    toks = retriever.tokenize("检索")
    assert "检索" in toks  # 中文二元


def test_parse_json_block():
    assert parse_json_block('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_block('结果 {"b": 2} 结束') == {"b": 2}
    assert parse_json_block("没有 json") is None
