"""初始化：建表 + 首次导入示例题库 + 构建知识图谱。可重复执行（幂等）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, graph, knowledge  # noqa: E402
from app.db import init_db  # noqa: E402


def main() -> None:
    print("1/3 创建数据库表...")
    init_db()
    if knowledge.count_documents(config.DEFAULT_USER_ID) == 0:
        print("2/3 首次启动，导入内置示例题库（需要本地 Ollama 嵌入模型，可能需几分钟）...")
        n = knowledge.seed_documents(config.DEFAULT_USER_ID)
        print(f"      已导入 {n} 份示例资料")
    else:
        print("2/3 知识库已有文档，跳过示例导入")
    print("3/3 构建知识图谱...")
    r = graph.build_graph(config.DEFAULT_USER_ID)
    print(f"      图谱：{r['nodes']} 个节点 / {r['edges']} 条关系")
    print("初始化完成。")


if __name__ == "__main__":
    main()
