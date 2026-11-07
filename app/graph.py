"""轻量知识图谱：基于术语抽取与共现关系，无需独立图数据库（节点/边存 SQLite）。

- 节点：英文技术词、代码标识符、标题中的中文知识点；
- 边：同一内容块内术语共现，权重按共现次数累加；
- 图扩展：检索命中后，沿关系边找到关联术语，再映射到相关块，作为融合加分通道。
"""
from __future__ import annotations

import re
import uuid
from collections import defaultdict
from itertools import combinations

from . import config, knowledge
from .db import db_cursor, get_conn

_EN_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#._\-]{1,}")
# 代码标识符（def/class/import/from 后的名称）
_CODE_ID_RE = re.compile(
    r"(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)|(?:from|import)\s+([A-Za-z_][A-Za-z0-9_.]*)"
)

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "of", "to", "in",
    "on", "for", "and", "or", "not", "no", "if", "then", "else", "elif", "this", "that",
    "these", "those", "it", "its", "with", "as", "by", "can", "could", "will", "would",
    "should", "may", "might", "must", "shall", "we", "you", "they", "he", "she", "i",
    "use", "used", "using", "uses", "do", "does", "did", "done", "have", "has", "had",
    "http", "https", "www", "com", "html", "true", "false", "none", "null", "self",
    "print", "return", "import", "from", "def", "class", "len", "str", "int", "list",
    "dict", "set", "get", "add", "new", "one", "all", "any", "but", "so", "at", "up",
}


def _now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def extract_terms(content: str, heading: str = "", meta_type: str = "text") -> list[str]:
    terms: set[str] = set()
    for raw in _EN_TERM_RE.findall(content):
        w = raw.strip("._-")
        if len(w) < 2 or w.lower() in _STOP or w.isdigit():
            continue
        terms.add(w)
    for g1, g2 in _CODE_ID_RE.findall(content):
        name = (g1 or g2).split(".")[0]
        if name and name.lower() not in _STOP:
            terms.add(name)
    # 标题中的中文知识点（限长，避免整句）
    h = (heading or "").strip()
    if h and len(h) <= 20 and not h.startswith(("|", "```")):
        terms.add(h)
    return sorted(terms)


def build_graph(user_id: str) -> dict:
    """扫描全部 ready 块，（重）建节点与共现边。增量入库后可再次调用以补齐。"""
    chunks = knowledge.user_ready_chunks(user_id)
    all_terms: set[str] = set()
    chunk_terms: list[set[str]] = []
    for c in chunks:
        ts = set(extract_terms(c["content"], c.get("title") or "", c["meta_type"]))
        chunk_terms.append(ts)
        all_terms |= ts

    now = _now()
    with db_cursor(commit=True) as cur:
        # 清空旧图谱后重建（单设备、数据量可控，保证与当前文档一致）
        cur.execute("DELETE FROM graph_edges")
        cur.execute("DELETE FROM graph_nodes")
        node_id: dict[str, str] = {}
        for term in sorted(all_terms):
            nid = str(uuid.uuid4())
            node_id[term] = nid
            cur.execute(
                "INSERT INTO graph_nodes (id,name,node_type,created_at) VALUES (?,?,?,?)",
                (nid, term, "concept", now),
            )

        edge_weight: dict[tuple, float] = defaultdict(float)
        edge_doc: dict[tuple, str] = {}
        for c, ts in zip(chunks, chunk_terms):
            ordered = sorted(ts)
            for a, b in combinations(ordered, 2):
                key = (node_id[a], node_id[b])
                edge_weight[key] += 1.0
                edge_doc.setdefault(key, c["doc_id"])
        for (src, tgt), w in edge_weight.items():
            cur.execute(
                "INSERT INTO graph_edges (id,source_id,target_id,relation,weight,doc_id,created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), src, tgt, "co-occur", w, edge_doc[(src, tgt)], now),
            )

    return {"nodes": len(all_terms), "edges": len(edge_weight), "documents": len(chunks)}


def graph_stats() -> dict:
    conn = get_conn()
    try:
        return {
            "nodes": conn.execute("SELECT COUNT(*) c FROM graph_nodes").fetchone()["c"],
            "edges": conn.execute("SELECT COUNT(*) c FROM graph_edges").fetchone()["c"],
        }
    finally:
        conn.close()


def _neighbor_terms(terms: list[str], limit: int = 15) -> list[str]:
    conn = get_conn()
    try:
        placeholders = ",".join("?" * len(terms))
        rows = conn.execute(
            f"SELECT n2.name AS name, e.weight AS w FROM graph_nodes n1 "
            f"JOIN graph_edges e ON e.source_id=n1.id OR e.target_id=n1.id "
            f"JOIN graph_nodes n2 ON n2.id=CASE WHEN e.source_id=n1.id THEN e.target_id "
            f"ELSE e.source_id END WHERE n1.name IN ({placeholders}) "
            f"ORDER BY e.weight DESC LIMIT ?",
            [*terms, limit],
        ).fetchall()
        return [r["name"] for r in rows if r["name"] not in terms]
    finally:
        conn.close()


def graph_expand(user_id: str, hit_chunk_ids: list[str], top_n: int = 8) -> dict[str, float]:
    """对命中块做图扩展，返回 {chunk_id: 图加分}（仅返回未直接命中的关联块）。"""
    if not hit_chunk_ids:
        return {}
    chunks = knowledge.user_ready_chunks(user_id)
    hit_set = set(hit_chunk_ids)
    seed_terms: set[str] = set()
    for c in chunks:
        if c["chunk_id"] in hit_set:
            seed_terms |= set(extract_terms(c["content"], c.get("title") or "", c["meta_type"]))
    if not seed_terms:
        return {}
    neighbors = _neighbor_terms(sorted(seed_terms), limit=15)
    if not neighbors:
        return {}

    scores: dict[str, float] = {}
    for c in chunks:
        if c["chunk_id"] in hit_set:
            continue
        cterms = extract_terms(c["content"], c.get("title") or "", c["meta_type"])
        overlap = len(set(neighbors) & set(cterms))
        if overlap:
            scores[c["chunk_id"]] = round(config.GRAPH_BOOST * overlap, 4)
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n])


def list_relations(limit: int = 50) -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT n1.name AS source, n2.name AS target, e.weight, e.relation "
            "FROM graph_edges e JOIN graph_nodes n1 ON n1.id=e.source_id "
            "JOIN graph_nodes n2 ON n2.id=e.target_id ORDER BY e.weight DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
