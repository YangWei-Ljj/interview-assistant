"""知识库服务：结构化分块、入库（块级去重）、增量更新、文件夹/标签/归档、检索块查询与重建。"""
from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .db import db_cursor, get_conn
from .document_loader import DocumentError, parse_upload
from .llm import get_embeddings


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ==================== 结构化分块 ====================

_HEADING_RE = re.compile(
    r"^(#{1,6}\s+.+|"
    r"第[一二三四五六七八九十百\d]+[章节].*|"
    r"[一二三四五六七八九十]+、.*|"
    r"\d+(\.\d+)*[.、]\s*\S+.*)$"
)
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def _clean_heading(s: str) -> str:
    return re.sub(r"^#{1,6}\s+", "", s).strip()


def _split_by_lines(content: str, meta_type: str, chunk_size: int, heading: str) -> list[tuple]:
    """代码/表格等结构化块按行切分，尽量不破坏结构。"""
    if len(content) <= chunk_size:
        return [(content, meta_type, heading)]
    out, buf, size = [], [], 0
    for line in content.split("\n"):
        if buf and size + len(line) > chunk_size:
            out.append(("\n".join(buf), meta_type, heading))
            buf, size = [], 0
        buf.append(line)
        size += len(line) + 1
    if buf:
        out.append(("\n".join(buf), meta_type, heading))
    return out


def _split_plain(text: str, chunk_size: int, overlap: int, state: dict) -> list[tuple]:
    """普通文本：识别表格与标题，文本用递归分块器（带重叠）二次切分。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        length_function=len,
    )
    out: list[tuple] = []
    text_buf: list[str] = []
    table_buf: list[str] = []

    def flush_table() -> None:
        if table_buf:
            out.extend(_split_by_lines("\n".join(table_buf), "table", chunk_size, state["heading"]))
            table_buf.clear()

    def flush_text() -> None:
        if text_buf:
            block = "\n".join(text_buf).strip()
            if block:
                for piece in splitter.split_text(block):
                    if piece.strip():
                        out.append((piece, "text", state["heading"]))
            text_buf.clear()

    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("|"):
            flush_text()
            table_buf.append(line)
        else:
            flush_table()
            if s and _HEADING_RE.match(s):
                flush_text()
                state["heading"] = _clean_heading(s)
            text_buf.append(line)
    flush_table()
    flush_text()
    return out


def structured_split(text: str, chunk_size: int, overlap: int) -> list[tuple]:
    """返回 [(content, meta_type, heading)]；代码块整体保护，表格/标题/文本分别标记。"""
    state = {"heading": ""}
    out: list[tuple] = []
    cursor = 0
    for m in _CODE_FENCE_RE.finditer(text):
        out.extend(_split_plain(text[cursor:m.start()], chunk_size, overlap, state))
        out.extend(_split_by_lines(m.group(0), "code", chunk_size, state["heading"]))
        cursor = m.end()
    out.extend(_split_plain(text[cursor:], chunk_size, overlap, state))
    return out or [(text, "text", "")]


# ==================== 向量库 ====================

def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


# ==================== 文件夹 ====================

def create_folder(name: str, folder_type: str = "custom", parent_id: str | None = None) -> str:
    fid = str(uuid.uuid4())
    with db_cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO folders (id, name, folder_type, parent_id, created_at) VALUES (?,?,?,?,?)",
            (fid, name.strip(), folder_type, parent_id, _now()),
        )
    return fid


def list_folders() -> list[dict]:
    conn = get_conn()
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM folders ORDER BY created_at").fetchall()]
    finally:
        conn.close()


def delete_folder(folder_id: str) -> None:
    """删除文件夹，文档 folder_id 置空（不删除文档本身）。"""
    with db_cursor(commit=True) as cur:
        cur.execute("UPDATE documents SET folder_id=NULL WHERE folder_id=?", (folder_id,))
        cur.execute("DELETE FROM folders WHERE id=?", (folder_id,))


# ==================== 文档查询 ====================

def list_documents(user_id: str, include_archived: bool = True) -> list[dict]:
    conn = get_conn()
    try:
        sql = "SELECT * FROM documents WHERE user_id=?"
        if not include_archived:
            sql += " AND status!='archived'"
        sql += " ORDER BY created_at DESC"
        return [dict(r) for r in conn.execute(sql, (user_id,)).fetchall()]
    finally:
        conn.close()


def count_documents(user_id: str, ready_only: bool = False) -> int:
    conn = get_conn()
    try:
        sql = "SELECT COUNT(*) c FROM documents WHERE user_id=?"
        if ready_only:
            sql += " AND status='ready'"
        return conn.execute(sql, (user_id,)).fetchone()["c"]
    finally:
        conn.close()


def get_document(user_id: str, doc_id: str) -> dict | None:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM documents WHERE user_id=? AND id=?", (user_id, doc_id)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ==================== 入库 ====================

def _existing_chunk_hashes(user_id: str) -> set[str]:
    conn = get_conn()
    try:
        return {
            r["chunk_hash"]
            for r in conn.execute(
                "SELECT c.chunk_hash FROM document_chunks c "
                "JOIN documents d ON d.id=c.doc_id WHERE d.user_id=?",
                (user_id,),
            ).fetchall()
        }
    finally:
        conn.close()


def _build_metadata(doc_id, chunk_id, filename, title, folder_id, tags, meta_type, heading, version):
    return {
        "user_id": config.DEFAULT_USER_ID,
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "source": filename,
        "title": title or "",
        "folder_id": folder_id or "",
        "tags": tags or "",
        "meta_type": meta_type,
        "heading": heading or "",
        "version": version,
    }


def ingest_text(
    user_id: str,
    filename: str,
    text: str,
    title: str | None = None,
    folder_id: str | None = None,
    tags: str = "",
    doc_type: str = "text",
) -> dict:
    """入库文本。同名文档内容变化走增量更新；全新文档块级去重后嵌入。"""
    text = text.strip()
    if not text:
        raise DocumentError("文档内容为空")
    file_hash = _md5(text)

    conn = get_conn()
    same = conn.execute(
        "SELECT id FROM documents WHERE user_id=? AND file_hash=?", (user_id, file_hash)
    ).fetchone()
    if same:
        conn.close()
        return {"status": "skipped", "doc_id": same["id"], "added": 0, "removed": 0, "kept": 0,
                "message": "文档内容未变化，跳过"}
    same_name = conn.execute(
        "SELECT id FROM documents WHERE user_id=? AND filename=?", (user_id, filename)
    ).fetchone()
    conn.close()
    if same_name:  # 同名文件内容已变 → 增量更新
        return _apply_update(same_name["id"], text, folder_id, tags)

    doc_id = str(uuid.uuid4())
    now = _now()
    title = (title or Path(filename).stem).strip()
    raw_chunks = structured_split(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    with db_cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO documents (id,user_id,filename,file_hash,title,folder_id,tags,doc_type,"
            "summary,version,chunk_count,status,error_message,created_at,updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (doc_id, user_id, filename, file_hash, title, folder_id, tags, doc_type,
             text[:200].replace("\n", " "), 1, 0, "processing", "", now, now),
        )

    known = _existing_chunk_hashes(user_id)
    prepared = []
    for content, meta_type, heading in raw_chunks:
        chash = _md5(content)
        if chash in known:
            continue
        cid = str(uuid.uuid4())
        meta = _build_metadata(doc_id, cid, filename, title, folder_id, tags, meta_type, heading, 1)
        prepared.append((cid, content, meta))

    try:
        if prepared:
            get_vector_store().add_texts(
                [c for _, c, _ in prepared],
                metadatas=[m for _, _, m in prepared],
                ids=[cid for cid, _, _ in prepared],
            )
        with db_cursor(commit=True) as cur:
            for idx, (cid, content, meta) in enumerate(prepared):
                cur.execute(
                    "INSERT INTO document_chunks (id,doc_id,chunk_index,content,chunk_hash,meta_type,created_at)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (cid, doc_id, idx, content, _md5(content), meta["meta_type"], now),
                )
            cur.execute(
                "UPDATE documents SET status='ready',chunk_count=?,updated_at=? WHERE id=?",
                (len(prepared), now, doc_id),
            )
    except Exception as exc:  # noqa: BLE001
        with db_cursor(commit=True) as cur:
            cur.execute(
                "UPDATE documents SET status='failed',error_message=?,updated_at=? WHERE id=?",
                (str(exc)[:500], now, doc_id),
            )
        raise DocumentError(f"文档入库失败：{exc}") from exc

    return {
        "status": "ok", "doc_id": doc_id,
        "added": len(prepared), "removed": 0, "kept": 0,
        "message": f"已入库 {len(prepared)} 个内容块（共 {len(raw_chunks)} 块，"
                   f"{len(raw_chunks)-len(prepared)} 块重复已跳过）",
    }


def ingest_file(user_id, filename, raw, title=None, folder_id=None, tags="") -> dict:
    parsed = parse_upload(filename, raw)
    return ingest_text(
        user_id, filename, parsed["text"],
        title=title, folder_id=folder_id, tags=tags, doc_type=parsed["doc_type"],
    )


def _apply_update(doc_id: str, text: str, folder_id, tags) -> dict:
    """增量更新：按块 hash 保留未变块，删除消失块，新增新块，版本号 +1。"""
    conn = get_conn()
    doc = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    old_rows = conn.execute(
        "SELECT id, chunk_hash FROM document_chunks WHERE doc_id=?", (doc_id,)
    ).fetchall()
    conn.close()
    old_by_hash = {r["chunk_hash"]: r["id"] for r in old_rows}

    new_chunks = structured_split(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    new_by_hash = {_md5(c): (c, mt, h) for c, mt, h in new_chunks}

    kept_hashes = set(old_by_hash) & set(new_by_hash)
    remove_ids = [cid for h, cid in old_by_hash.items() if h not in new_by_hash]
    add_items = [(h, *new_by_hash[h]) for h in new_by_hash if h not in old_by_hash]

    now = _now()
    store = get_vector_store()
    if remove_ids:
        store.delete(ids=remove_ids)
        left = store.get(ids=remove_ids)
        if left and left.get("ids"):
            store.delete(ids=remove_ids)  # 重试一次

    added = []
    for chash, content, meta_type, heading in add_items:
        cid = str(uuid.uuid4())
        meta = _build_metadata(
            doc_id, cid, doc["filename"], doc["title"],
            folder_id if folder_id is not None else doc["folder_id"],
            tags or doc["tags"], meta_type, heading, doc["version"] + 1,
        )
        added.append((cid, content, meta))
    if added:
        store.add_texts(
            [c for _, c, _ in added],
            metadatas=[m for _, _, m in added],
            ids=[cid for cid, _, _ in added],
        )

    with db_cursor(commit=True) as cur:
        if remove_ids:
            placeholders = ",".join("?" * len(remove_ids))
            cur.execute(
                f"DELETE FROM document_chunks WHERE doc_id=? AND id IN ({placeholders})",
                [doc_id, *remove_ids],
            )
        for cid, content, meta in added:
            cur.execute(
                "INSERT INTO document_chunks (id,doc_id,chunk_index,content,chunk_hash,meta_type,created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (cid, doc_id, 0, content, _md5(content), meta["meta_type"], now),
            )
        # 重新统一编号
        all_rows = cur.execute(
            "SELECT id FROM document_chunks WHERE doc_id=? ORDER BY rowid", (doc_id,)
        ).fetchall()
        for idx, r in enumerate(all_rows):
            cur.execute("UPDATE document_chunks SET chunk_index=? WHERE id=?", (idx, r["id"]))
        cur.execute(
            "UPDATE documents SET version=version+1, chunk_count=?, summary=?, folder_id=?,"
            " tags=?, status='ready', error_message='', updated_at=? WHERE id=?",
            (len(all_rows), text[:200].replace("\n", " "),
             folder_id if folder_id is not None else doc["folder_id"],
             tags or doc["tags"], now, doc_id),
        )

    return {
        "status": "updated", "doc_id": doc_id,
        "added": len(added), "removed": len(remove_ids), "kept": len(kept_hashes),
        "message": f"增量更新完成：新增 {len(added)} 块、删除 {len(remove_ids)} 块、"
                   f"保留 {len(kept_hashes)} 块（版本 v{doc['version']+1}）",
    }


# ==================== 删除 / 归档 / 移动 ====================

def delete_document(user_id: str, doc_id: str) -> None:
    conn = get_conn()
    doc = conn.execute(
        "SELECT id FROM documents WHERE user_id=? AND id=?", (user_id, doc_id)
    ).fetchone()
    if not doc:
        conn.close()
        return
    chunk_ids = [r["id"] for r in conn.execute(
        "SELECT id FROM document_chunks WHERE doc_id=?", (doc_id,)
    ).fetchall()]
    conn.close()

    if chunk_ids:
        store = get_vector_store()
        for _ in range(2):
            store.delete(ids=chunk_ids)
            left = store.get(ids=chunk_ids)
            if not left or not left.get("ids"):
                break
        else:
            raise RuntimeError("向量库删除未完全，残留块；请运行 rebuild_index")
    with db_cursor(commit=True) as cur:
        cur.execute("DELETE FROM document_chunks WHERE doc_id=?", (doc_id,))
        cur.execute("DELETE FROM documents WHERE id=?", (doc_id,))


def set_archive(user_id: str, doc_ids: list[str], archived: bool) -> int:
    status = "archived" if archived else "ready"
    with db_cursor(commit=True) as cur:
        for did in doc_ids:
            cur.execute(
                "UPDATE documents SET status=?, updated_at=? WHERE user_id=? AND id=?",
                (status, _now(), user_id, did),
            )
    return len(doc_ids)


def move_documents(user_id: str, doc_ids: list[str], folder_id: str | None) -> int:
    with db_cursor(commit=True) as cur:
        for did in doc_ids:
            cur.execute(
                "UPDATE documents SET folder_id=?, updated_at=? WHERE user_id=? AND id=?",
                (folder_id, _now(), user_id, did),
            )
    return len(doc_ids)


# ==================== 检索块查询（仅 ready 文档）====================

def user_ready_chunks(user_id: str, folder_id: str | None = None) -> list[dict]:
    conn = get_conn()
    try:
        sql = (
            "SELECT c.id AS chunk_id, c.content, c.chunk_index, c.meta_type, d.filename AS source,"
            " d.title AS title, d.id AS doc_id, d.folder_id, d.tags"
            " FROM document_chunks c JOIN documents d ON d.id=c.doc_id"
            " WHERE d.user_id=? AND d.status='ready'"
        )
        params: list = [user_id]
        if folder_id:
            sql += " AND d.folder_id=?"
            params.append(folder_id)
        sql += " ORDER BY d.id, c.chunk_index"
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


# ==================== 重建 / 统计 / 种子 ====================

def rebuild_index() -> int:
    store = get_vector_store()
    try:
        store.delete_collection()
    except Exception:  # noqa: BLE001
        pass
    store = get_vector_store()
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT c.id AS chunk_id, c.content, c.meta_type, d.filename AS source, d.title AS title,"
            " d.id AS doc_id, d.user_id, d.folder_id, d.tags, d.version"
            " FROM document_chunks c JOIN documents d ON d.id=c.doc_id WHERE d.status='ready'"
        ).fetchall()
        data = [dict(r) for r in rows]
    finally:
        conn.close()
    if data:
        store.add_texts(
            [c["content"] for c in data],
            metadatas=[
                {
                    "user_id": c["user_id"], "doc_id": c["doc_id"], "chunk_id": c["chunk_id"],
                    "source": c["source"], "title": c["title"] or "",
                    "folder_id": c["folder_id"] or "", "tags": c["tags"] or "",
                    "meta_type": c["meta_type"], "heading": "", "version": c["version"],
                }
                for c in data
            ],
            ids=[c["chunk_id"] for c in data],
        )
    return len(data)


def index_stats() -> dict:
    conn = get_conn()
    try:
        total = conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
        ready = conn.execute("SELECT COUNT(*) c FROM documents WHERE status='ready'").fetchone()["c"]
        archived = conn.execute("SELECT COUNT(*) c FROM documents WHERE status='archived'").fetchone()["c"]
        failed = conn.execute("SELECT COUNT(*) c FROM documents WHERE status='failed'").fetchone()["c"]
        chunks = conn.execute("SELECT COUNT(*) c FROM document_chunks").fetchone()["c"]
    finally:
        conn.close()
    try:
        vec_count = len(get_vector_store().get().get("ids", []))
    except Exception:  # noqa: BLE001
        vec_count = -1
    return {
        "documents": total, "ready": ready, "archived": archived, "failed": failed,
        "chunks_sqlite": chunks, "chunks_vector": vec_count,
        "consistent": vec_count == chunks,
    }


def seed_documents(user_id: str) -> int:
    """把 seed_docs 下示例资料幂等入库。"""
    n = 0
    for f in sorted(Path(config.SEED_DOC_DIR).glob("*")):
        if f.suffix.lower() not in (".txt", ".md"):
            continue
        r = ingest_text(user_id, f.name, f.read_text(encoding="utf-8"), doc_type="seed")
        if r["status"] in ("ok", "updated"):
            n += 1
    return n
