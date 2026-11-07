"""SQLite 存储层（WAL + 外键）：文件夹、文档、文档块、知识图谱、问答记录、面试、学习日志。"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from . import config

SCHEMA_SQL = """
-- 文件夹/分类体系（岗位、技术栈、面试轮次等）
CREATE TABLE IF NOT EXISTS folders (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    folder_type TEXT NOT NULL DEFAULT 'custom',
    parent_id   TEXT,
    created_at  TEXT NOT NULL
);

-- 文档
CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    filename     TEXT NOT NULL,
    file_hash    TEXT NOT NULL,
    title        TEXT NOT NULL DEFAULT '',
    folder_id    TEXT,
    tags         TEXT NOT NULL DEFAULT '',
    doc_type     TEXT NOT NULL DEFAULT 'text',
    summary      TEXT NOT NULL DEFAULT '',
    version      INTEGER NOT NULL DEFAULT 1,
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'ready',
    error_message TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    UNIQUE(user_id, file_hash)
);

-- 文档块
CREATE TABLE IF NOT EXISTS document_chunks (
    id           TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,
    content      TEXT NOT NULL,
    chunk_hash   TEXT NOT NULL,
    meta_type    TEXT NOT NULL DEFAULT 'text',
    created_at   TEXT NOT NULL,
    UNIQUE(doc_id, chunk_index)
);

-- 知识图谱节点（知识点/技术术语）
CREATE TABLE IF NOT EXISTS graph_nodes (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    node_type   TEXT NOT NULL DEFAULT 'concept',
    created_at  TEXT NOT NULL
);

-- 知识图谱边（知识点关系）
CREATE TABLE IF NOT EXISTS graph_edges (
    id          TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
    target_id   TEXT NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
    relation    TEXT NOT NULL DEFAULT 'related',
    weight      REAL NOT NULL DEFAULT 1.0,
    doc_id      TEXT,
    created_at  TEXT NOT NULL,
    UNIQUE(source_id, target_id, relation)
);

-- 问答记录（访问统计/反馈/缓存命中/延迟）
CREATE TABLE IF NOT EXISTS qa_records (
    id             TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    question       TEXT NOT NULL,
    question_type  TEXT NOT NULL DEFAULT '其他',
    answer         TEXT NOT NULL DEFAULT '',
    sources        TEXT,
    rating         INTEGER,
    marked_wrong   INTEGER NOT NULL DEFAULT 0,
    cache_hit      INTEGER NOT NULL DEFAULT 0,
    latency_ms     REAL NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL
);

-- 模拟面试会话
CREATE TABLE IF NOT EXISTS interview_sessions (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    job_type     TEXT NOT NULL,
    difficulty   TEXT NOT NULL,
    stress_mode  INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'ongoing',
    started_at   TEXT NOT NULL,
    ended_at     TEXT,
    export_path  TEXT
);

-- 模拟面试消息（题目/参考答案/用户回答/评分/反馈）
CREATE TABLE IF NOT EXISTS interview_messages (
    id               TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL REFERENCES interview_sessions(id) ON DELETE CASCADE,
    role             TEXT NOT NULL,
    question         TEXT NOT NULL DEFAULT '',
    reference_answer TEXT NOT NULL DEFAULT '',
    user_answer      TEXT NOT NULL DEFAULT '',
    score            REAL,
    feedback         TEXT,
    star_hint        INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL
);

-- 学习日志（每日练习时长/题数/错题数）
CREATE TABLE IF NOT EXISTS study_log (
    id               TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    log_date         TEXT NOT NULL,
    practice_seconds INTEGER NOT NULL DEFAULT 0,
    question_count   INTEGER NOT NULL DEFAULT 0,
    wrong_count      INTEGER NOT NULL DEFAULT 0,
    UNIQUE(user_id, log_date)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON document_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_docs_user ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_docs_folder ON documents(folder_id);
CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_qa_user ON qa_records(user_id);
CREATE INDEX IF NOT EXISTS idx_interview_user ON interview_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_interview_msgs ON interview_messages(session_id);
"""


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db_cursor(commit: bool = False) -> Iterator[sqlite3.Cursor]:
    conn = get_conn()
    try:
        cur = conn.cursor()
        yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with db_cursor(commit=True) as cur:
        cur.executescript(SCHEMA_SQL)
