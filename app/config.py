"""全局配置：路径、模型、分块、混合检索权重、缓存、上传与面试参数。

设计说明：
- 环境变量（.env）控制模型提供方等部署相关配置；
- 用户在「系统设置」页调整的运行参数写入 data/settings.json，启动时覆盖默认值，
  新手无需改代码即可调分块长度、检索条数、融合权重等。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ---------------- 路径 ----------------
DATA_DIR = Path(os.getenv("INTERVIEW_DATA_DIR", str(BASE_DIR / "data")))
DB_PATH = DATA_DIR / "interview.db"
CHROMA_DIR = DATA_DIR / "chroma_db"
UPLOAD_DIR = DATA_DIR / "uploads"
SEED_DOC_DIR = DATA_DIR / "seed_docs"
CACHE_DIR = DATA_DIR / "cache"
BACKUP_DIR = DATA_DIR / "backups"
L2_CACHE_PATH = CACHE_DIR / "l2_cache.json"
SETTINGS_PATH = DATA_DIR / "settings.json"
LOG_DIR = BASE_DIR / "logs"
for _d in (DATA_DIR, CHROMA_DIR, UPLOAD_DIR, SEED_DOC_DIR, CACHE_DIR, BACKUP_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------- 模型（部署相关，来自 .env）----------------
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama").strip().lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b").strip()
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b").strip()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "90"))
LLM_MAX_HISTORY_MESSAGES = int(os.getenv("LLM_MAX_HISTORY_MESSAGES", "20"))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8516"))

# ---------------- 运行参数默认值（可被 settings.json 覆盖）----------------
_DEFAULTS: dict = {
    # 分块：需求允许 300-800 字符可调
    "chunk_size": 500,
    "chunk_overlap": 100,
    # 检索
    "top_k": 6,
    "hybrid_search_enabled": True,
    "graph_search_enabled": True,
    # 融合权重（落地需求“6:4”口径：语义向量 0.6 + 关键词 BM25 0.4；图谱做关联扩展加分）
    "vector_weight": 0.6,
    "bm25_weight": 0.4,
    "graph_boost": 0.15,
    # 向量双阈值（Chroma L2 距离，越小越相关）
    # 实测：相关问题首条距离 0.3-0.8，离题首条 1.295，取 1.2 干净分隔
    "distance_threshold": 1.2,
    "relative_spread": 0.6,
    # BM25 地板
    "bm25_min_score": 1.5,
    "bm25_min_matched_tokens": 2,
    # 缓存
    "cache_enabled": True,
    "l1_max_entries": 200,
    # 校验
    "triple_check_enabled": True,
    # 语音
    "asr_model": "base",
    # 面试默认
    "interview_question_count": 5,
}

# 分块允许范围（界面校验用）
CHUNK_SIZE_MIN, CHUNK_SIZE_MAX = 300, 800


def _load_user_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


_SETTINGS = {**_DEFAULTS, **_load_user_settings()}


def get_setting(key: str):
    return _SETTINGS.get(key, _DEFAULTS.get(key))


def save_user_settings(new_values: dict) -> dict:
    """校验并保存运行参数，更新内存并写 settings.json；返回清洗后的设置。"""
    cleaned = dict(_SETTINGS)
    for key, value in new_values.items():
        if key not in _DEFAULTS:
            continue
        if key == "chunk_size":
            value = int(value)
            if not (CHUNK_SIZE_MIN <= value <= CHUNK_SIZE_MAX):
                raise ValueError(f"分块长度需在 {CHUNK_SIZE_MIN}-{CHUNK_SIZE_MAX} 字符之间")
        elif key in ("chunk_overlap", "top_k", "l1_max_entries", "interview_question_count",
                      "bm25_min_matched_tokens"):
            value = max(0, int(value))
        elif key in ("vector_weight", "bm25_weight", "graph_boost", "distance_threshold",
                      "relative_spread", "bm25_min_score"):
            value = max(0.0, float(value))
        elif key in ("hybrid_search_enabled", "graph_search_enabled", "cache_enabled",
                      "triple_check_enabled"):
            value = bool(value)
        else:
            value = str(value)
        cleaned[key] = value
    # 归一化融合权重，使向量+BM25 合计为 1
    total_w = cleaned["vector_weight"] + cleaned["bm25_weight"]
    if total_w > 0:
        cleaned["vector_weight"] = round(cleaned["vector_weight"] / total_w, 4)
        cleaned["bm25_weight"] = round(cleaned["bm25_weight"] / total_w, 4)
    SETTINGS_PATH.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    _SETTINGS.clear()
    _SETTINGS.update(cleaned)
    return cleaned


def reset_user_settings() -> None:
    if SETTINGS_PATH.exists():
        SETTINGS_PATH.unlink()
    _SETTINGS.clear()
    _SETTINGS.update(_DEFAULTS)


# 便捷常量（导入时固定；设置页保存后需重启进程生效，界面会提示）
CHUNK_SIZE = int(get_setting("chunk_size"))
CHUNK_OVERLAP = int(get_setting("chunk_overlap"))
RETRIEVER_TOP_K = int(get_setting("top_k"))
VECTOR_WEIGHT = float(get_setting("vector_weight"))
BM25_WEIGHT = float(get_setting("bm25_weight"))
GRAPH_BOOST = float(get_setting("graph_boost"))
SCORE_DISTANCE_THRESHOLD = float(get_setting("distance_threshold"))
SCORE_RELATIVE_SPREAD = float(get_setting("relative_spread"))
BM25_MIN_SCORE = float(get_setting("bm25_min_score"))
BM25_MIN_MATCHED_TOKENS = int(get_setting("bm25_min_matched_tokens"))
ENABLE_CACHE = bool(get_setting("cache_enabled"))
L1_MAX_ENTRIES = int(get_setting("l1_max_entries"))
ENABLE_TRIPLE_CHECK = bool(get_setting("triple_check_enabled"))
ASR_MODEL = str(get_setting("asr_model"))
INTERVIEW_QUESTION_COUNT = int(get_setting("interview_question_count"))

COLLECTION_NAME = "interview_kb"

# ---------------- 上传 ----------------
ALLOWED_UPLOAD_EXT = {
    ".txt", ".md", ".pdf", ".docx",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp",
}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100MB

# ---------------- 面试选项 ----------------
JOB_TYPES = ["AI应用/全栈工程师", "后端开发工程师", "前端开发工程师", "数据分析师", "通用技术岗"]
DIFFICULTY_LEVELS = ["初级(校招)", "中级", "高级/压力面"]
QUESTION_TYPES = ["技术知识", "项目经历", "行为面试", "算法编码", "其他"]

DEFAULT_USER_ID = "local_owner"
