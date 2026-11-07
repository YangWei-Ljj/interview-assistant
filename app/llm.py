"""模型接入层：本地 Ollama（默认）/ DeepSeek（备选），嵌入与视觉 OCR 均走本地。"""
from __future__ import annotations

import base64
import functools
import json
import re

from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI

from . import config


@functools.lru_cache(maxsize=2)
def get_chat_model(temperature: float | None = None) -> ChatOpenAI:
    temp = config.LLM_TEMPERATURE if temperature is None else temperature
    if config.MODEL_PROVIDER == "ollama":
        return ChatOpenAI(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            api_key="ollama",
            temperature=temp,
            timeout=config.LLM_TIMEOUT_SECONDS,
        )
    if not config.DEEPSEEK_API_KEY:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，请在 .env 填写或把 MODEL_PROVIDER 改回 ollama")
    return ChatOpenAI(
        model=config.DEEPSEEK_MODEL,
        base_url=config.DEEPSEEK_BASE_URL,
        api_key=config.DEEPSEEK_API_KEY,
        temperature=temp,
        timeout=config.LLM_TIMEOUT_SECONDS,
    )


@functools.lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    """嵌入统一使用本地 Ollama：免费、无配额、隐私不出本机。"""
    return OllamaEmbeddings(
        model=config.OLLAMA_EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL.replace("/v1", ""),
    )


def chat_model_label() -> str:
    if config.MODEL_PROVIDER == "ollama":
        return f"Ollama / {config.OLLAMA_MODEL}"
    return f"DeepSeek / {config.DEEPSEEK_MODEL}"


def embed_model_label() -> str:
    return f"Ollama / {config.OLLAMA_EMBED_MODEL}"


# ---------------- 视觉 OCR（本地多模态，无需 tesseract）----------------

def vision_ocr(image_bytes: bytes, mime_type: str = "image/png") -> str:
    """调用本地多模态模型识别图片/截图中的文字，返回提取出的纯文本。"""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "请完整、准确地识别这张图片中的全部文字内容（含代码、表格、标题），"
                    "保持原有换行与层级，直接输出识别到的文字，不要添加解释或评价。"
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64}"},
            },
        ],
    }
    resp = get_chat_model().invoke([message])
    return (resp.content or "").strip()


# ---------------- 结构化输出辅助 ----------------

def parse_json_block(text: str):
    """从模型输出中容错解析 JSON：优先 ```json 代码块，否则截取首个 {...}。"""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fence.group(1).strip() if fence else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


def invoke_json(prompt: str, temperature: float = 0.0, retries: int = 1):
    """要求模型输出 JSON 并解析；失败按 retries 重试，仍失败返回 None。"""
    for _ in range(retries + 1):
        resp = get_chat_model(temperature).invoke(prompt)
        data = parse_json_block(resp.content or "")
        if data is not None:
            return data
    return None


def health_check() -> tuple[bool, str]:
    """检测对话模型与嵌入模型是否可用。"""
    try:
        get_chat_model().invoke("ping")
        chat_ok = True
    except Exception as exc:  # noqa: BLE001
        return False, f"对话模型不可用：{exc}"
    try:
        get_embeddings().embed_query("测试")
    except Exception as exc:  # noqa: BLE001
        if chat_ok:
            return True, f"对话模型已连接；嵌入模型未就绪（ollama pull {config.OLLAMA_EMBED_MODEL}）：{exc}"
        return False, f"嵌入模型不可用：{exc}"
    return True, "对话模型与嵌入模型均已连接"
