"""多格式文档解析：txt/md/pdf/docx/图片(OCR)/网页链接，统一返回纯文本。"""
from __future__ import annotations

import io
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from . import config
from .llm import vision_ocr


class DocumentError(Exception):
    pass


_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def _read_text(raw: bytes) -> str:
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise DocumentError("文本编码无法识别，请另存为 UTF-8 后再上传")


def _read_pdf(raw: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    parts = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(parts).strip()


def _read_docx(raw: bytes) -> str:
    """提取段落与表格；表格用 | 分隔成行，便于后续结构化识别。"""
    from docx import Document

    document = Document(io.BytesIO(raw))
    lines: list[str] = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines).strip()


def _read_image(raw: bytes, ext: str) -> str:
    return vision_ocr(raw, _MIME.get(ext, "image/png"))


def parse_upload(filename: str, raw: bytes) -> dict:
    """解析上传文件，返回 {text, doc_type}。"""
    ext = Path(filename).suffix.lower()
    if ext not in config.ALLOWED_UPLOAD_EXT:
        raise DocumentError(f"仅支持 {', '.join(sorted(config.ALLOWED_UPLOAD_EXT))} 格式")
    if len(raw) == 0:
        raise DocumentError("文件为空")
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise DocumentError(f"文件超过 {config.MAX_UPLOAD_BYTES // 1024 // 1024}MB 限制")

    if ext in (".txt", ".md"):
        return {"text": _read_text(raw), "doc_type": "text"}
    if ext == ".pdf":
        return {"text": _read_pdf(raw), "doc_type": "pdf"}
    if ext == ".docx":
        return {"text": _read_docx(raw), "doc_type": "docx"}
    if ext in config.IMAGE_EXT:
        return {"text": _read_image(raw, ext), "doc_type": "image_ocr"}
    raise DocumentError("不支持的格式")


def fetch_url(url: str, timeout: int = 20) -> dict:
    """抓取网页正文，返回 {text, title, doc_type}。"""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        raise DocumentError("链接需以 http:// 或 https:// 开头")
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; InterviewKB/1.0)"},
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise DocumentError(f"网页抓取失败：{exc}") from exc

    soup = BeautifulSoup(resp.text, "html.parser")
    title = (soup.title.string.strip() if soup.title and soup.title.string else url)
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()
    # 优先取文章主体，退化到整页文本
    main = soup.find(["article", "main"]) or soup
    text = main.get_text("\n", strip=True)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned = "\n".join(lines)
    if len(cleaned) < 20:
        raise DocumentError("网页正文内容过少，无法导入")
    return {"text": f"# {title}\n来源：{url}\n\n{cleaned}", "title": title, "doc_type": "webpage"}
