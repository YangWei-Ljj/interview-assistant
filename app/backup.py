"""本地备份与恢复：把 data 目录打包为 zip；恢复前自动备份并做路径安全校验。"""
from __future__ import annotations

import shutil
import sqlite3
import zipfile
from datetime import datetime
from pathlib import Path

from . import config

_SKIP_SUFFIX = ("-wal", "-shm", ".log")
_SKIP_DIRS = {"backups", "__pycache__"}


def _checkpoint_sqlite() -> None:
    """备份前把 WAL 日志合并进主库，降低备份遗漏最近写入的风险。"""
    try:
        conn = sqlite3.connect(config.DB_PATH)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
    except sqlite3.Error:
        pass


def create_backup(custom_name: str | None = None) -> Path:
    _checkpoint_sqlite()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = custom_name or f"backup_{stamp}"
    if not name.endswith(".zip"):
        name += ".zip"
    out_path = config.BACKUP_DIR / name

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in config.DATA_DIR.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(config.DATA_DIR)
            if any(part in _SKIP_DIRS for part in rel.parts):
                continue
            if path.name.endswith(_SKIP_SUFFIX):
                continue
            zf.write(path, rel.as_posix())
    return out_path


def list_backups() -> list[dict]:
    files = sorted(config.BACKUP_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"name": p.name, "size_kb": round(p.stat().st_size / 1024, 1),
             "modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")}
            for p in files]


def _safe_members(zf: zipfile.ZipFile, target: Path) -> list[zipfile.ZipInfo]:
    """防 zip slip：确保每个成员解压后仍位于目标目录内，且不触碰 backups。"""
    members = []
    target = target.resolve()
    for info in zf.infolist():
        if info.is_dir():
            continue
        dest = (target / info.filename).resolve()
        if not str(dest).startswith(str(target)):
            raise ValueError(f"备份包含非法路径：{info.filename}")
        if info.filename.split("/")[0] in _SKIP_DIRS:
            continue
        members.append(info)
    return members


def restore_backup(zip_name: str) -> Path:
    """恢复备份：先自动备份当前数据，再清空 data（保留 backups）后解压。"""
    zip_path = config.BACKUP_DIR / zip_name
    if not zip_path.exists():
        raise FileNotFoundError(f"备份文件不存在：{zip_name}")

    safety = create_backup("pre_restore_safety")  # 恢复前兜底，可随时回退

    with zipfile.ZipFile(zip_path) as zf:
        members = _safe_members(zf, config.DATA_DIR)
        # 删除 data 下除 backups 外的现有内容
        for child in config.DATA_DIR.iterdir():
            if child.name in _SKIP_DIRS:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        for info in members:
            zf.extract(info, config.DATA_DIR)
    return safety


def delete_backup(zip_name: str) -> None:
    p = config.BACKUP_DIR / zip_name
    if p.exists() and p.parent.resolve() == config.BACKUP_DIR.resolve():
        p.unlink()
