"""Admin adapter for the host-independent, shared local backup engine."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from scripts import postgres_backup as engine

logger = logging.getLogger(__name__)
BACKUP_DIR = Path(os.getenv("COMPANION_BACKUP_DIR", "/var/backups/companion"))


def availability() -> dict:
    reasons = []
    try:
        if not BACKUP_DIR.is_dir():
            reasons.append("备份目录未挂载到容器")
        elif not os.access(BACKUP_DIR, os.R_OK | os.W_OK):
            reasons.append("备份目录无读写权限")
    except OSError:
        reasons.append("备份目录不可访问")
    try:
        if engine.read_json(BACKUP_DIR / ".scheduler.json").get("schema_version") != 1:
            reasons.append("宿主机备份执行器未配置")
    except (OSError, engine.BackupError):
        reasons.append("宿主机备份执行器未配置或配置不可读")
    return {"available": not reasons, "directory": str(BACKUP_DIR), "reasons": reasons}


def list_report(offset: int = 0, limit: int = 20) -> dict:
    state = availability()
    try:
        report = engine.snapshot(BACKUP_DIR, offset, limit)
    except (OSError, engine.BackupError, ValueError, TypeError):
        return {**state, "available": False, "reasons": state["reasons"] + ["备份记录未采集或读取失败，需运维检查"],
                "backups": [], "summary": None, "total": None, "as_of": engine.stamp(),
                "offset": offset, "limit": limit, "has_more": False}
    return {**state, **report}


def delete_backup(name: str) -> None:
    try:
        engine.remove(BACKUP_DIR, name)
    except engine.BackupError as exc:
        raise ValueError(str(exc)) from exc
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValueError("备份目录不可访问，未执行删除") from exc
    logger.warning("database backup deleted: %s", name)


async def create_backup() -> dict:
    state = availability()
    if not state["available"]:
        raise RuntimeError("；".join(state["reasons"]))
    try:
        return await asyncio.to_thread(engine.request_backup, BACKUP_DIR)
    except OSError as exc:
        raise RuntimeError("备份请求写入失败，需运维检查") from exc
