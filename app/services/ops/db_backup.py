"""数据库备份的查看 / 手动触发 / 删除.

定时备份由宿主机 cron 跑 (scripts/companion-db-backup.sh), **刻意不搬进应用进程**:
备份最需要生效的时刻恰恰是应用出问题的时候, 挂在 APScheduler 上等于让它跟着一起
死. 这个模块只提供后台管理界面要的三个操作, 手动触发时复刻 cron 那套写入约定
(先落 .partial 再改名、dump 后校验), 两边产出的文件可以互相替换.

容器要能读写宿主机的备份目录, 靠 docker-compose 里的挂载; 手动触发要 pg_dump,
靠镜像里装的 postgresql-client. 缺任何一个, 这里的接口会明确报不可用而不是假装
成功 —— 备份功能"看起来能用其实没用"比直接报错危险得多.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from app.config import settings

logger = logging.getLogger(__name__)

BACKUP_DIR = Path(os.getenv("COMPANION_BACKUP_DIR", "/var/backups/companion"))

# 备份文件名的唯一合法形状. 删除接口拿它做白名单, 而不是过滤 "../" ——
# 黑名单式过滤总有漏网写法 (%2e%2e, 符号链接, 绝对路径), 白名单没有.
_NAME_RE = re.compile(r"^companion-\d{8}-\d{4}\.dump$")

_MIN_TABLES = 20  # 少于这个数说明 dump 不完整, 跟 cron 脚本保持一致
_TRIGGER_TIMEOUT_S = 300

_trigger_lock = asyncio.Lock()


@dataclass(frozen=True)
class BackupFile:
    name: str
    size_bytes: int
    created_at: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
        }


def _pg_dump_available() -> bool:
    return shutil.which("pg_dump") is not None


def _dir_state() -> str | None:
    """备份目录不可用的原因; None 表示可用.

    不能直接用 is_dir() —— 目录存在但没有权限时它**抛 PermissionError 而不是返回
    False**, 于是"报告可用性"的函数自己先崩了, 界面拿到 500 而不是原因说明.
    """
    try:
        if not BACKUP_DIR.is_dir():
            return f"备份目录 {BACKUP_DIR} 未挂载到容器"
    except OSError as exc:
        return f"备份目录 {BACKUP_DIR} 不可访问: {exc.strerror or exc}"
    if not os.access(BACKUP_DIR, os.R_OK | os.W_OK):
        return f"备份目录 {BACKUP_DIR} 无读写权限"
    return None


def availability() -> dict:
    """备份功能是否可用, 不可用时说清缺什么 —— 界面据此显示原因而不是空列表."""
    reasons = [r for r in (_dir_state(),) if r]
    if not _pg_dump_available():
        reasons.append("镜像内缺少 pg_dump (postgresql-client)")
    return {
        "available": not reasons,
        "directory": str(BACKUP_DIR),
        "reasons": reasons,
    }


def list_backups() -> list[BackupFile]:
    if _dir_state() is not None:
        return []
    out: list[BackupFile] = []
    try:
        candidates = list(BACKUP_DIR.glob("companion-*.dump"))
    except OSError:
        return []
    for path in candidates:
        if not _NAME_RE.match(path.name):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        out.append(BackupFile(
            name=path.name,
            size_bytes=stat.st_size,
            created_at=datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(timespec="seconds"),
        ))
    # 按真实修改时间排, 不按文件名。文件名里的时刻由生成方决定时区, 宿主机 cron
    # 用本地时间而容器默认 UTC —— 混在一起时名字的字典序不等于时间序, 刚生成的
    # 备份会排到下面去。mtime 没有这个歧义。
    return sorted(out, key=lambda b: b.created_at, reverse=True)


def _resolved_path(name: str) -> Path:
    """把文件名解析成目录内的真实路径, 越界就拒绝.

    两道锁: 名字必须完全匹配备份命名, 且解析后 (跟随符号链接) 仍要落在备份目录里.
    单靠名字校验挡不住目录里被放了指向别处的符号链接。
    """
    if not _NAME_RE.match(name or ""):
        raise ValueError("非法的备份文件名")
    target = (BACKUP_DIR / name).resolve()
    if target.parent != BACKUP_DIR.resolve():
        raise ValueError("备份文件不在备份目录内")
    return target


def delete_backup(name: str) -> None:
    target = _resolved_path(name)
    if not target.is_file():
        raise FileNotFoundError(name)
    remaining = [b for b in list_backups() if b.name != name]
    if not remaining:
        # 删到一个不剩需要显式确认 —— 误点一下就没有任何退路, 代价不对称.
        raise ValueError("这是最后一份备份, 拒绝删除")
    target.unlink()
    logger.warning(
        "db backup deleted: %s (剩余 %d 份)", name, len(remaining),
        extra={"event": "ops.backup_deleted", "backup_name": name},
    )


def _dsn_parts() -> dict:
    parsed = urlparse(settings.database_url)
    return {
        "host": parsed.hostname or "postgres",
        "port": str(parsed.port or 5432),
        "user": parsed.username or "companion",
        "password": parsed.password or "",
        "dbname": (parsed.path or "/companion").lstrip("/").split("?")[0],
    }


async def create_backup() -> BackupFile:
    """手动触发一次备份. 写入约定与 cron 脚本一致, 产物可互换."""
    state = availability()
    if not state["available"]:
        raise RuntimeError("；".join(state["reasons"]))
    if _trigger_lock.locked():
        raise RuntimeError("已有备份正在进行, 请稍候")

    async with _trigger_lock:
        parts = _dsn_parts()
        # 与宿主机 cron 脚本的 `date +%Y%m%d-%H%M` 对齐: 都用业务时区。容器默认
        # UTC, 直接 astimezone() 会产出跟 cron 差 8 小时的名字。
        stamp = datetime.now(ZoneInfo(settings.schedule_timezone)).strftime(
            "%Y%m%d-%H%M"
        )
        target = BACKUP_DIR / f"companion-{stamp}.dump"
        partial = target.with_suffix(".dump.partial")
        env = {**os.environ, "PGPASSWORD": parts["password"]}

        async def _run(*cmd: str) -> tuple[int, bytes]:
            proc = await asyncio.create_subprocess_exec(
                *cmd, env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await asyncio.wait_for(
                proc.communicate(), timeout=_TRIGGER_TIMEOUT_S
            )
            return proc.returncode or 0, out or b""

        code, out = await _run(
            "pg_dump", "-h", parts["host"], "-p", parts["port"],
            "-U", parts["user"], "-d", parts["dbname"], "-Fc",
            "-f", str(partial),
        )
        if code != 0:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"pg_dump 失败: {out.decode(errors='replace')[:200]}")

        # 校验后再改名 —— 没验证过能读的文件不该以正式备份的名字出现在列表里
        code, out = await _run("pg_restore", "--list", str(partial))
        tables = out.decode(errors="replace").count("TABLE DATA")
        if code != 0 or tables < _MIN_TABLES:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"备份校验失败: 归档内仅 {tables} 张表")

        partial.rename(target)
        stat = target.stat()
        logger.info(
            "db backup created manually: %s (%d bytes, %d tables)",
            target.name, stat.st_size, tables,
            extra={"event": "ops.backup_created", "backup_name": target.name},
        )
        return BackupFile(
            name=target.name,
            size_bytes=stat.st_size,
            created_at=datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(timespec="seconds"),
        )
