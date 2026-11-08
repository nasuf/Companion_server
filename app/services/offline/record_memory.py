"""随业务记录当前状态改写的记忆 (VOICE01.10 活动, B01.08 礼物) 的共用写入。

每条记录双方最多各一条记忆, 链接表记着是哪两条。状态变化时按记录的**当前**状态重写这一对,
不追加新条目: 撤销/取消后"打算去 X""礼物在路上"不会残留。重写只看表里的现状, 不看事件
顺序, 并发或乱序触发也写不乱; 内容没变就不写。

各业务只提供 Kind: 怎么读记录 (load) 和按状态写出双方文字 (render)。render 返回
{"user": str | None, "ai": str | None, "level", "importance"} 或 None (不该有记忆);
某一侧为 None 时, 那一侧已有的记忆归档。
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from app.config import settings
from app.db import db
from app.services.runtime.tasks import fire_background

logger = logging.getLogger(__name__)

_TZ = ZoneInfo(settings.schedule_timezone)
SIDES = ("user", "ai")
_ATTEMPTS = 3


@dataclass(frozen=True)
class Kind:
    name: str  # 日志、changelog 操作名和锁的前缀: activity_sync / gift_archived
    link_table: str
    key_column: str
    provenance: str
    main_category: str
    sub_category: str
    load: Callable[[Any, str, str], Awaitable[dict[str, Any] | None]]
    render: Callable[[dict[str, Any]], dict[str, Any] | None]


def day(value: Any) -> str:
    """本地日期 YYYY-MM-DD; 读不出来返回空串。"""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return ""
    if not isinstance(value, datetime):
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(_TZ).strftime("%Y-%m-%d")


def _text(texts: dict[str, Any] | None, side: str) -> str | None:
    return (texts or {}).get(side) or None


def _state_key(texts: dict[str, Any] | None) -> str:
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


async def sync(kind: Kind, record_id: str, user_id: str) -> str:
    """把这条记录的记忆对改成与记录当前状态一致。返回 unchanged/written/archived/missing/raced。"""
    from app.services.memory.storage.embedding import generate_embedding

    for _ in range(_ATTEMPTS):
        record = await kind.load(db, record_id, user_id)
        if record is None:
            return "missing"  # 记录已删除: 链接随之级联删除, 记忆由删除入口先归档
        texts = kind.render(record)
        # 向量在事务外算 (模型调用); 事务里重读状态, 变了就重来, 不会写进过时的文字
        vectors = {side: await generate_embedding(text) for side in SIDES if (text := _text(texts, side))}
        result = await _write(kind, record_id, user_id, texts, vectors)
        if result != "raced":
            return result
    logger.warning("[%s-memory] gave up after concurrent changes id=%s", kind.name, record_id[:8])
    return "raced"


async def _write(kind: Kind, record_id: str, user_id: str, texts: dict | None, vectors: dict) -> str:
    from app.services.memory.storage import repo as memory_repo
    from app.services.memory.storage.embedding import store_embedding
    from app.services.memory.storage.persistence import after_memory_changelog

    key = _state_key(texts)
    texts = texts or {}
    wanted = any(_text(texts, side) for side in SIDES)
    changed: list[tuple[str, str]] = []
    async with db.tx(timeout=timedelta(seconds=20)) as tx:
        await tx.execute_raw("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", f"{kind.name}-memory:{record_id}")
        current = await kind.load(tx, record_id, user_id)
        if current is None:
            return "missing"
        if _state_key(kind.render(current)) != key:
            return "raced"
        links = await tx.query_raw(
            f"SELECT user_memory_id, ai_memory_id, state_key FROM {kind.link_table} WHERE {kind.key_column}=$1",
            record_id,
        )
        if not links and not wanted:
            return "unchanged"  # 从没写过记忆, 现在也不该有
        link = dict(links[0]) if links else {"user_memory_id": None, "ai_memory_id": None, "state_key": ""}
        if link["state_key"] == key:
            return "unchanged"
        workspace_id = current.get("workspace_id")
        now = datetime.now(UTC)
        ids: dict[str, str | None] = {}
        for side in SIDES:
            table = "memories_ai" if side == "ai" else "memories_user"
            model = tx.aimemory if side == "ai" else tx.usermemory
            ident = link[f"{side}_memory_id"]
            row = (await tx.query_raw(f"SELECT id, content, is_archived FROM {table} WHERE id=$1 FOR UPDATE", ident)
                   if ident else [])
            row = row[0] if row else None
            if ident and row is None:
                # 用户自己删掉了这条记忆: 记住它被删过, 之后的状态变化也不再写回
                ids[side] = ident
                continue
            content = _text(texts, side)
            if content is None:
                if row and not row["is_archived"]:
                    await model.update(where={"id": ident}, data={"isArchived": True})
                    changed.append((ident, f"{kind.name}_archived"))
                    await _audit(tx, user_id, workspace_id, ident, f"{kind.name}_archived", row["content"], None)
                ids[side] = ident if row else None
                continue
            operation = f"{kind.name}_sync"
            if row:
                await model.update(where={"id": ident}, data={
                    "content": content, "level": texts["level"], "importance": texts["importance"],
                    "isArchived": False,
                    "statementTime": now, "valueUpdatedAt": now})
                await _audit(tx, user_id, workspace_id, ident, operation, row["content"], content)
            else:
                record = await memory_repo.create(
                    side, database=tx, userId=user_id, workspaceId=workspace_id, content=content,
                    level=texts["level"], importance=texts["importance"], type="life",
                    mainCategory=kind.main_category, subCategory=kind.sub_category, provenance=kind.provenance,
                    statementTime=now, valueUpdatedAt=now,
                )
                ident = record.id
                operation = "insert"
                await _audit(tx, user_id, workspace_id, ident, operation, None, content)
            await store_embedding(ident, vectors[side], database=tx)
            ids[side] = ident
            changed.append((ident, operation))
        await tx.execute_raw(
            f"""INSERT INTO {kind.link_table}({kind.key_column},user_id,workspace_id,user_memory_id,ai_memory_id,
                   state_key,updated_at)
               VALUES($1,$2,$3,$4,$5,$6,CURRENT_TIMESTAMP)
               ON CONFLICT ({kind.key_column}) DO UPDATE SET user_memory_id=EXCLUDED.user_memory_id,
                 ai_memory_id=EXCLUDED.ai_memory_id, state_key=EXCLUDED.state_key, updated_at=CURRENT_TIMESTAMP""",
            record_id, user_id, workspace_id, ids.get("user"), ids.get("ai"), key,
        )
    await memory_repo.invalidate_scope(user_id, workspace_id)
    for ident, operation in changed:
        await after_memory_changelog(user_id=user_id, memory_id=ident, operation=operation, workspace_id=workspace_id)
    if not wanted:
        return "archived" if changed else "unchanged"
    return "written"


async def _audit(tx, user_id, workspace_id, memory_id, operation, old, new) -> None:
    await tx.memorychangelog.create(data={
        "userId": user_id, "workspaceId": workspace_id, "memoryId": memory_id,
        "operation": operation, "oldValue": old, "newValue": new,
    })


async def _sync_logged(kind: Kind, record_id: str, user_id: str) -> None:
    try:
        result = await sync(kind, record_id, user_id)
    except Exception as exc:  # 后台写: 失败只记日志, 下次状态变化或修复脚本会补上
        logger.warning("[%s-memory] sync failed id=%s err=%s", kind.name, record_id[:8], type(exc).__name__)
        return
    logger.info("[%s-memory] id=%s result=%s", kind.name, record_id[:8], result)


def schedule_sync(kind: Kind, record_id: str | None, user_id: str | None) -> None:
    """状态变化后调用 (不阻塞接口); 同一记录多次触发只会收敛到最新状态。"""
    if record_id and user_id:
        fire_background(_sync_logged(kind, str(record_id), str(user_id)))


async def archive_for_user(kind: Kind, user_id: str) -> int:
    """清空某类记录前调用: 对应的状态记忆一起归档 (链接随记录级联删除)。"""
    from app.services.memory.storage import repo as memory_repo

    archived = 0
    scopes: set[str | None] = set()
    async with db.tx(timeout=timedelta(seconds=20)) as tx:
        links = await tx.query_raw(
            f"SELECT workspace_id, user_memory_id, ai_memory_id FROM {kind.link_table} WHERE user_id=$1 FOR UPDATE",
            user_id,
        )
        for link in links:
            for side, table in (("user", "memories_user"), ("ai", "memories_ai")):
                ident = link[f"{side}_memory_id"]
                if not ident:
                    continue
                rows = await tx.query_raw(
                    f"UPDATE {table} SET is_archived=TRUE, updated_at=CURRENT_TIMESTAMP "
                    "WHERE id=$1 AND user_id=$2 AND NOT is_archived RETURNING content", ident, user_id)
                if rows:
                    archived += 1
                    scopes.add(link["workspace_id"])
                    await _audit(tx, user_id, link["workspace_id"], ident, f"{kind.name}_archived",
                                 rows[0]["content"], None)
    for workspace_id in scopes:
        await memory_repo.invalidate_scope(user_id, workspace_id)
    return archived
