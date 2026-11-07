"""线下活动的状态记忆 (VOICE01.10): 用户回应过的活动, 双方各一条记忆, 始终与活动当前状态一致。

用户接受、忽略、到达、完成、取消, 或活动过期时, 按活动记录的**当前**状态重写同一对记忆
(offline_activity_memories 记着是哪两条), 不追加新条目: 取消后"用户打算去 X"不会残留
(40cc0d69 删掉接受记忆就是因为这个)。重写只看表里的现状, 不看事件顺序, 并发或乱序触发也
写不乱; 内容没变就不写。

用户从没回应过的推荐不写 (生产上大多数推荐属于这类, 写进去只会冲淡记忆), 用户问起"推荐过
什么"由 recommendation_history 直接读表。行的 provenance 是 activity_record: 写入期合并、
去重整理和 L3 压缩都不碰它们, 否则下一次状态变化就找不到、改不到了。
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from app.config import settings
from app.db import db
from app.services.memory.provenance import ACTIVITY_RECORD
from app.services.runtime.tasks import fire_background

logger = logging.getLogger(__name__)

_TZ = ZoneInfo(settings.schedule_timezone)
MAIN, SUB = "生活", "其他特殊事件"
EXTRACTOR = "activity-memory-v1"
# 去过 / 正在去的是关系里的事, 进 L2; 没去成的落 L3, 像真人一样慢慢淡出
_KEPT = (2, 0.55)
_FADED = (3, 0.4)
_ATTEMPTS = 3


def _day(value: Any) -> str:
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


def responded(activity: dict[str, Any]) -> bool:
    """用户对这条推荐做过选择 (接受/不考虑/到过/完成); 只是收到推荐的不算。"""
    return bool(activity.get("accepted_at") or activity.get("ignored_at") or activity.get("reached")
                or activity.get("status") == "completed")


def _outcome(a: dict[str, Any]) -> tuple[str, tuple[int, float]] | None:
    """(用户这边做了什么, (层级, 重要度)); None = 不该有记忆。"""
    status, reached = a.get("status"), bool(a.get("reached"))
    arrived = _day(a.get("arrival_confirmed_at"))
    if status == "completed":
        return (f"{arrived}到过现场，这次活动已经完成" if reached else "这次活动已经完成"), _KEPT
    if status == "accepted":
        if reached:
            return f"{arrived}到了现场", _KEPT
        return f"{_day(a.get('accepted_at'))}接受了，放进了待出行，还没去", _KEPT
    if status == "ignored":
        return f"{_day(a.get('ignored_at'))}说暂时不考虑", _FADED
    if status == "cancelled" and a.get("accepted_at"):
        return f"接受后又取消了（{_day(a.get('updated_at'))}），没去", _FADED
    if status == "expired" and (a.get("accepted_at") or a.get("ignored_at")):
        return f"活动过期了（{_day(a.get('updated_at'))}），没去", _FADED
    return None


def render(activity: dict[str, Any], agent_name: str | None) -> dict[str, Any] | None:
    """按活动当前状态写出双方的记忆; 不该有记忆时返回 None。"""
    if not responded(activity):
        return None
    outcome = _outcome(activity)
    if outcome is None:
        return None
    what, (level, importance) = outcome
    title = str(activity.get("title") or "").strip()
    if not title:
        return None
    place = str(activity.get("location_name") or "").strip()
    if place and place in title:
        place = ""  # 标题已经是这个地方, 不再重复一遍
    city = str(activity.get("city") or "").strip()
    if city and city not in place:
        place = f"{city}·{place}" if place else city
    where = f"（{place}）" if place else ""
    day = _day(activity.get("created_at"))
    agent = (agent_name or "").strip() or "对方"
    return {
        # 用户侧写用户的经历; AI 侧写清"是我推荐给用户的、在用户那边", 不会被当成自己去过
        "user": f"{day} {agent}给用户推荐了线下活动「{title}」{where}，用户{what}",
        "ai": f"{day} 我给用户推荐了用户那边的线下活动「{title}」{where}，用户{what}",
        "level": level,
        "importance": importance,
    }


def _state_key(texts: dict[str, Any] | None) -> str:
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


async def _load(database, activity_id: str, user_id: str) -> dict[str, Any] | None:
    rows = await database.query_raw(
        """SELECT a.id, a.user_id, a.workspace_id, a.title, a.location_name, a.city, a.status, a.reached,
                  a.accepted_at, a.ignored_at, a.arrival_confirmed_at, a.completed_at, a.created_at,
                  a.updated_at, g.name AS agent_name
           FROM offline_activity_recommendations a LEFT JOIN ai_agents g ON g.id = a.agent_id
           WHERE a.id = $1 AND a.user_id = $2""",
        activity_id, user_id,
    )
    return dict(rows[0]) if rows else None


async def sync(activity_id: str, user_id: str) -> str:
    """把这个活动的记忆对改成与活动当前状态一致。返回 unchanged/written/archived/missing。"""
    from app.services.memory.storage.embedding import generate_embedding

    for _ in range(_ATTEMPTS):
        activity = await _load(db, activity_id, user_id)
        if activity is None:
            return "missing"  # 活动行已删除: 链接随之级联删除, 记忆由删除入口先归档
        texts = render(activity, activity.get("agent_name"))
        # 向量在事务外算 (模型调用); 事务里重读状态, 变了就重来, 不会写进过时的文字
        vectors = {side: await generate_embedding(texts[side]) for side in ("user", "ai")} if texts else {}
        result = await _write(activity_id, user_id, texts, vectors)
        if result != "raced":
            return result
    logger.warning("[activity-memory] gave up after concurrent changes activity=%s", activity_id[:8])
    return "raced"


async def _write(activity_id: str, user_id: str, texts: dict | None, vectors: dict) -> str:
    from app.services.memory.storage import repo as memory_repo
    from app.services.memory.storage.embedding import store_embedding
    from app.services.memory.storage.persistence import after_memory_changelog

    key = _state_key(texts)
    changed: list[tuple[str, str]] = []
    async with db.tx(timeout=timedelta(seconds=20)) as tx:
        await tx.execute_raw("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", "activity-memory:" + activity_id)
        current = await _load(tx, activity_id, user_id)
        if current is None:
            return "missing"
        if _state_key(render(current, current.get("agent_name"))) != key:
            return "raced"
        links = await tx.query_raw(
            "SELECT user_memory_id, ai_memory_id, state_key FROM offline_activity_memories WHERE activity_id=$1",
            activity_id,
        )
        if not links and texts is None:
            return "unchanged"  # 从没写过记忆, 现在也不该有
        link = dict(links[0]) if links else {"user_memory_id": None, "ai_memory_id": None, "state_key": ""}
        if link["state_key"] == key:
            return "unchanged"
        workspace_id = current.get("workspace_id")
        now = datetime.now(UTC)
        ids: dict[str, str | None] = {}
        for side in ("user", "ai"):
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
            if texts is None:
                if row and not row["is_archived"]:
                    await model.update(where={"id": ident}, data={"isArchived": True})
                    changed.append((ident, "activity_archived"))
                    await _audit(tx, user_id, workspace_id, ident, "activity_archived", row["content"], None)
                ids[side] = ident if row else None
                continue
            content = texts[side]
            operation = "activity_sync"
            if row:
                await model.update(where={"id": ident}, data={
                    "content": content, "level": texts["level"], "importance": texts["importance"],
                    "isArchived": False,
                    "statementTime": now, "valueUpdatedAt": now})
                await _audit(tx, user_id, workspace_id, ident, "activity_sync", row["content"], content)
            else:
                record = await memory_repo.create(
                    side, database=tx, userId=user_id, workspaceId=workspace_id, content=content,
                    level=texts["level"], importance=texts["importance"], type="life",
                    mainCategory=MAIN, subCategory=SUB, provenance=ACTIVITY_RECORD,
                    statementTime=now, valueUpdatedAt=now,
                )
                ident = record.id
                operation = "insert"
                await _audit(tx, user_id, workspace_id, ident, operation, None, content)
            await store_embedding(ident, vectors[side], database=tx)
            ids[side] = ident
            changed.append((ident, operation))
        await tx.execute_raw(
            """INSERT INTO offline_activity_memories(activity_id,user_id,workspace_id,user_memory_id,ai_memory_id,state_key,updated_at)
               VALUES($1,$2,$3,$4,$5,$6,CURRENT_TIMESTAMP)
               ON CONFLICT (activity_id) DO UPDATE SET user_memory_id=EXCLUDED.user_memory_id,
                 ai_memory_id=EXCLUDED.ai_memory_id, state_key=EXCLUDED.state_key, updated_at=CURRENT_TIMESTAMP""",
            activity_id, user_id, workspace_id, ids.get("user"), ids.get("ai"), key,
        )
    await memory_repo.invalidate_scope(user_id, workspace_id)
    for ident, operation in changed:
        await after_memory_changelog(user_id=user_id, memory_id=ident, operation=operation, workspace_id=workspace_id)
    if texts is None:
        return "archived" if changed else "unchanged"
    return "written"


async def _audit(tx, user_id, workspace_id, memory_id, operation, old, new) -> None:
    await tx.memorychangelog.create(data={
        "userId": user_id, "workspaceId": workspace_id, "memoryId": memory_id,
        "operation": operation, "oldValue": old, "newValue": new,
    })


async def _sync_logged(activity_id: str, user_id: str) -> None:
    try:
        result = await sync(activity_id, user_id)
    except Exception as exc:  # 后台写: 失败只记日志, 下次状态变化或修复脚本会补上
        logger.warning("[activity-memory] sync failed activity=%s err=%s", activity_id[:8], type(exc).__name__)
        return
    logger.info("[activity-memory] activity=%s result=%s", activity_id[:8], result)


def schedule_sync(activity_id: str | None, user_id: str | None) -> None:
    """状态变化后调用 (不阻塞接口); 同一活动多次触发只会收敛到最新状态。"""
    if activity_id and user_id:
        fire_background(_sync_logged(str(activity_id), str(user_id)))


async def archive_for_user(user_id: str) -> int:
    """用户清空活动列表前调用: 对应的活动记忆一起归档 (链接随活动级联删除)。"""
    from app.services.memory.storage import repo as memory_repo

    archived = 0
    scopes: set[str | None] = set()
    async with db.tx(timeout=timedelta(seconds=20)) as tx:
        links = await tx.query_raw(
            "SELECT workspace_id, user_memory_id, ai_memory_id FROM offline_activity_memories WHERE user_id=$1 FOR UPDATE",
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
                    await _audit(tx, user_id, link["workspace_id"], ident, "activity_archived", rows[0]["content"], None)
    for workspace_id in scopes:
        await memory_repo.invalidate_scope(user_id, workspace_id)
    return archived
