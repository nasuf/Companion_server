"""运行时状态清理服务。

用于工作区归档时清理旧 agent 的缓存、图谱状态和触发器，保留 PostgreSQL 历史数据。
"""

import logging

from app.db import db
from app.neo4j_client import run_write
from app.redis_client import get_redis

logger = logging.getLogger(__name__)


async def clear_agent_runtime_state(
    workspace_id: str,
    agent_id: str,
    user_id: str,
    conversation_ids: list[str] | None = None,
) -> dict[str, int]:
    """清理指定 agent + user 的运行时状态。"""
    stats: dict[str, int] = {}

    conv_ids = list(conversation_ids or [])
    if not conv_ids:
        convs = await db.query_raw(
            "SELECT id FROM conversations WHERE agent_id = $1",
            agent_id,
        )
        conv_ids = [c["id"] for c in (convs or [])]

    stats["postgres"] = await _clear_runtime_postgres(agent_id)
    stats["neo4j"] = await _clear_neo4j(workspace_id, agent_id, user_id)
    stats["redis"] = await _clear_redis(agent_id, user_id, conv_ids)

    logger.info(f"Runtime state cleared for agent={agent_id} user={user_id}: {stats}")
    return stats


async def reset_agent_data(agent_id: str, user_id: str) -> dict[str, int]:
    """兼容旧调用方名称。"""
    return await clear_agent_runtime_state("legacy", agent_id, user_id)


async def _clear_runtime_postgres(agent_id: str) -> int:
    """清理影响线上行为但不需要长期保留的 PostgreSQL 运行时状态。"""
    total = 0
    queries = [
        ("UPDATE time_triggers SET is_active = false WHERE ai_agent_id = $1", agent_id),
        (
            """
            UPDATE proactive_states
            SET status = 'stopped', stop_reason = 'workspace_archived', updated_at = CURRENT_TIMESTAMP
            WHERE agent_id = $1
            """,
            agent_id,
        ),
    ]

    for sql, param in queries:
        try:
            cnt = await db.execute_raw(sql, param)
            total += cnt or 0
        except Exception as e:
            logger.warning(f"PG runtime cleanup failed: {sql[:60]}... — {e}")

    return total


async def _clear_neo4j(workspace_id: str, agent_id: str, user_id: str) -> int:
    """归档 Neo4j 当前 workspace 节点，不硬删历史。"""
    updated = 0
    queries = [
        (
            """
            MATCH (a:AI {id: $agent_id, workspace_id: $workspace_id})
            SET a.status = 'archived', a.archived_at = datetime()
            """,
            {"agent_id": agent_id, "workspace_id": workspace_id},
        ),
        (
            """
            MATCH (u:User {id: $user_id, workspace_id: $workspace_id})
            SET u.status = 'archived', u.archived_at = datetime()
            """,
            {"user_id": user_id, "workspace_id": workspace_id},
        ),
        (
            """
            MATCH (m:Memory {workspace_id: $workspace_id})
            SET m.status = 'archived', m.archived_at = datetime()
            """,
            {"workspace_id": workspace_id},
        ),
    ]
    for query, params in queries:
        try:
            await run_write(query, params)
            updated += 1
        except Exception as e:
            logger.warning(f"Neo4j archive failed: {e}")
    return updated


async def _clear_redis(agent_id: str, user_id: str, conv_ids: list[str]) -> int:
    """删除所有相关 Redis 键。"""
    redis = await get_redis()

    # 精确 key 列表
    exact_keys = [
        f"emotion:{agent_id}",
        f"life_overview:{agent_id}",
        f"patience:{agent_id}:{user_id}",
        f"blacklist_timer:{agent_id}:{user_id}",
        f"attack_history:{agent_id}:{user_id}",
        f"trigger_last:{agent_id}:{user_id}",
        f"intimacy:{agent_id}:{user_id}",
        f"topic_intimacy:{agent_id}:{user_id}",
        f"pending:msgs:{user_id}",
        f"pending:ctx:{user_id}",
        f"last_reply:{agent_id}:{user_id}",
    ]

    # conversation 相关的精确 key
    for cid in conv_ids:
        exact_keys.append(f"topics:{cid}")
        exact_keys.append(f"context_window:{cid}")
        exact_keys.append(f"working_facts:{cid}")
        exact_keys.append(f"delayed:msgs:{cid}")

    # 通配符 patterns（需要 SCAN）
    scan_patterns = [
        f"schedule:{agent_id}:*",
        f"schedule_adj:{agent_id}:*",
        f"trait_adj:{agent_id}:*",
        f"trait_adj_week:{agent_id}:*",
        f"trigger_count:{agent_id}:{user_id}:*",
        f"proactive_count:{agent_id}:{user_id}:*",
        f"cache:sum:*",   # summarizer cache（MD5 key，无法按 conv 过滤，清全部）
        f"cache:ret:*",   # retrieval cache
        f"cache:graph:*", # graph cache
        f"cache:emb:*",   # embedding cache
    ]

    deleted = 0

    # 批量删除精确 key
    if exact_keys:
        try:
            deleted += await redis.delete(*exact_keys)
        except Exception:
            pass

    if conv_ids:
        try:
            deleted += await redis.zrem("delayed:due", *conv_ids)
        except Exception:
            pass

    # SCAN 删除通配符 key
    for pattern in scan_patterns:
        try:
            cursor = 0
            while True:
                cursor, keys = await redis.scan(cursor, match=pattern, count=100)
                if keys:
                    deleted += await redis.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.warning(f"Redis scan delete failed for {pattern}: {e}")

    return deleted


async def hard_delete_agent_data(agent_id: str, user_id: str) -> dict:
    """彻底物理删除用户与某个 Agent 的全部数据，不影响其他 Agent。"""
    stats: dict[str, int] = {}

    # 1. 找到该 agent 的所有 workspace
    workspaces = await db.chatworkspace.find_many(
        where={"agentId": agent_id, "userId": user_id},
    )
    workspace_ids = [w.id for w in workspaces]

    # 2. 找到该 agent 所有 conversation
    conversations = await db.conversation.find_many(
        where={"agentId": agent_id, "userId": user_id},
    )
    conv_ids = [c.id for c in conversations]

    # 3. 删除 messages
    if conv_ids:
        cnt = await db.message.delete_many(
            where={"conversationId": {"in": conv_ids}},
        )
        stats["messages"] = cnt

    # 4. 删除 conversations
    if conv_ids:
        cnt = await db.conversation.delete_many(
            where={"id": {"in": conv_ids}},
        )
        stats["conversations"] = cnt

    # 5. 删除 workspace 下的 memories + embeddings
    if workspace_ids:
        # 收集 memory ids 用于删除 embeddings
        user_mems = await db.usermemory.find_many(
            where={"workspaceId": {"in": workspace_ids}},
        )
        ai_mems = await db.aimemory.find_many(
            where={"workspaceId": {"in": workspace_ids}},
        )
        mem_ids = [m.id for m in user_mems] + [m.id for m in ai_mems]

        if mem_ids:
            try:
                cnt = await db.execute_raw(
                    "DELETE FROM memory_embeddings WHERE memory_id = ANY($1::text[])",
                    mem_ids,
                )
                stats["embeddings"] = cnt or 0
            except Exception as e:
                logger.warning(f"Embedding delete failed: {e}")

        cnt = await db.usermemory.delete_many(
            where={"workspaceId": {"in": workspace_ids}},
        )
        stats["user_memories"] = cnt

        cnt = await db.aimemory.delete_many(
            where={"workspaceId": {"in": workspace_ids}},
        )
        stats["ai_memories"] = cnt

        # UserProfile, MemoryChangelog
        cnt = await db.userprofile.delete_many(
            where={"workspaceId": {"in": workspace_ids}},
        )
        stats["profiles"] = cnt

        cnt = await db.memorychangelog.delete_many(
            where={"workspaceId": {"in": workspace_ids}},
        )
        stats["changelogs"] = cnt

    # 删除 workspaces (按 agentId 确保全部清除)
    cnt = await db.chatworkspace.delete_many(
        where={"agentId": agent_id, "userId": user_id},
    )
    stats["workspaces"] = cnt

    # 6. 删除 agent 级别数据
    try:
        cnt = await db.intimacy.delete_many(
            where={"agentId": agent_id, "userId": user_id},
        )
        stats["intimacy"] = cnt
    except Exception:
        pass

    for model_name, model in [
        ("emotion_states", db.aiemotionstate),
        ("schedules", db.aidailyschedule),
        ("trait_logs", db.traitfeedbacklog),
        ("proactive_logs", db.proactivechatlog),
    ]:
        try:
            cnt = await model.delete_many(where={"agentId": agent_id})
            stats[model_name] = cnt
        except Exception:
            pass

    try:
        cnt = await db.execute_raw(
            "DELETE FROM proactive_event_logs WHERE agent_id = $1",
            agent_id,
        )
        stats["proactive_event_logs"] = cnt or 0
    except Exception:
        pass

    try:
        cnt = await db.execute_raw(
            "DELETE FROM proactive_states WHERE agent_id = $1",
            agent_id,
        )
        stats["proactive_states"] = cnt or 0
    except Exception:
        pass

    try:
        cnt = await db.timetrigger.delete_many(where={"aiAgentId": agent_id})
        stats["triggers"] = cnt
    except Exception:
        pass

    try:
        cnt = await db.userportrait.delete_many(
            where={"agentId": agent_id, "userId": user_id},
        )
        stats["portraits"] = cnt
    except Exception:
        pass

    # ScheduleAdjustLog (no FK, raw SQL)
    try:
        cnt = await db.execute_raw(
            "DELETE FROM schedule_adjust_logs WHERE agent_id = $1", agent_id,
        )
        stats["schedule_logs"] = cnt or 0
    except Exception:
        pass

    # 7. 清理可能遗漏的无 workspace 的 memories (workspaceId=null, userId匹配)
    try:
        cnt = await db.execute_raw(
            "DELETE FROM memories_user WHERE user_id = $1 AND workspace_id IS NULL "
            "AND id IN (SELECT mu.id FROM memories_user mu "
            "JOIN conversations c ON c.user_id = mu.user_id "
            "WHERE c.agent_id = $2 AND mu.workspace_id IS NULL "
            "GROUP BY mu.id)",
            user_id, agent_id,
        )
        stats["orphan_user_memories"] = cnt or 0
    except Exception:
        pass

    # 8. 删除 Agent 本身
    await db.aiagent.delete(where={"id": agent_id})
    stats["agent"] = 1

    # 8. 清理 Redis
    stats["redis"] = await _clear_redis(agent_id, user_id, conv_ids)

    # 9. 清理 Neo4j（物理删除）
    stats["neo4j"] = await _hard_delete_neo4j(workspace_ids, agent_id, user_id)

    logger.info(f"Hard deleted agent={agent_id} user={user_id}: {stats}")
    return stats


async def _hard_delete_neo4j(
    workspace_ids: list[str], agent_id: str, user_id: str,
) -> int:
    """物理删除 Neo4j 中该 agent 相关的所有节点和关系。"""
    deleted = 0
    queries = [
        (
            "MATCH (n {agent_id: $agent_id}) DETACH DELETE n",
            {"agent_id": agent_id},
        ),
    ]
    for ws_id in workspace_ids:
        queries.append((
            "MATCH (n {workspace_id: $ws_id}) DETACH DELETE n",
            {"ws_id": ws_id},
        ))
    for query, params in queries:
        try:
            await run_write(query, params)
            deleted += 1
        except Exception as e:
            logger.warning(f"Neo4j hard delete failed: {e}")
    return deleted
