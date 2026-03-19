"""数据重置服务。

彻底清除某个 AI Agent 及其关联用户的所有数据，包括 PostgreSQL、Neo4j、Redis。
"""

import logging

from app.db import db
from app.neo4j_client import run_write
from app.redis_client import get_redis

logger = logging.getLogger(__name__)


async def reset_agent_data(agent_id: str, user_id: str) -> dict[str, int]:
    """清除指定 agent + user 的所有数据。返回各阶段删除计数。

    调用方负责最后删除 AiAgent 和 User 记录本身。
    """
    stats: dict[str, int] = {}

    # 1. 查出所有 conversation_id（后面清 Redis 要用）
    convs = await db.query_raw(
        "SELECT id FROM conversations WHERE agent_id = $1",
        agent_id,
    )
    conv_ids = [c["id"] for c in (convs or [])]

    # 2. PostgreSQL — 按外键依赖顺序删除
    stats["postgres"] = await _clear_postgres(agent_id, user_id, conv_ids)

    # 3. Neo4j — 删除图谱节点和关系
    stats["neo4j"] = await _clear_neo4j(agent_id, user_id)

    # 4. Redis — 删除所有缓存键
    stats["redis"] = await _clear_redis(agent_id, user_id, conv_ids)

    logger.info(f"Data reset complete for agent={agent_id} user={user_id}: {stats}")
    return stats


async def _clear_postgres(agent_id: str, user_id: str, conv_ids: list[str]) -> int:
    """按外键依赖顺序删除 PostgreSQL 数据。"""
    total = 0

    # 子表 → 父表顺序
    queries = [
        # messages (通过 conversation)
        (
            "DELETE FROM messages WHERE conversation_id = ANY($1)",
            conv_ids,
        ),
        # conversation 相关
        ("DELETE FROM conversations WHERE agent_id = $1", agent_id),
        ("DELETE FROM proactive_chat_logs WHERE agent_id = $1", agent_id),
        ("DELETE FROM trait_feedback_logs WHERE agent_id = $1", agent_id),
        ("DELETE FROM schedule_adjust_logs WHERE agent_id = $1", agent_id),
        ("DELETE FROM ai_daily_schedules WHERE agent_id = $1", agent_id),
        ("DELETE FROM ai_emotion_states WHERE agent_id = $1", agent_id),
        ("DELETE FROM user_portraits WHERE agent_id = $1", agent_id),
        ("DELETE FROM intimacies WHERE agent_id = $1", agent_id),
        ("DELETE FROM time_triggers WHERE ai_agent_id = $1", agent_id),
        # memory embeddings (通过 memory_id)
        (
            "DELETE FROM memory_embeddings WHERE memory_id IN "
            "(SELECT id FROM memories_user WHERE user_id = $1 "
            "UNION SELECT id FROM memories_ai WHERE user_id = $1)",
            user_id,
        ),
        ("DELETE FROM memory_changelogs WHERE user_id = $1", user_id),
        ("DELETE FROM memories_user WHERE user_id = $1", user_id),
        ("DELETE FROM memories_ai WHERE user_id = $1", user_id),
        ("DELETE FROM user_profiles WHERE user_id = $1", user_id),
    ]

    for sql, param in queries:
        try:
            cnt = await db.execute_raw(sql, param)
            total += cnt or 0
        except Exception as e:
            logger.warning(f"PG delete failed: {sql[:60]}... — {e}")

    return total


async def _clear_neo4j(agent_id: str, user_id: str) -> int:
    """删除 Neo4j 中 AI 和 User 节点及其所有关系。"""
    deleted = 0
    try:
        await run_write(
            "MATCH (a:AI {id: $id}) DETACH DELETE a",
            {"id": agent_id},
        )
        deleted += 1
    except Exception as e:
        logger.warning(f"Neo4j AI node delete failed: {e}")

    try:
        await run_write(
            "MATCH (u:User {id: $id}) DETACH DELETE u",
            {"id": user_id},
        )
        deleted += 1
    except Exception as e:
        logger.warning(f"Neo4j User node delete failed: {e}")

    return deleted


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
