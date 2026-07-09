"""运行时状态清理服务。

用于工作区归档时清理旧 agent 的缓存、图谱状态和触发器，保留 PostgreSQL 历史数据。
"""

import logging
from typing import Any

from app.db import db
from app.redis_client import get_redis
from app.services.chat_media import storage as chat_media_storage
from app.services.offline import activity_media_storage

logger = logging.getLogger(__name__)


def _row_value(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key, None)


def _delete_chat_media_files(storage_keys: list[str]) -> int:
    deleted = 0
    for storage_key in sorted({key for key in storage_keys if key}):
        try:
            path = chat_media_storage.storage_path(storage_key)
            existed = path.exists() and path.is_file()
            chat_media_storage.delete_media_file(storage_key)
            if existed and not path.exists():
                deleted += 1
        except Exception as exc:
            logger.warning(
                "[chat-media] failed to delete storage_key=%s during agent reset: %s",
                storage_key,
                exc,
            )
    return deleted


def _delete_offline_media_files(storage_keys: list[str]) -> int:
    deleted = 0
    for storage_key in sorted({key for key in storage_keys if key}):
        try:
            path = activity_media_storage.storage_path(storage_key)
            existed = path.exists() and path.is_file()
            activity_media_storage.delete_media_file(storage_key)
            if existed and not path.exists():
                deleted += 1
        except Exception as exc:
            logger.warning(
                "[offline-media] failed to delete storage_key=%s during data reset: %s",
                storage_key,
                exc,
            )
    return deleted


def _add_count(stats: dict[str, int], key: str, count: int | None) -> None:
    stats[key] = stats.get(key, 0) + int(count or 0)


def _merge_stats(stats: dict[str, int], extra: dict[str, int]) -> None:
    for key, count in extra.items():
        _add_count(stats, key, count)


async def _execute_counted(
    stats: dict[str, int],
    key: str,
    sql: str,
    *args: Any,
) -> None:
    try:
        cnt = await db.execute_raw(sql, *args)
        _add_count(stats, key, cnt or 0)
    except Exception as exc:
        logger.warning("Data reset SQL failed for %s: %s", key, exc)


async def _delete_chat_media_for_conversations(
    *, user_id: str, conversation_ids: list[str],
) -> dict[str, int]:
    if not conversation_ids:
        return {}

    stats: dict[str, int] = {}
    storage_keys: list[str] = []

    attachment_rows = await db.query_raw(
        """
        DELETE FROM chat_message_attachments
        WHERE user_id = $1
          AND conversation_id = ANY($2::text[])
        RETURNING storage_key
        """,
        user_id,
        conversation_ids,
    )
    stats["chat_attachments"] = len(attachment_rows or [])
    storage_keys.extend(
        str(key)
        for row in (attachment_rows or [])
        if (key := _row_value(row, "storage_key"))
    )

    cover_rows = await db.query_raw(
        """
        SELECT metadata ->> 'cover_storage_key' AS storage_key
        FROM chat_link_cards
        WHERE user_id = $1
          AND conversation_id = ANY($2::text[])
          AND metadata ? 'cover_storage_key'
        """,
        user_id,
        conversation_ids,
    )
    stats["chat_link_cover_media"] = len(cover_rows or [])
    storage_keys.extend(
        str(key)
        for row in (cover_rows or [])
        if (key := _row_value(row, "storage_key"))
    )

    stats["chat_media_files"] = _delete_chat_media_files(storage_keys)
    return stats


async def _delete_offline_activities_for_agent(
    *, user_id: str, agent_id: str,
) -> dict[str, int]:
    stats: dict[str, int] = {}
    media_rows = await db.query_raw(
        """
        DELETE FROM offline_activity_media
        WHERE user_id = $1
          AND recommendation_id IN (
            SELECT id
            FROM offline_activity_recommendations
            WHERE user_id = $1 AND agent_id = $2
          )
        RETURNING storage_key
        """,
        user_id,
        agent_id,
    )
    storage_keys = [
        str(key)
        for row in (media_rows or [])
        if (key := _row_value(row, "storage_key"))
    ]
    stats["offline_activity_media"] = len(media_rows or [])
    stats["offline_activity_media_files"] = _delete_offline_media_files(storage_keys)

    await _execute_counted(
        stats,
        "offline_activity_feedback",
        """
        DELETE FROM offline_activity_feedback
        WHERE user_id = $1
          AND recommendation_id IN (
            SELECT id
            FROM offline_activity_recommendations
            WHERE user_id = $1 AND agent_id = $2
          )
        """,
        user_id,
        agent_id,
    )
    await _execute_counted(
        stats,
        "offline_activity_recommendations",
        """
        DELETE FROM offline_activity_recommendations
        WHERE user_id = $1 AND agent_id = $2
        """,
        user_id,
        agent_id,
    )
    return stats


async def _delete_agent_auxiliary_rows(
    *,
    user_id: str,
    agent_id: str,
    workspace_ids: list[str],
    conversation_ids: list[str],
) -> dict[str, int]:
    """Delete side tables that do not have complete FK cascade coverage."""
    stats: dict[str, int] = {}

    if conversation_ids:
        await _execute_counted(
            stats,
            "message_traces",
            """
            DELETE FROM message_traces
            WHERE conversation_id = ANY($1::text[])
            """,
            conversation_ids,
        )

    await _execute_counted(
        stats,
        "llm_usage",
        """
        DELETE FROM llm_usage
        WHERE agent_id = $1
           OR (
                user_id = $2
                AND (
                    ($3::text[] IS NOT NULL AND conversation_id = ANY($3::text[]))
                    OR agent_id = $1
                )
           )
        """,
        agent_id,
        user_id,
        conversation_ids,
    )
    await _execute_counted(
        stats,
        "memory_visible_use_events",
        """
        DELETE FROM memory_visible_use_events
        WHERE agent_id = $1
           OR (
                user_id = $2
                AND (
                    ($3::text[] IS NOT NULL AND workspace_id = ANY($3::text[]))
                    OR ($4::text[] IS NOT NULL AND conversation_id = ANY($4::text[]))
                )
           )
        """,
        agent_id,
        user_id,
        workspace_ids,
        conversation_ids,
    )
    await _execute_counted(
        stats,
        "crisis_events",
        """
        DELETE FROM crisis_events
        WHERE agent_id = $1
           OR (
                user_id = $2
                AND (
                    ($3::text[] IS NOT NULL AND workspace_id = ANY($3::text[]))
                    OR ($4::text[] IS NOT NULL AND conversation_id = ANY($4::text[]))
                )
           )
        """,
        agent_id,
        user_id,
        workspace_ids,
        conversation_ids,
    )
    await _execute_counted(
        stats,
        "memory_repair_items",
        """
        DELETE FROM memory_repair_items
        WHERE agent_id = $1
           OR (
                user_id = $2
                AND (
                    ($3::text[] IS NOT NULL AND workspace_id = ANY($3::text[]))
                    OR ($4::text[] IS NOT NULL AND conversation_id = ANY($4::text[]))
                )
           )
        """,
        agent_id,
        user_id,
        workspace_ids,
        conversation_ids,
    )
    if workspace_ids:
        await _execute_counted(
            stats,
            "memory_quality_states",
            """
            DELETE FROM memory_quality_states
            WHERE user_id = $1 AND workspace_id = ANY($2::text[])
            """,
            user_id,
            workspace_ids,
        )
        await _execute_counted(
            stats,
            "memory_consolidation_runs",
            """
            DELETE FROM memory_consolidation_runs
            WHERE user_id = $1 AND workspace_id = ANY($2::text[])
            """,
            user_id,
            workspace_ids,
        )
        await _execute_counted(
            stats,
            "memory_entities",
            """
            DELETE FROM memory_entities
            WHERE user_id = $1 AND workspace_id = ANY($2::text[])
            """,
            user_id,
            workspace_ids,
        )

    await _execute_counted(
        stats,
        "notification_events",
        """
        DELETE FROM notification_events
        WHERE user_id = $1 AND agent_id = $2
        """,
        user_id,
        agent_id,
    )
    return stats


async def _delete_remaining_user_chat_data(user_id: str) -> dict[str, int]:
    stats: dict[str, int] = {}
    conversations = await db.conversation.find_many(where={"userId": user_id})
    conv_ids = [c.id for c in conversations]
    workspaces = await db.chatworkspace.find_many(where={"userId": user_id})
    workspace_ids = [w.id for w in workspaces]

    _merge_stats(
        stats,
        await _delete_chat_media_for_conversations(
            user_id=user_id,
            conversation_ids=conv_ids,
        ),
    )

    orphan_attachment_rows = await db.query_raw(
        """
        DELETE FROM chat_message_attachments
        WHERE user_id = $1
        RETURNING storage_key
        """,
        user_id,
    )
    orphan_storage_keys = [
        str(key)
        for row in (orphan_attachment_rows or [])
        if (key := _row_value(row, "storage_key"))
    ]
    _add_count(stats, "chat_attachments", len(orphan_attachment_rows or []))

    if conv_ids:
        remaining_cover_rows = await db.query_raw(
            """
            SELECT metadata ->> 'cover_storage_key' AS storage_key
            FROM chat_link_cards
            WHERE user_id = $1
              AND NOT (conversation_id = ANY($2::text[]))
              AND metadata ? 'cover_storage_key'
            """,
            user_id,
            conv_ids,
        )
    else:
        remaining_cover_rows = await db.query_raw(
            """
            SELECT metadata ->> 'cover_storage_key' AS storage_key
            FROM chat_link_cards
            WHERE user_id = $1
              AND metadata ? 'cover_storage_key'
            """,
            user_id,
        )
    remaining_cover_keys = [
        str(key)
        for row in (remaining_cover_rows or [])
        if (key := _row_value(row, "storage_key"))
    ]
    _add_count(stats, "chat_link_cover_media", len(remaining_cover_rows or []))
    _add_count(
        stats,
        "chat_media_files",
        _delete_chat_media_files([*orphan_storage_keys, *remaining_cover_keys]),
    )

    if conv_ids:
        await _execute_counted(
            stats,
            "message_traces",
            "DELETE FROM message_traces WHERE conversation_id = ANY($1::text[])",
            conv_ids,
        )
        await _execute_counted(
            stats,
            "bug_reports",
            """
            DELETE FROM bug_reports
            WHERE message_id IN (
                SELECT id FROM messages WHERE conversation_id = ANY($1::text[])
            )
            """,
            conv_ids,
        )
        cnt = await db.message.delete_many(
            where={"conversationId": {"in": conv_ids}},
        )
        stats["messages"] = cnt
        cnt = await db.conversation.delete_many(where={"id": {"in": conv_ids}})
        stats["conversations"] = cnt

    if workspace_ids:
        cnt = await db.chatworkspace.delete_many(where={"id": {"in": workspace_ids}})
        stats["workspaces"] = cnt
    return stats


async def _delete_remaining_user_memory_data(user_id: str) -> dict[str, int]:
    stats: dict[str, int] = {}
    user_mems = await db.usermemory.find_many(where={"userId": user_id})
    ai_mems = await db.aimemory.find_many(where={"userId": user_id})
    mem_ids = [m.id for m in user_mems] + [m.id for m in ai_mems]

    if mem_ids:
        await _execute_counted(
            stats,
            "embeddings",
            "DELETE FROM memory_embeddings WHERE memory_id = ANY($1::text[])",
            mem_ids,
        )
        await _execute_counted(
            stats,
            "memory_mentions",
            "DELETE FROM memory_mentions WHERE memory_id = ANY($1::text[])",
            mem_ids,
        )

    cnt = await db.usermemory.delete_many(where={"userId": user_id})
    stats["user_memories"] = cnt
    cnt = await db.aimemory.delete_many(where={"userId": user_id})
    stats["ai_memories"] = cnt

    for key, sql in [
        ("profiles", "DELETE FROM user_profiles WHERE user_id = $1"),
        ("portraits", "DELETE FROM user_portraits WHERE user_id = $1"),
        ("profile_tags", "DELETE FROM user_profile_tags WHERE user_id = $1"),
        ("changelogs", "DELETE FROM memory_changelogs WHERE user_id = $1"),
        ("memory_entities", "DELETE FROM memory_entities WHERE user_id = $1"),
        ("memory_mentions", "DELETE FROM memory_mentions WHERE user_id = $1"),
        ("memory_quality_states", "DELETE FROM memory_quality_states WHERE user_id = $1"),
        ("memory_repair_items", "DELETE FROM memory_repair_items WHERE user_id = $1"),
        ("memory_consolidation_runs", "DELETE FROM memory_consolidation_runs WHERE user_id = $1"),
    ]:
        await _execute_counted(stats, key, sql, user_id)
    return stats


async def _delete_remaining_user_side_tables(user_id: str) -> dict[str, int]:
    stats: dict[str, int] = {}

    media_rows = await db.query_raw(
        """
        DELETE FROM offline_activity_media
        WHERE user_id = $1
        RETURNING storage_key
        """,
        user_id,
    )
    storage_keys = [
        str(key)
        for row in (media_rows or [])
        if (key := _row_value(row, "storage_key"))
    ]
    stats["offline_activity_media"] = len(media_rows or [])
    stats["offline_activity_media_files"] = _delete_offline_media_files(storage_keys)
    stats["offline_user_media_files"] = activity_media_storage.delete_user_media_files(user_id)

    for key, sql in [
        ("bug_reports_resolved", "UPDATE bug_reports SET resolved_by_id = NULL WHERE resolved_by_id = $1"),
        ("bug_reports_filed", "DELETE FROM bug_reports WHERE reporter_id = $1"),
        ("last_will_deliveries", "DELETE FROM last_will_deliveries WHERE last_will_id IN (SELECT id FROM last_wills WHERE user_id = $1)"),
        ("last_wills", "DELETE FROM last_wills WHERE user_id = $1"),
        ("time_capsules", "DELETE FROM time_capsules WHERE user_id = $1"),
        ("offline_activity_feedback", "DELETE FROM offline_activity_feedback WHERE user_id = $1"),
        ("offline_activity_recommendations", "DELETE FROM offline_activity_recommendations WHERE user_id = $1"),
        ("gift_tracking_events", "DELETE FROM gift_tracking_events WHERE gift_id IN (SELECT id FROM real_world_gifts WHERE user_id = $1)"),
        ("real_world_gifts", "DELETE FROM real_world_gifts WHERE user_id = $1"),
        ("real_world_trigger_states", "DELETE FROM real_world_trigger_states WHERE user_id = $1"),
        ("real_world_recharge_ledger", "DELETE FROM real_world_recharge_ledger WHERE user_id = $1"),
        ("gift_addresses", "DELETE FROM gift_addresses WHERE user_id = $1"),
        ("wallet_ledger", "DELETE FROM wallet_ledger WHERE user_id = $1"),
        ("user_wallets", "DELETE FROM user_wallets WHERE user_id = $1"),
        ("notification_events", "DELETE FROM notification_events WHERE user_id = $1"),
        ("push_devices", "DELETE FROM push_devices WHERE user_id = $1"),
        ("achievement_events", "DELETE FROM achievement_events WHERE user_id = $1"),
        ("achievement_unlocks", "DELETE FROM achievement_unlocks WHERE user_id = $1"),
        ("user_daily_activity", "DELETE FROM user_daily_activity WHERE user_id = $1"),
        ("game_sessions", "DELETE FROM game_sessions WHERE user_id = $1"),
        ("music_co_listening_sessions", "DELETE FROM music_co_listening_sessions WHERE user_id = $1"),
        ("music_playbacks", "DELETE FROM music_playbacks WHERE user_id = $1"),
        ("music_favorites", "DELETE FROM music_favorites WHERE user_id = $1"),
        ("time_triggers", "DELETE FROM time_triggers WHERE user_id = $1"),
        ("patience_states", "DELETE FROM patience_states WHERE user_id = $1"),
        ("intimacies", "DELETE FROM intimacies WHERE user_id = $1"),
        ("proactive_event_logs", "DELETE FROM proactive_event_logs WHERE user_id = $1"),
        ("proactive_states", "DELETE FROM proactive_states WHERE user_id = $1"),
        ("proactive_chat_logs", "DELETE FROM proactive_chat_logs WHERE user_id = $1"),
        ("proactive_counters", "DELETE FROM proactive_counters WHERE user_id = $1"),
        ("memory_visible_use_events", "DELETE FROM memory_visible_use_events WHERE user_id = $1"),
        ("crisis_events", "DELETE FROM crisis_events WHERE user_id = $1"),
        ("llm_usage", "DELETE FROM llm_usage WHERE user_id = $1"),
        ("auth_identities", "DELETE FROM auth_identities WHERE user_id = $1"),
        ("meal_vouchers", "DELETE FROM meal_vouchers WHERE user_id = $1"),
    ]:
        await _execute_counted(stats, key, sql, user_id)
    return stats


async def _clear_user_redis(user_id: str) -> int:
    redis = await get_redis()
    deleted = 0
    patterns = [
        f"*:{user_id}",
        f"*:{user_id}:*",
        f"*:{user_id}_*",
        f"login_fail:*:{user_id}",
        f"register_rate:*:{user_id}",
    ]
    for pattern in patterns:
        try:
            cursor = 0
            while True:
                cursor, keys = await redis.scan(cursor, match=pattern, count=100)
                if keys:
                    deleted += await redis.delete(*keys)
                if cursor == 0:
                    break
        except Exception as exc:
            logger.warning("Redis user cleanup failed for %s: %s", pattern, exc)
    return deleted


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
    stats["redis"] = await _clear_redis(agent_id, user_id, conv_ids)
    _ = workspace_id  # reserved for future per-workspace runtime state cleanup

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


async def _clear_redis(agent_id: str, user_id: str, conv_ids: list[str]) -> int:
    """删除所有相关 Redis 键。"""
    redis = await get_redis()

    # 精确 key 列表
    exact_keys = [
        f"life_overview:{agent_id}",
        f"patience:{agent_id}:{user_id}",
        f"attack_history:{agent_id}:{user_id}",
        f"trigger_last:{agent_id}:{user_id}",
        f"intimacy:{agent_id}:{user_id}",
        f"topic_intimacy:{agent_id}:{user_id}",
        f"pending:msgs:{agent_id}:{user_id}",
        f"pending:conv:{agent_id}:{user_id}",
        f"pending:ctx:{agent_id}:{user_id}",
        f"last_reply:{agent_id}:{user_id}",
        f"proactive_2day:{agent_id}:{user_id}",
        f"attack_history:{agent_id}:{user_id}:L0",
        f"attack_history:{agent_id}:{user_id}:L1",
        f"attack_history:{agent_id}:{user_id}:L2",
        f"memgen:lock:{agent_id}",
        f"memgen:report:{agent_id}",
        f"provision_progress:{agent_id}",
    ]

    # conversation 相关的精确 key
    for cid in conv_ids:
        exact_keys.append(f"topics:{cid}")
        exact_keys.append(f"context_window:{cid}")
        exact_keys.append(f"delayed:msgs:{cid}")

    # 通配符 patterns（需要 SCAN）
    scan_patterns = [
        f"schedule:{agent_id}:*",
        f"schedule_adj:{agent_id}:*",
        f"trait_adj:{agent_id}:*",
        f"trait_adj_week:{agent_id}:*",
        f"trigger_count:{agent_id}:{user_id}:*",
        f"proactive_count:{agent_id}:{user_id}:*",
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

    # aggregation pending ZSET: 成员格式 "{agent_id}:{user_id}"
    try:
        deleted += await redis.zrem("pending:delayed", f"{agent_id}:{user_id}")
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

    # 3. 先清理文件和旁路表。DB 外键不会覆盖磁盘文件 / 无 FK 诊断表。
    _merge_stats(
        stats,
        await _delete_chat_media_for_conversations(
            user_id=user_id,
            conversation_ids=conv_ids,
        ),
    )
    _merge_stats(
        stats,
        await _delete_agent_auxiliary_rows(
            user_id=user_id,
            agent_id=agent_id,
            workspace_ids=workspace_ids,
            conversation_ids=conv_ids,
        ),
    )
    _merge_stats(
        stats,
        await _delete_offline_activities_for_agent(user_id=user_id, agent_id=agent_id),
    )

    # 4. 删除 messages
    if conv_ids:
        cnt = await db.message.delete_many(
            where={"conversationId": {"in": conv_ids}},
        )
        stats["messages"] = cnt

    # 5. 删除 conversations
    if conv_ids:
        cnt = await db.conversation.delete_many(
            where={"id": {"in": conv_ids}},
        )
        stats["conversations"] = cnt

    # 6. 删除 workspace 下的 memories + embeddings
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

    # 7. 删除 agent 级别数据
    try:
        cnt = await db.intimacy.delete_many(
            where={"agentId": agent_id, "userId": user_id},
        )
        stats["intimacy"] = cnt
    except Exception:
        pass

    for model_name, model in [
        ("schedules", db.aidailyschedule),
        ("trait_logs", db.traitfeedbacklog),
        ("proactive_logs", db.proactivechatlog),
        ("proactive_counters", db.proactivecounter),
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

    # 8. 清理可能遗漏的无 workspace 的 memories (workspaceId=null, userId匹配)
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

    # 9. 清除所有引用 agent_id 的表（防止 FK 约束阻止 agent 删除）
    # 逐表列举容易遗漏，这里用通用列表一次性处理
    _FK_TABLES_TO_DELETE = [
        "patience_states",
        "ai_daily_schedules",
        "trait_feedback_logs",
        "intimacies",
    ]
    for table in _FK_TABLES_TO_DELETE:
        try:
            cnt = await db.execute_raw(
                f'DELETE FROM "{table}" WHERE "agent_id" = $1', agent_id,
            )
            stats[table] = cnt or 0
        except Exception:
            pass

    # Plan B 后已无 character_profiles 表 (DROP 见 migration 20260427180000),
    # 旧 UPDATE 解绑逻辑随之失效, 此处不再需要任何 hook.

    # 10. 删除 Agent 本身
    await db.aiagent.delete(where={"id": agent_id})
    stats["agent"] = 1

    # 11. 清理 Redis
    stats["redis"] = await _clear_redis(agent_id, user_id, conv_ids)

    # 12. 清理 entity knowledge layer (memory_mentions 由 memories_* 删除
    # 触发器自动级联；memory_entities 按 workspace + user 清干净即可)
    if workspace_ids:
        try:
            cnt = await db.execute_raw(
                "DELETE FROM memory_entities WHERE user_id = $1 "
                "AND workspace_id = ANY($2::text[])",
                user_id, workspace_ids,
            )
            stats["entities"] = cnt or 0
        except Exception as e:
            logger.warning(f"Entity cleanup failed: {e}")

    logger.info(f"Hard deleted agent={agent_id} user={user_id}: {stats}")
    return stats


async def hard_delete_user_data(user_id: str) -> dict[str, int]:
    """彻底物理删除某个用户及其所有 Agent、运行时、媒体和旁路数据。"""
    stats: dict[str, int] = {}

    agents = await db.aiagent.find_many(where={"userId": user_id})
    agent_ids = [agent.id for agent in agents]
    stats["agents_found"] = len(agent_ids)
    for agent_id in agent_ids:
        _merge_stats(stats, await hard_delete_agent_data(agent_id, user_id))

    _merge_stats(stats, await _delete_remaining_user_chat_data(user_id))
    _merge_stats(stats, await _delete_remaining_user_memory_data(user_id))
    _merge_stats(stats, await _delete_remaining_user_side_tables(user_id))

    # Any agent row left here would be an orphaned edge case after partial legacy data.
    try:
        cnt = await db.aiagent.delete_many(where={"userId": user_id})
        stats["remaining_agents"] = cnt
    except Exception as exc:
        logger.warning("Remaining agent delete failed for user=%s: %s", user_id, exc)

    await db.user.delete(where={"id": user_id})
    stats["user"] = 1
    stats["redis"] = stats.get("redis", 0) + await _clear_user_redis(user_id)

    logger.info("Hard deleted user=%s: %s", user_id, stats)
    return stats
