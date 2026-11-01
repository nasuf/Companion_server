"""One scoped transaction for reminder state and its memory projection.

The trigger snapshot is a CAS token. A scheduler holding an old due row cannot
consume a reminder that the user has just moved, cancelled, or completed.
"""

from datetime import timedelta
import json

from app.services.memory.evidence import _resolve_source
from app.services.memory.input_evidence import (
    prepare_input_origin,
    persist_input_origin,
)
from app.services.memory.lifecycle.lazy_update import scope_lock_key
from app.services.memory.manual_write import (
    MemoryWriteError,
    lock_actor,
    lock_memories,
    memory_scope,
)
from app.services.memory.storage import repo
from app.services.memory.storage.embedding import generate_embedding, store_embedding


def trigger_version(trigger):
    return {
        key: getattr(trigger, key)
        for key in (
            "id",
            "userId",
            "aiAgentId",
            "actionType",
            "triggerTime",
            "isActive",
            "lastFired",
            "actionData",
        )
    }


async def mutate_reminder(
    database,
    *,
    trigger,
    trigger_changes,
    memory_changes=None,
    operation,
    actor=None,
    sources=(),
):
    memory_changes = memory_changes or {}
    if set(memory_changes) - {
        "content",
        "occurTime",
        "recurrence",
        "isArchived",
    } or set(trigger_changes) - {"triggerTime", "actionData", "isActive", "lastFired"}:
        raise MemoryWriteError(422, "unsupported_reminder_mutation")
    if memory_changes and not actor and not sources:
        raise MemoryWriteError(409, "reminder_mutation_source_required")
    if len(sources) > 40:
        raise MemoryWriteError(422, "reminder_mutation_source_limit")
    data = trigger.actionData or {}
    side = data.get("memory_side") or "user"
    mid = data.get("memory_id")
    if side not in {"user", "ai"} or not isinstance(mid, str) or not mid:
        raise MemoryWriteError(409, "reminder_memory_unavailable")
    model = database.usermemory if side == "user" else database.aimemory
    row = await model.find_unique(where={"id": mid})
    if row is None:
        if actor and set(memory_changes) <= {"isArchived"} and memory_changes.get("isArchived", True):
            # The memory is already gone: an owner may still pin, complete or
            # delete the leftover trigger, but never revive or retime it.
            return await _mutate_orphan_trigger(
                database, trigger=trigger, trigger_changes=trigger_changes,
                operation=operation, actor=actor)
        raise MemoryWriteError(409, "reminder_memory_unavailable")
    record = repo._to_record(row, side)
    if (
        record.userId != trigger.userId
        or record.subCategory != "提醒"
        or not record.workspaceId
    ):
        raise MemoryWriteError(409, "reminder_memory_scope_changed")
    expected_scope = await memory_scope(database, record)
    if (
        expected_scope["agent_id"] != trigger.aiAgentId
        or trigger.actionType != "reminder"
    ):
        raise MemoryWriteError(409, "reminder_agent_scope_changed")
    if actor and (
        actor.get("role") not in {"user", "admin"}
        or (actor.get("role") != "admin" and actor.get("sub") != trigger.userId)
    ):
        raise MemoryWriteError(403, "reminder_actor_invalid")
    changed_content = (
        "content" in memory_changes and memory_changes["content"] != record.content
    )
    embedding = (
        await generate_embedding(memory_changes["content"]) if changed_content else None
    )
    async with database.tx(timeout=timedelta(seconds=20)) as tx:
        await tx.execute_raw(
            "SELECT pg_advisory_xact_lock($1::bigint)",
            scope_lock_key(side, record.userId, record.workspaceId),
        )
        if await memory_scope(tx, record, lock=True) != expected_scope:
            raise MemoryWriteError(409, "reminder_scope_changed")
        if actor:
            await lock_actor(tx, actor["sub"], actor["role"])
        else:
            await tx.query_raw(
                "SELECT id FROM human_takeovers WHERE user_id=$1 ORDER BY id FOR SHARE",
                record.userId,
            )
            if await tx.query_raw(
                "SELECT id FROM human_takeovers WHERE user_id=$1 AND status='active' LIMIT 1",
                record.userId,
            ):
                raise MemoryWriteError(409, "reminder_human_takeover")
        for source in sources:
            if (
                source.kind != "message"
                or source.expected_role != "user"
                or not source.expected_version
                or not source.expected_statement_time
            ):
                raise MemoryWriteError(409, "reminder_mutation_source_invalid")
            await _resolve_source(
                tx,
                source,
                {
                    "user_id": record.userId,
                    "workspace_id": record.workspaceId,
                    "agent_id": trigger.aiAgentId,
                },
            )
        if sources:
            rows = await tx.query_raw(
                "SELECT id FROM messages WHERE id=ANY($1::text[]) AND NOT COALESCE(metadata,'{}'::jsonb) ? 'component_card' FOR SHARE",
                [x.ref for x in sources],
            )
            if len(rows) != len({x.ref for x in sources}):
                raise MemoryWriteError(409, "reminder_mutation_source_invalid")
        proposed_data = trigger_changes.get("actionData")
        if proposed_data is not None:
            proposed_data = getattr(proposed_data, "data", proposed_data)
            conversation_id = proposed_data.get("conversation_id")
            if conversation_id:
                allowed = await tx.query_raw(
                    """SELECT id FROM conversations WHERE id=$1 AND user_id=$2
                    AND agent_id=$3 AND workspace_id=$4 AND NOT is_deleted
                    AND archived_at IS NULL FOR SHARE""",
                    conversation_id,
                    record.userId,
                    trigger.aiAgentId,
                    record.workspaceId,
                )
                if not allowed:
                    raise MemoryWriteError(409, "reminder_conversation_scope_changed")
        await lock_memories(tx, [record], allow_archived=True)
        await tx.query_raw(
            "SELECT id FROM time_triggers WHERE id=$1 FOR UPDATE", trigger.id
        )
        current = await tx.timetrigger.find_unique(where={"id": trigger.id})
        if current is None or trigger_version(current) != trigger_version(trigger):
            raise MemoryWriteError(409, "reminder_changed_refresh_and_retry")
        evidence = tuple(sources)
        if actor and memory_changes:
            origin = prepare_input_origin(
                kind="admin_edit" if actor["role"] == "admin" else "user_edit",
                payload={
                    "operation": operation,
                    "actor_id": actor["sub"],
                    "actor_role": actor["role"],
                    "trigger_id": trigger.id,
                    "memory_id": mid,
                    "memory_side": side,
                    "old_time": (
                        record.occurTime.isoformat() if record.occurTime else None
                    ),
                    "changes": {
                        k: v.isoformat() if hasattr(v, "isoformat") else v
                        for k, v in memory_changes.items()
                    },
                },
            )
            source = await persist_input_origin(
                tx,
                user_id=record.userId,
                workspace_id=record.workspaceId,
                agent_id=trigger.aiAgentId,
                origin=origin,
            )
            evidence = (source, *evidence)
        if memory_changes:
            await repo.update(
                mid,
                source=side,
                record=record,
                database=tx,
                **memory_changes,
                **(
                    {
                        "evidence_sources": evidence,
                        "extractor_version": "reminder-mutation-v1",
                    }
                    if evidence
                    else {}
                ),
            )
        if changed_content:
            await store_embedding(mid, embedding, database=tx)
        updated = await tx.timetrigger.update(
            where={"id": trigger.id}, data=trigger_changes
        )
        await tx.memorychangelog.create(
            data={
                "userId": record.userId,
                "workspaceId": record.workspaceId,
                "memoryId": mid,
                "operation": operation,
                "newValue": json.dumps(
                    {
                        "trigger_id": trigger.id,
                        "actor_id": actor.get("sub") if actor else None,
                        "origin": "authenticated_action" if actor else "scheduler",
                        "source_ids": [s.ref for s in sources],
                    }
                ),
            }
        )
    if memory_changes:
        await repo.invalidate_scope(record.userId, record.workspaceId)
    return updated


async def _mutate_orphan_trigger(database, *, trigger, trigger_changes, operation, actor):
    if (
        trigger.actionType != "reminder"
        or set(trigger_changes) - {"actionData", "isActive"}
        or trigger_changes.get("isActive", False)
        or actor.get("role") not in {"user", "admin"}
        or (actor.get("role") != "admin" and actor.get("sub") != trigger.userId)
    ):
        raise MemoryWriteError(409, "reminder_memory_unavailable")
    async with database.tx(timeout=timedelta(seconds=20)) as tx:
        await lock_actor(tx, actor["sub"], actor["role"])
        await tx.query_raw("SELECT id FROM time_triggers WHERE id=$1 FOR UPDATE", trigger.id)
        current = await tx.timetrigger.find_unique(where={"id": trigger.id})
        if current is None or trigger_version(current) != trigger_version(trigger):
            raise MemoryWriteError(409, "reminder_changed_refresh_and_retry")
        return await tx.timetrigger.update(where={"id": trigger.id}, data=trigger_changes)


async def claim_due_reminder(database, trigger, now):
    """Bounded SQL CAS; a stale scan cannot claim an updated reminder."""
    data = trigger.actionData or {}
    side = data.get("memory_side") or "user"
    if side not in {"user", "ai"} or not isinstance(data.get("memory_id"), str):
        return False
    table = "memories_user" if side == "user" else "memories_ai"
    rows = await database.query_raw(
        f"""UPDATE time_triggers t SET is_active=false,last_fired=$6::timestamptz AT TIME ZONE 'UTC'
        WHERE t.id=$1 AND t.user_id=$2 AND t.ai_agent_id=$3 AND t.action_type='reminder'
          AND t.is_active AND t.trigger_time=$4::timestamptz AT TIME ZONE 'UTC'
          AND t.action_data=$5::jsonb AND t.trigger_time<=$6::timestamptz AT TIME ZONE 'UTC'
          AND t.last_fired IS NOT DISTINCT FROM ($7::timestamptz AT TIME ZONE 'UTC')
          AND EXISTS (SELECT 1 FROM {table} m JOIN chat_workspaces w ON w.id=m.workspace_id
            JOIN ai_agents a ON a.id=w.agent_id JOIN users u ON u.id=w.user_id
            WHERE m.id=$8 AND m.user_id=t.user_id AND w.user_id=t.user_id AND w.agent_id=t.ai_agent_id
              AND NOT m.is_archived AND m.sub_category='提醒' AND w.status='active'
              AND a.status='active' AND u.status='active' AND w.archived_at IS NULL
              AND a.archived_at IS NULL AND u.archived_at IS NULL)
        RETURNING t.id""",
        trigger.id,
        trigger.userId,
        trigger.aiAgentId,
        trigger.triggerTime.isoformat(),
        json.dumps(data),
        now.isoformat(),
        trigger.lastFired.isoformat() if trigger.lastFired else None,
        data["memory_id"],
    )
    return bool(rows)


async def retry_claimed_reminder(
    database, trigger, *, claimed_at, retry_at, retry_count
):
    """Retry only this exact dispatch attempt; never undo a later user change.

    retry_at is transport scheduling, not a new occurrence or completion date.
    All predicates and the reactivation are one SQL statement.
    """
    if claimed_at is None or getattr(claimed_at, "tzinfo", None) is None:
        return False
    data = trigger.actionData or {}
    side = data.get("memory_side") or "user"
    if side not in {"user", "ai"} or not isinstance(data.get("memory_id"), str):
        return False
    table = "memories_user" if side == "user" else "memories_ai"
    rows = await database.query_raw(
        f"""UPDATE time_triggers t
        SET is_active=true,last_fired=NULL,trigger_time=$7::timestamptz AT TIME ZONE 'UTC',
          action_data=$8::jsonb
        WHERE t.id=$1 AND t.user_id=$2 AND t.ai_agent_id=$3 AND t.action_type='reminder'
          AND NOT t.is_active AND t.trigger_time=$4::timestamptz AT TIME ZONE 'UTC'
          AND t.action_data=$5::jsonb
          AND t.last_fired=($6::timestamptz AT TIME ZONE 'UTC')::timestamp(3)
          AND EXISTS(SELECT 1 FROM {table} m JOIN chat_workspaces w ON w.id=m.workspace_id
            JOIN ai_agents a ON a.id=w.agent_id JOIN users u ON u.id=w.user_id
            WHERE m.id=$9 AND m.user_id=t.user_id AND w.user_id=t.user_id
              AND w.agent_id=t.ai_agent_id AND NOT m.is_archived AND m.sub_category='提醒'
              AND w.status='active' AND a.status='active' AND u.status='active'
              AND w.archived_at IS NULL AND a.archived_at IS NULL AND u.archived_at IS NULL)
        RETURNING t.id""",
        trigger.id,
        trigger.userId,
        trigger.aiAgentId,
        trigger.triggerTime.isoformat(),
        json.dumps(data),
        claimed_at.isoformat(),
        retry_at.isoformat(),
        json.dumps({**data, "retry_count": retry_count}),
        data["memory_id"],
    )
    return bool(rows)
