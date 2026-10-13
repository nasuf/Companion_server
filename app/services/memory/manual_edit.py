"""One transactional boundary for authenticated manual memory edits.

Imported/edited text is a recorded input, never an assertion of verified truth.
Embedding generation occurs before row locks; all durable writes commit together.
"""
from __future__ import annotations

from datetime import timedelta
import json
import math

from app.db import db
from app.services.memory.evidence import bind_memory_evidence, content_version
from app.services.memory.input_evidence import prepare_input_origin, persist_input_origin
from app.services.memory.provenance import KNOWLEDGE_SEED
from app.services.memory.retrieval.knowledge_hits import invalidate_knowledge_rows
from app.services.memory.storage import repo as memory_repo
from app.services.memory.storage.embedding import generate_embedding, store_embedding
from app.services.memory.storage.persistence import after_memory_changelog


from app.services.memory.manual_write import (
    MemoryWriteError as MemoryEditError, memory_scope as _scope,
    lock_actor, lock_repair, lock_memories, resolve_repair,
)
from app.services.memory.lifecycle.lazy_update import scope_lock_key


async def edit_memory_with_evidence(*, record: memory_repo.MemoryRecord, content: str,
                                    actor: dict, level: int | None = None,
                                    importance: float | None = None,
                                    repair_item: dict | None = None,
                                    resolution_note: str | None = None,
                                    reason: str | None = None) -> tuple[memory_repo.MemoryRecord, dict | None]:
    """CAS against the observed memory and scope; stale concurrent edits fail 409."""
    if record.source not in {"user", "ai"} or not record.workspaceId:
        raise MemoryEditError(409, "memory_scope_changed_or_inactive")
    actor_id, actor_role = actor.get("sub"), actor.get("role")
    if not actor_id or actor_role not in {"user", "admin"}:
        raise MemoryEditError(403, "memory_edit_actor_invalid")
    if actor_role != "admin" and actor_id != record.userId:
        raise MemoryEditError(403, "Not your memory")
    if repair_item is not None and actor_role != "admin":
        raise MemoryEditError(403, "Admin access required")
    if level is not None and level not in {1, 2, 3}:
        raise MemoryEditError(400, "invalid_memory_level")
    if importance is not None and (not math.isfinite(importance) or not .1 <= importance <= .99):
        raise MemoryEditError(400, "invalid_memory_importance")
    expected_scope = await _scope(db, record)
    changed_content = content != record.content
    embedding = await generate_embedding(content) if changed_content else None
    operation = "repair_edit" if repair_item is not None else "user_edit"
    update_data = {"content": content} if changed_content else {}
    if level is not None and level != record.level:
        update_data["level"] = level
    if importance is not None and importance != record.importance:
        update_data["importance"] = importance
    origin = prepare_input_origin(kind="admin_edit" if actor_role == "admin" else "user_edit", payload={
        "memory_id": record.id, "memory_side": record.source,
        "actor_id": actor_id, "actor_role": actor_role,
        "previous_content": record.content, "previous_version": content_version(record.content),
        "previous_updated_at": record.updatedAt.isoformat(),
        "content": content, "repair_item_id": repair_item.get("id") if repair_item else None,
        "reason": reason,
    }) if changed_content else None
    resolved = None
    async with db.tx(timeout=timedelta(seconds=30)) as tx:
        await tx.execute_raw("SELECT pg_advisory_xact_lock($1::bigint)", scope_lock_key(record.source, record.userId, record.workspaceId))
        if await _scope(tx, record, lock=True) != expected_scope:
            raise MemoryEditError(409, "memory_scope_changed_or_inactive")
        await lock_actor(tx, actor_id, actor_role)
        if repair_item is not None:
            await lock_repair(tx, repair_item, record, expected_scope)
        await lock_memories(tx, [record])
        model = tx.aimemory if record.source == "ai" else tx.usermemory
        if update_data:
            await memory_repo.update(record.id, source=record.source, record=record,
                                     database=tx, **update_data)
        if changed_content:
            await store_embedding(record.id, embedding, database=tx)
            source = await persist_input_origin(tx, user_id=record.userId, workspace_id=record.workspaceId,
                agent_id=expected_scope["agent_id"], origin=origin)
            await bind_memory_evidence(database=tx, memory_id=record.id, side=record.source,
                user_id=record.userId, workspace_id=record.workspaceId, sources=[source],
                extractor_version="manual-edit-v1")
        if update_data or repair_item is not None:
            old_value = record.content
            new_value = content
            if repair_item is not None:
                old_value = json.dumps({"content": record.content, "level": record.level,
                                        "importance": record.importance}, ensure_ascii=False)
                new_value = json.dumps({"repair_item_id": repair_item["id"], "admin_id": actor_id,
                    "action": "edit_memory", "reason": reason,
                    "after": json.dumps({"content": content, **update_data}, ensure_ascii=False)}, ensure_ascii=False)
            await tx.memorychangelog.create(data={"userId": record.userId, "workspaceId": record.workspaceId,
                "memoryId": record.id, "operation": operation, "oldValue": old_value, "newValue": new_value})
        if repair_item is not None:
            resolved = await resolve_repair(tx, repair_item, actor_id, resolution_note, {
                "action": "edit_memory", "memory_id": record.id, "memory_source": record.source,
                "user_id": record.userId, "workspace_id": record.workspaceId,
            })
        updated = memory_repo._to_record(await model.find_unique(where={"id": record.id}), record.source)
    if update_data:
        await memory_repo.invalidate_scope(record.userId, record.workspaceId)
    if changed_content and record.source == "ai" and record.provenance == KNOWLEDGE_SEED:
        await invalidate_knowledge_rows(record.workspaceId)
    if update_data or repair_item is not None:
        await after_memory_changelog(user_id=record.userId, memory_id=record.id,
                                     operation=operation, workspace_id=record.workspaceId)
    return updated, resolved
