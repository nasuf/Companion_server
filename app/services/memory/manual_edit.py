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
from app.services.memory.repair_queue import serialize_repair_item
from app.services.memory.provenance import KNOWLEDGE_SEED
from app.services.memory.retrieval.knowledge_hits import invalidate_knowledge_rows
from app.services.memory.storage import repo as memory_repo
from app.services.memory.storage.embedding import generate_embedding, store_embedding
from app.services.memory.storage.persistence import after_memory_changelog


class MemoryEditError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


async def _scope(database, record, *, lock: bool = False) -> dict:
    rows = await database.query_raw(
        """SELECT w.agent_id, w.execution_generation AS workspace_generation,
                  a.execution_generation AS agent_generation, u.execution_generation AS user_generation
           FROM chat_workspaces w JOIN ai_agents a ON a.id=w.agent_id AND a.user_id=w.user_id
             JOIN users u ON u.id=w.user_id
           WHERE w.id=$1 AND w.user_id=$2 AND w.status='active' AND a.status='active'
             AND u.status='active' AND w.archived_at IS NULL AND a.archived_at IS NULL
             AND u.archived_at IS NULL""" + (" FOR SHARE OF w,a,u" if lock else ""),
        record.workspaceId, record.userId,
    )
    if not rows:
        raise MemoryEditError(409, "memory_scope_changed_or_inactive")
    return rows[0]


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
        if await _scope(tx, record, lock=True) != expected_scope:
            raise MemoryEditError(409, "memory_scope_changed_or_inactive")
        actors = await tx.query_raw("SELECT role,status,archived_at FROM users WHERE id=$1 FOR SHARE", actor_id)
        if not actors or actors[0]["role"] != actor_role or actors[0]["status"] != "active" or actors[0]["archived_at"] is not None:
            raise MemoryEditError(403, "memory_edit_actor_invalid")
        if repair_item is not None:
            items = await tx.query_raw("SELECT * FROM memory_repair_items WHERE id=$1 FOR UPDATE", repair_item.get("id"))
            if not items:
                raise MemoryEditError(404, "memory_repair_item_not_found")
            item = items[0]
            if item["status"] != "open":
                raise MemoryEditError(409, "memory_repair_item_is_not_open")
            if serialize_repair_item(item) != repair_item:
                raise MemoryEditError(409, "memory_repair_item_changed_refresh_and_retry")
            # A queue entry with no scope is permitted only when every populated
            # field agrees. Its target cannot change while embedding runs.
            for key, expected in (("memory_id", record.id), ("memory_source", record.source),
                                  ("user_id", record.userId), ("workspace_id", record.workspaceId),
                                  ("agent_id", expected_scope["agent_id"])):
                if item.get(key) != repair_item.get(key) or (item.get(key) is not None and item[key] != expected):
                    raise MemoryEditError(409, "memory_repair_scope_changed")
        # Both tables share the vector key. Refuse ambiguous historical IDs
        # rather than overwrite another side's vector or evidence.
        user_rows = await tx.query_raw("SELECT id FROM memories_user WHERE id=$1 FOR UPDATE", record.id)
        ai_rows = await tx.query_raw("SELECT id FROM memories_ai WHERE id=$1 FOR UPDATE", record.id)
        if user_rows and ai_rows:
            raise MemoryEditError(409, "memory_id_is_ambiguous")
        model = tx.aimemory if record.source == "ai" else tx.usermemory
        current = await model.find_unique(where={"id": record.id})
        if not current or current.isArchived:
            raise MemoryEditError(404, "Memory not found")
        if (current.userId != record.userId or current.workspaceId != record.workspaceId
                or current.content != record.content or current.updatedAt != record.updatedAt
                or current.level != record.level or current.importance != record.importance):
            raise MemoryEditError(409, "memory_changed_refresh_and_retry")
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
            rows = await tx.query_raw("""UPDATE memory_repair_items SET status='resolved',resolution_note=$2,
                resolved_by_id=$3,resolved_at=CURRENT_TIMESTAMP,updated_at=CURRENT_TIMESTAMP
                WHERE id=$1 AND status='open' RETURNING *""", repair_item["id"],resolution_note,actor_id)
            if not rows:
                raise MemoryEditError(409, "memory_repair_item_is_not_open")
            resolved = serialize_repair_item(rows[0])
        updated = memory_repo._to_record(await model.find_unique(where={"id": record.id}), record.source)
    if update_data:
        await memory_repo.invalidate_scope(record.userId, record.workspaceId)
    if changed_content and record.source == "ai" and record.provenance == KNOWLEDGE_SEED:
        await invalidate_knowledge_rows(record.workspaceId)
    if update_data or repair_item is not None:
        await after_memory_changelog(user_id=record.userId, memory_id=record.id,
                                     operation=operation, workspace_id=record.workspaceId)
    return updated, resolved
