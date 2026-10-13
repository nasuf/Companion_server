"""Scope, optimistic versions and row locks shared by manual memory writes.

Callers prepare embeddings first, then lock scope/actor/repair/rows and commit all
business state together. These guards perform no network or postcommit effects.
"""
import json
from app.services.memory.storage import repo as memory_repo
from app.services.memory.repair_queue import serialize_repair_item

class MemoryWriteError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


async def memory_scope(database, record, *, lock: bool = False) -> dict:
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
        raise MemoryWriteError(409, "memory_scope_changed_or_inactive")
    return rows[0]


async def lock_actor(database, actor_id, actor_role):
    actors = await database.query_raw("SELECT role,status,archived_at FROM users WHERE id=$1 FOR SHARE", actor_id)
    if not actors or actors[0]["role"] != actor_role or actors[0]["status"] != "active" or actors[0]["archived_at"] is not None:
        raise MemoryWriteError(403, "memory_edit_actor_invalid")


async def lock_repair(database, repair_item, record, scope):
    items = await database.query_raw("SELECT * FROM memory_repair_items WHERE id=$1 FOR UPDATE", repair_item.get("id"))
    if not items:
        raise MemoryWriteError(404, "memory_repair_item_not_found")
    item = items[0]
    if item["status"] != "open":
        raise MemoryWriteError(409, "memory_repair_item_is_not_open")
    if serialize_repair_item(item) != repair_item:
        raise MemoryWriteError(409, "memory_repair_item_changed_refresh_and_retry")
    for key, expected in (("memory_id", record.id), ("memory_source", record.source),
                          ("user_id", record.userId), ("workspace_id", record.workspaceId),
                          ("agent_id", scope["agent_id"])):
        if item.get(key) is not None and item[key] != expected:
            raise MemoryWriteError(409, "memory_repair_scope_changed")


async def lock_memories(database, records, *, allow_archived=False):
    # Both tables share the vector key; reject ambiguous legacy IDs. Global
    # table/id order also prevents opposing merges locking their rows backwards.
    ids = sorted(record.id for record in records)
    if not ids:
        return
    user_rows = await database.query_raw("SELECT id FROM memories_user WHERE id=ANY($1::text[]) ORDER BY id FOR UPDATE", ids)
    ai_rows = await database.query_raw("SELECT id FROM memories_ai WHERE id=ANY($1::text[]) ORDER BY id FOR UPDATE", ids)
    if {r['id'] for r in user_rows} & {r['id'] for r in ai_rows}:
        raise MemoryWriteError(409, "memory_id_is_ambiguous")
    for record in records:
        model = database.aimemory if record.source == "ai" else database.usermemory
        row = await model.find_unique(where={"id": record.id})
        if not row or (row.isArchived and not allow_archived):
            raise MemoryWriteError(404, "Memory not found")
        current = memory_repo._to_record(row, record.source)
        # Compare the full persisted record, including taxonomy and provenance.
        if current != record:
            raise MemoryWriteError(409, "memory_changed_refresh_and_retry")


async def resolve_repair(database, repair_item, actor_id, note, result):
    rows = await database.query_raw("""UPDATE memory_repair_items
        SET status='resolved',resolution_note=$2,resolved_by_id=$3,
            resolved_at=CURRENT_TIMESTAMP,updated_at=CURRENT_TIMESTAMP,
            evidence=COALESCE(evidence,'{}'::jsonb)||jsonb_build_object('repair_result',$4::jsonb)
        WHERE id=$1 AND status='open' RETURNING *""",
        repair_item['id'],note,actor_id,json.dumps(result))
    if not rows:
        raise MemoryWriteError(409, "memory_repair_item_is_not_open")
    return serialize_repair_item(rows[0])
