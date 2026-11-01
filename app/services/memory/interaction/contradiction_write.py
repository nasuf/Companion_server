"""A confirmed user-memory change commits with its vector, origins and audit.

One immutable input per pending ticket also acts as the durable replay receipt.
Neither model IDs nor Redis alone authorize a memory mutation.
"""

from datetime import datetime, timedelta, timezone
import json
import logging

from app.db import db
from app.services.memory.evidence import (
    EvidenceSource,
    bind_memory_evidence,
    content_version,
    _resolve_source,
    message_statement_time,
)
from app.services.memory.input_evidence import (
    prepare_input_origin,
    persist_input_origin,
)
from app.services.memory.lifecycle.lazy_update import scope_lock_key
from app.services.memory.manual_write import MemoryWriteError, lock_memories
from app.services.memory.storage import repo
from app.services.memory.storage.embedding import generate_embedding, store_embedding
from app.services.memory.storage.persistence import (
    after_memory_changelog,
    normalize_memory_type,
)
from app.services.memory.taxonomy import resolve_taxonomy, is_singleton
from .contradiction_state import conversation_scope, memory_version, validate_pending
from .change_semantics import resolve_change_semantics, select_change_statements
from .change_history import current_version_id, persist_change

logger = logging.getLogger(__name__)
EXTRACTOR = "contradiction-confirmation-v1"


def replacement_text(analysis):
    change = analysis.get("change_type")
    if change not in {"变化", "错误", "新增"}:
        raise MemoryWriteError(422, "contradiction_analysis_invalid")
    value = (
        analysis.get("new_memory")
        or (analysis.get("updated_memory") if change in {"变化", "错误"} else "")
        or ""
    )
    if not isinstance(value, str) or len(value) > 10000:
        raise MemoryWriteError(422, "contradiction_analysis_invalid")
    return change, value.strip()


async def apply_confirmed_change(
    conflict,
    analysis,
    *,
    conversation_id,
    user_id,
    workspace_id,
    agent_id,
    confirmation_sources,
):
    from .contradiction import _resolve_new_importance

    ticket = validate_pending(
        conflict,
        conversation_id=conversation_id,
        user_id=user_id,
        workspace_id=workspace_id,
        agent_id=agent_id,
    )
    change, content = replacement_text(analysis)
    if not content:
        return {"status": "no_change"}
    sources = tuple(
        EvidenceSource(**{**s, 'expected_statement_time':
            message_statement_time(s['expected_statement_time']) if s.get('expected_statement_time') else None})
        for s in [*ticket["sources"], *confirmation_sources]
    )
    if not confirmation_sources or len(sources) > 80:
        raise MemoryWriteError(409, "contradiction_confirmation_missing")
    if (
        await conversation_scope(
            db,
            conversation_id=conversation_id,
            user_id=user_id,
            workspace_id=workspace_id,
            agent_id=agent_id,
        )
        != ticket["scope"]
    ):
        raise MemoryWriteError(409, "contradiction_scope_changed")
    if await confirmation_receipt(db, workspace_id, ticket["id"]):
        return {"status": "already_applied"}
    old = await repo.find_unique(conflict.get("conflicting_memory_id"))
    if (
        not old
        or old.source != "user"
        or (old.userId, old.workspaceId) != (user_id, workspace_id)
    ):
        raise MemoryWriteError(409, "contradiction_memory_changed")
    original_sources = sources[:len(ticket["sources"])]
    message_rows = await db.query_raw(
        """SELECT id,content,created_at FROM messages WHERE id=ANY($1::text[])
        AND conversation_id=$2 AND role='user'
        AND NOT COALESCE(metadata,'{}'::jsonb) ? 'component_card'
        ORDER BY created_at,id""",
        [s.ref for s in sources], conversation_id,
    )
    pins = {s.ref: s for s in sources}
    if len(message_rows) != len(pins) or any(
        content_version(r["content"]) != pins[r["id"]].expected_version
        for r in message_rows
    ):
        raise MemoryWriteError(409, "contradiction_source_changed")
    if any(
        message_statement_time(r["created_at"]) != pins[r["id"]].expected_statement_time
        for r in message_rows
    ):
        # Preserve the evidence layer's rejection contract, even when detecting
        # a changed clock before embedding or transaction acquisition.
        raise ValueError("evidence_source_time_changed")
    original_ids = {s.ref for s in original_sources}
    original_rows = [r for r in message_rows if r["id"] in original_ids]
    confirmation_rows = [r for r in message_rows if r["id"] not in original_ids]
    fact_rows = select_change_statements(original_rows, confirmation_rows)
    semantics = resolve_change_semantics(change, fact_rows)
    if semantics.kind == "unknown":
        return {"status": "needs_clarification", "change_semantics": semantics.diagnostic()}
    old_statement = message_statement_time(old.statementTime) if old.statementTime else None
    if old_statement and semantics.statement_at < old_statement:
        raise MemoryWriteError(409, "contradiction_proposal_older_than_current_fact")
    planned = semantics.kind == "planned"
    if planned:
        # Keep the user's plan as a plan. A proposed future address must never
        # replace the currently true identity, even when the model says 变化.
        content = "用户的待实施计划（未证实已发生）：" + "；".join(r["content"] for r in fact_rows)
        if len(content) > 10000:
            raise MemoryWriteError(422, "contradiction_plan_over_limit")
    elif semantics.kind == "correction":
        change = "错误"
    importance = _resolve_new_importance(change, old.importance)
    if planned:
        importance = 0.6
    level = 1 if importance >= 0.85 else 2
    taxonomy = resolve_taxonomy(
        main_category="生活" if planned else analysis.get("new_memory_main_category") or old.mainCategory,
        sub_category="其他" if planned else analysis.get("new_memory_sub_category") or old.subCategory,
        legacy_type=normalize_memory_type(old.type),
        source="user",
        level=level,
        provenance="user_stated",
    )
    if (
        not taxonomy.allowed
        or taxonomy.sub_category == "提醒"
        or old.subCategory == "提醒"
    ):
        raise MemoryWriteError(422, "contradiction_taxonomy_not_allowed")
    embedding = await generate_embedding(content)
    origin = prepare_input_origin(
        kind="contradiction_confirmation",
        payload={
            "pending_id": ticket["id"],
            "change_type": change,
            "conversation_id": conversation_id,
            "old_memory_id": old.id,
            "old_memory_version": ticket["memory_version"],
            "proposed_content": content,
            "sources": [*ticket["sources"], *confirmation_sources],
            "analysis": analysis,
            "change_semantics": semantics.diagnostic(),
        },
    )
    events = []
    async with db.tx(timeout=timedelta(seconds=30)) as tx:
        await tx.execute_raw(
            "SELECT pg_advisory_xact_lock($1::bigint)",
            scope_lock_key("user", user_id, workspace_id),
        )
        scope = await conversation_scope(
            tx,
            conversation_id=conversation_id,
            user_id=user_id,
            workspace_id=workspace_id,
            agent_id=agent_id,
            lock=True,
        )
        if scope != ticket["scope"]:
            raise MemoryWriteError(409, "contradiction_scope_changed")
        # This indexed receipt survives deletion of its result: never resurrect it.
        receipt = await confirmation_receipt(tx, workspace_id, ticket["id"])
        if receipt:
            return {"status": "already_applied"}
        await lock_memories(tx, [old])
        if memory_version(old) != ticket["memory_version"]:
            raise MemoryWriteError(409, "contradiction_memory_changed")
        old_version_id = await current_version_id(tx, old)
        target_scope = {
            "user_id": user_id,
            "workspace_id": workspace_id,
            "agent_id": agent_id,
        }
        # Recheck all pinned messages before writes, including no takeover/deletion.
        for source in sources:
            await _resolve_source(tx, source, target_scope)
        plaintext = await tx.query_raw(
            """SELECT id FROM messages WHERE id=ANY($1::text[])
            AND NOT COALESCE(metadata,'{}'::jsonb) ? 'component_card' FOR SHARE""",
            list(pins),
        )
        if len(plaintext) != len(pins):
            raise MemoryWriteError(409, "contradiction_source_changed")
        if content == old.content:
            return {"status": "no_change"}
        if level == 1 and is_singleton(taxonomy.main_category, taxonomy.sub_category):
            taken = await tx.query_raw(
                """SELECT id FROM memories_user WHERE user_id=$1 AND workspace_id=$2
                AND level=1 AND NOT is_archived AND main_category=$3 AND sub_category=$4
                AND ($5::boolean OR id<>$6) LIMIT 1""",
                user_id,
                workspace_id,
                taxonomy.main_category,
                taxonomy.sub_category,
                change == "新增",
                old.id,
            )
            if taken:
                raise MemoryWriteError(409, "contradiction_singleton_conflict")
        if await tx.query_raw(
            """SELECT id FROM memories_user WHERE user_id=$1 AND workspace_id=$2
                AND NOT is_archived AND content=$3 AND id<>$4 LIMIT 1""",
            user_id,
            workspace_id,
            content,
            old.id,
        ):
            raise MemoryWriteError(409, "contradiction_duplicate_fact")
        now = datetime.now(timezone.utc)
        # The statement belongs to the messages that carry the fact (the
        # original proposal, or a substantive correction), never to a bare
        # "yes". A multi-message span is not one precise statement.
        times = [pins[r["id"]].expected_statement_time for r in fact_rows]
        statement = times[0] if times and all(t == times[0] for t in times) else None
        new = await repo.create(
            source="user",
            database=tx,
            userId=user_id,
            workspaceId=workspace_id,
            content=content,
            level=level,
            importance=importance,
            type=normalize_memory_type(taxonomy.legacy_type),
            mainCategory=taxonomy.main_category,
            subCategory=taxonomy.sub_category,
            provenance="user_stated",
            valueUpdatedAt=now,
            statementTime=statement,
            # Rewritten model text is not a validated event-time quote. Neither
            # confirmation time nor execution time proves when a change began.
            occurTime=None,
        )
        await store_embedding(new.id, embedding, database=tx)
        input_source = await persist_input_origin(
            tx,
            user_id=user_id,
            workspace_id=workspace_id,
            agent_id=agent_id,
            origin=origin,
        )
        await bind_memory_evidence(
            database=tx,
            memory_id=new.id,
            side="user",
            user_id=user_id,
            workspace_id=workspace_id,
            sources=(
                input_source,
                EvidenceSource(
                    "memory",
                    old.id,
                    side="user",
                    relation="derived_from",
                    expected_version=content_version(old.content),
                ),
                *sources,
            ),
            extractor_version=EXTRACTOR,
        )
        if change in {"变化", "错误"} and not planned:
            await repo.update(
                old.id, source="user", record=old, database=tx, isArchived=True
            )
            events.append((old.id, "contradiction_archived"))
            await tx.memorychangelog.create(
                data={
                    "userId": user_id,
                    "workspaceId": workspace_id,
                    "memoryId": old.id,
                    "operation": "contradiction_archived",
                    "oldValue": old.content,
                    "newValue": json.dumps(
                        {"superseded_by": new.id, "pending_id": ticket["id"]}
                    ),
                }
            )
        await persist_change(tx, old=old, new=new, origin_id=input_source.ref,
            old_version_id=old_version_id,
            semantics=semantics)
        await tx.memorychangelog.create(
            data={
                "userId": user_id,
                "workspaceId": workspace_id,
                "memoryId": new.id,
                "operation": "contradiction_new",
                "newValue": json.dumps(
                    {
                        "change_type": change,
                        "replaces": old.id,
                        "input_origin": input_source.ref,
                    },
                    ensure_ascii=False,
                ),
            }
        )
        events.append((new.id, "contradiction_new"))
    await repo.invalidate_scope(user_id, workspace_id)
    for mid, operation in events:
        try:
            await after_memory_changelog(
                user_id=user_id,
                memory_id=mid,
                operation=operation,
                workspace_id=workspace_id,
            )
        except Exception:
            logger.warning("Contradiction postcommit hook failed", exc_info=True)
    return {"status": "plan_recorded" if planned else "applied", "memory_id": new.id,
            "change_type": change, "change_semantics": semantics.diagnostic()}


async def confirmation_receipt(database, workspace_id, pending_id):
    return await database.query_raw(
        """SELECT id FROM memory_input_origins WHERE workspace_id=$1
        AND source_status='contradiction_confirmation'
        AND payload_text::jsonb->'input'->>'pending_id'=$2""",
        workspace_id,
        pending_id,
    )
