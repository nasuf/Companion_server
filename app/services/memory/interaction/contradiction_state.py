"""Server-owned, versioned confirmation inputs; Redis never supplies identity."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from types import SimpleNamespace
from uuid import uuid4

from app.db import db
from app.services.memory.evidence import EvidenceSource, content_version, message_statement_time
from app.services.memory.manual_write import MemoryWriteError, memory_scope
from app.services.memory.storage import repo

FORMAT = "memory-confirmation-v1"
MAX_MESSAGES = 40  # two turns + input/parent must fit the 100-link evidence bound


def memory_version(record):
    """Pin semantic fields, without expiring on an ordinary usage-score update."""
    fields = (
        "id",
        "source",
        "userId",
        "workspaceId",
        "content",
        "type",
        "mainCategory",
        "subCategory",
        "provenance",
        "occurTime",
    )
    value = {key: getattr(record, key) for key in fields}
    return content_version(
        json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    )


async def conversation_scope(
    database, *, conversation_id, user_id, workspace_id, agent_id, lock=False
):
    scope = await memory_scope(
        database, SimpleNamespace(userId=user_id, workspaceId=workspace_id), lock=lock
    )
    rows = await database.query_raw(
        """SELECT id FROM conversations WHERE id=$1 AND user_id=$2
        AND workspace_id=$3 AND agent_id=$4 AND NOT is_deleted AND archived_at IS NULL"""
        + (" FOR SHARE" if lock else ""),
        conversation_id,
        user_id,
        workspace_id,
        agent_id,
    )
    if not rows or scope["agent_id"] != agent_id:
        raise MemoryWriteError(409, "contradiction_scope_changed")
    if lock:
        # Existing pending/active takeover rows serialize with administrator claim.
        await database.query_raw(
            "SELECT id FROM human_takeovers WHERE user_id=$1 ORDER BY id FOR SHARE",
            user_id,
        )
    if await database.query_raw(
        "SELECT id FROM human_takeovers WHERE user_id=$1 AND status='active' LIMIT 1",
        user_id,
    ):
        raise MemoryWriteError(409, "contradiction_human_takeover")
    return scope


async def message_sources(database, *, conversation_id, message_ids, text, lock=False):
    ids = list(dict.fromkeys(message_ids))
    if (
        not ids
        or len(ids) > MAX_MESSAGES
        or any(not isinstance(i, str) or not i or len(i) > 200 for i in ids)
    ):
        raise MemoryWriteError(409, "contradiction_message_ids_missing")
    rows = await database.query_raw(
        """SELECT id,content,created_at FROM messages
        WHERE id=ANY($1::text[]) AND conversation_id=$2 AND role='user'
        AND COALESCE(metadata->>'human_takeover','false')<>'true'
        ORDER BY created_at,id"""
        # The caller reserves these rows with UPDATE in the same transaction.
        # Lock exclusively now to avoid concurrent shared-lock upgrade deadlocks.
        + (" FOR UPDATE" if lock else ""),
        ids,
        conversation_id,
    )
    if len(rows) != len(ids) or any(
        not r["content"].strip() or r["content"].strip() not in text for r in rows
    ):
        raise MemoryWriteError(409, "contradiction_message_scope_or_content_changed")
    return [
        asdict(
            EvidenceSource(
                "message",
                r["id"],
                relation="recorded_from",
                expected_role="user",
                expected_version=content_version(r["content"]),
                expected_statement_time=message_statement_time(r["created_at"]).isoformat(),
                expected_conversation_id=conversation_id,
            )
        )
        for r in rows
    ]


async def exclude_messages(database, conversation_id, ids):
    # Preserve other metadata. Background extraction rechecks this at bind time.
    await database.execute_raw(
        """UPDATE messages SET metadata=COALESCE(metadata,'{}'::jsonb)
        || '{"memory_confirmation":true}'::jsonb WHERE id=ANY($1::text[]) AND conversation_id=$2""",
        ids,
        conversation_id,
    )


async def prepare_pending(
    conflict,
    *,
    conversation_id,
    user_id,
    workspace_id,
    agent_id,
    message_ids,
    user_message,
):
    scope = await conversation_scope(
        db,
        conversation_id=conversation_id,
        user_id=user_id,
        workspace_id=workspace_id,
        agent_id=agent_id,
    )
    old = await repo.find_unique(conflict.get("conflicting_memory_id"))
    if (
        not old
        or old.source != "user"
        or old.isArchived
        or (old.userId, old.workspaceId) != (user_id, workspace_id)
        or old.content != conflict.get("old_content")
    ):
        raise MemoryWriteError(409, "contradiction_memory_changed")
    sources = await message_sources(
        db, conversation_id=conversation_id, message_ids=message_ids, text=user_message
    )
    return {
        **{
            key: conflict.get(key)
            for key in (
                "has_conflict",
                "conflicting_memory_id",
                "old_content",
                "new_info",
                "conflict_description",
                "source",
            )
        },
        "_confirmation": {
            "format": FORMAT,
            "id": uuid4().hex,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "workspace_id": workspace_id,
            "agent_id": agent_id,
            "scope": scope,
            "memory_version": memory_version(old),
            "sources": sources,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def validate_pending(pending, *, conversation_id, user_id, workspace_id, agent_id):
    ticket = pending.get("_confirmation")
    if not isinstance(ticket, dict) or ticket.get("format") != FORMAT:
        raise MemoryWriteError(409, "contradiction_pending_expired")
    if any(
        ticket.get(k) != v
        for k, v in [
            ("conversation_id", conversation_id),
            ("user_id", user_id),
            ("workspace_id", workspace_id),
            ("agent_id", agent_id),
        ]
    ):
        raise MemoryWriteError(409, "contradiction_scope_changed")
    try:
        age = (
            datetime.now(timezone.utc) - datetime.fromisoformat(ticket["created_at"])
        ).total_seconds()
        if (
            not 0 <= age <= 1800
            or len(ticket["id"]) != 32
            or not all(c in "0123456789abcdef" for c in ticket["id"])
        ):
            raise ValueError
        sources = tuple(EvidenceSource(**s) for s in ticket["sources"])
        if not 1 <= len(sources) <= MAX_MESSAGES or any(
            s.kind != "message"
            or s.relation != "recorded_from"
            or s.expected_role != "user"
            or not s.expected_version
            for s in sources
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise MemoryWriteError(409, "contradiction_pending_expired") from exc
    return ticket


async def prepare_confirmation(
    pending,
    *,
    conversation_id,
    user_id,
    workspace_id,
    agent_id,
    message_ids,
    user_message,
):
    ticket = validate_pending(
        pending,
        conversation_id=conversation_id,
        user_id=user_id,
        workspace_id=workspace_id,
        agent_id=agent_id,
    )
    async with db.tx() as tx:
        if (
            await conversation_scope(
                tx,
                conversation_id=conversation_id,
                user_id=user_id,
                workspace_id=workspace_id,
                agent_id=agent_id,
                lock=True,
            )
            != ticket["scope"]
        ):
            raise MemoryWriteError(409, "contradiction_scope_changed")
        sources = await message_sources(
            tx,
            conversation_id=conversation_id,
            message_ids=message_ids,
            text=user_message,
            lock=True,
        )
        if {s["ref"] for s in sources} & {s["ref"] for s in ticket["sources"]}:
            raise MemoryWriteError(409, "contradiction_confirmation_must_be_new_turn")
        await exclude_messages(tx, conversation_id, message_ids)
    return sources
