from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from app.db import db


@dataclass(frozen=True)
class ChatAttachment:
    id: str
    user_id: str
    conversation_id: str
    message_id: str | None
    kind: str
    name: str | None
    mime: str
    size: int
    width: int | None
    height: int | None
    storage_key: str
    url: str
    vision_status: str
    vision_summary: str | None
    vision_error: str | None
    created_at: Any = None


def _value(row: Any, snake: str, camel: str | None = None) -> Any:
    if isinstance(row, dict):
        if snake in row:
            return row[snake]
        if camel and camel in row:
            return row[camel]
        return None
    if hasattr(row, snake):
        return getattr(row, snake)
    if camel and hasattr(row, camel):
        return getattr(row, camel)
    return None


def _attachment_from_row(row: Any) -> ChatAttachment:
    return ChatAttachment(
        id=str(_value(row, "id")),
        user_id=str(_value(row, "user_id", "userId")),
        conversation_id=str(_value(row, "conversation_id", "conversationId")),
        message_id=_value(row, "message_id", "messageId"),
        kind=str(_value(row, "kind") or "image"),
        name=_value(row, "name"),
        mime=str(_value(row, "mime") or "image/jpeg"),
        size=int(_value(row, "size") or 0),
        width=_int_or_none(_value(row, "width")),
        height=_int_or_none(_value(row, "height")),
        storage_key=str(_value(row, "storage_key", "storageKey") or ""),
        url=str(_value(row, "url") or ""),
        vision_status=str(_value(row, "vision_status", "visionStatus") or "pending"),
        vision_summary=_value(row, "vision_summary", "visionSummary"),
        vision_error=_value(row, "vision_error", "visionError"),
        created_at=_value(row, "created_at", "createdAt"),
    )


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


async def conversation_belongs_to_user(conversation_id: str, user_id: str) -> bool:
    row = await db.conversation.find_unique(where={"id": conversation_id})
    return bool(row and row.userId == user_id and not row.isDeleted)


async def create_attachment(
    *,
    user_id: str,
    conversation_id: str,
    storage_key: str,
    url: str,
    mime: str,
    size: int,
    name: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> ChatAttachment:
    rows = await db.query_raw(
        """
        INSERT INTO chat_message_attachments (
            user_id, conversation_id, kind, name, mime, size,
            width, height, storage_key, url, vision_status
        )
        VALUES ($1, $2, 'image', $3, $4, $5, $6, $7, $8, $9, 'pending')
        RETURNING *
        """,
        user_id,
        conversation_id,
        name,
        mime,
        size,
        width,
        height,
        storage_key,
        url,
    )
    return _attachment_from_row(rows[0])


async def find_attachments_for_message(message_id: str) -> list[ChatAttachment]:
    rows = await db.query_raw(
        """
        SELECT *
        FROM chat_message_attachments
        WHERE message_id = $1
        ORDER BY created_at ASC
        """,
        message_id,
    )
    return [_attachment_from_row(row) for row in rows or []]


async def get_message_attachments(
    *,
    attachment_ids: list[str],
    user_id: str,
    conversation_id: str,
) -> list[ChatAttachment]:
    if not attachment_ids:
        return []
    rows = await db.query_raw(
        """
        SELECT *
        FROM chat_message_attachments
        WHERE id = ANY($1::text[])
          AND user_id = $2
          AND conversation_id = $3
          AND message_id IS NULL
        ORDER BY created_at ASC
        """,
        attachment_ids,
        user_id,
        conversation_id,
    )
    found = [_attachment_from_row(row) for row in rows or []]
    found_by_id = {item.id: item for item in found}
    return [found_by_id[item_id] for item_id in attachment_ids if item_id in found_by_id]


async def bind_attachments_to_message(
    *,
    attachment_ids: list[str],
    message_id: str,
    user_id: str,
    conversation_id: str,
) -> None:
    if not attachment_ids:
        return
    await db.execute_raw(
        """
        UPDATE chat_message_attachments
        SET message_id = $1, updated_at = NOW()
        WHERE id = ANY($2::text[])
          AND user_id = $3
          AND conversation_id = $4
          AND message_id IS NULL
        """,
        message_id,
        attachment_ids,
        user_id,
        conversation_id,
    )


async def update_vision_result(
    attachment_id: str,
    *,
    status: str,
    summary: str | None = None,
    error: str | None = None,
) -> None:
    await db.execute_raw(
        """
        UPDATE chat_message_attachments
        SET vision_status = $2,
            vision_summary = $3,
            vision_error = $4,
            updated_at = NOW()
        WHERE id = $1
        """,
        attachment_id,
        status,
        summary,
        error,
    )


async def cleanup_unbound_attachments(user_id: str, *, max_age_seconds: int = 86400) -> list[ChatAttachment]:
    cutoff = datetime.now(UTC) - timedelta(seconds=max_age_seconds)
    rows = await db.query_raw(
        """
        DELETE FROM chat_message_attachments
        WHERE user_id = $1
          AND message_id IS NULL
          AND created_at < $2::timestamptz
        RETURNING *
        """,
        user_id,
        cutoff,
    )
    return [_attachment_from_row(row) for row in rows or []]
