from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any
from uuid import uuid4

from prisma.errors import DataError, RawQueryError

from app.db import db
from app.models.chat_links import DailyShareLink, DailyShareLinkGroup
from app.services.chat_links.cards import component_card_for_link
from app.services.chat_links.covers import cache_link_cover
from app.services.chat_links.extraction import LinkMetadata
from app.services.chat_media import storage as chat_media_storage


@dataclass(frozen=True)
class ChatLinkCard:
    id: str
    user_id: str
    conversation_id: str
    message_id: str | None
    role: str
    source_app: str | None
    source_url: str
    final_url: str
    platform: str
    title: str
    description: str
    author: str | None
    image_url: str | None
    content_text: str
    original_text: str
    summary: str
    status: str
    error: str | None
    metadata: dict[str, Any] | None
    created_at: Any = None
    updated_at: Any = None
    snapshot_id: str | None = None
    snapshot_recorded_at: Any = None


async def create_or_update_link_card(
    *,
    user_id: str,
    conversation_id: str,
    metadata: LinkMetadata,
    role: str = "user",
    source_app: str | None = None,
    message_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> ChatLinkCard:
    cover = await cache_link_cover(
        user_id=user_id,
        conversation_id=conversation_id,
        metadata=metadata,
    )
    metadata = cover.metadata
    merged_extra_metadata = {
        **cover.extra_metadata,
        **(extra_metadata or {}),
    }
    snapshot_id = str(uuid4())
    try:
        rows = await db.query_raw(
            """
            WITH cached AS (INSERT INTO chat_link_cards (
                user_id, conversation_id, message_id, role, source_app,
                source_url, final_url, platform, title, description, author,
                image_url, content_text, original_text, summary, status, error,
                metadata
            )
            SELECT $1, $2, $3, $4, $5,
                $6, $7, $8, $9, $10, $11,
                $12, $13, $14, $15, $16, $17, $18::jsonb
            FROM conversations c WHERE c.id=$2 AND c.user_id=$1 AND NOT c.is_deleted
              AND ($3::text IS NULL OR EXISTS (
                  SELECT 1 FROM messages m WHERE m.id=$3 AND m.conversation_id=c.id AND m.role=$4
              ))
            ON CONFLICT (user_id, conversation_id, final_url, role)
            DO UPDATE SET
                message_id = COALESCE(EXCLUDED.message_id, chat_link_cards.message_id),
                source_app = COALESCE(EXCLUDED.source_app, chat_link_cards.source_app),
                source_url = EXCLUDED.source_url,
                platform = EXCLUDED.platform,
                title = EXCLUDED.title,
                description = EXCLUDED.description,
                author = EXCLUDED.author,
                image_url = EXCLUDED.image_url,
                content_text = EXCLUDED.content_text,
                original_text = EXCLUDED.original_text,
                summary = EXCLUDED.summary,
                status = EXCLUDED.status,
                error = EXCLUDED.error,
                metadata = COALESCE(chat_link_cards.metadata, '{}'::jsonb)
                    || COALESCE(EXCLUDED.metadata, '{}'::jsonb),
                updated_at = NOW()
            RETURNING *), frozen AS (
                INSERT INTO chat_link_snapshots(id, link_card_id, payload)
                SELECT $19, id, to_jsonb(cached) - 'message_id' - 'created_at' - 'updated_at'
                FROM cached
                RETURNING id, recorded_at
            )
            SELECT cached.*, frozen.id AS snapshot_id,
                   frozen.recorded_at AT TIME ZONE 'UTC' AS snapshot_recorded_at FROM cached, frozen
            """,
            user_id,
            conversation_id,
            message_id,
            role,
            _clean_optional(source_app, 120),
            metadata.source_url[:2000],
            metadata.final_url[:2000],
            metadata.platform[:40],
            (metadata.title or "未命名链接")[:240],
            (metadata.description or "")[:1000],
            _clean_optional(metadata.author, 120),
            _clean_optional(metadata.image_url, 2000),
            (metadata.content_text or "")[:6000],
            (metadata.original_text or "")[:6000],
            (metadata.summary or "")[:1000],
            metadata.status[:40],
            _clean_optional(metadata.error, 300),
            json.dumps(merged_extra_metadata, ensure_ascii=False),
            snapshot_id,
        )
    except (DataError, RawQueryError):
        # Only an explicit SQL rejection proves rollback. A network timeout may
        # leave a COMMIT in flight: even a later absent SELECT cannot prove it
        # will not commit, so transport failures deliberately retain the cover.
        chat_media_storage.delete_media_file(
            cover.extra_metadata.get("cover_storage_key")
        )
        raise
    if not rows:
        chat_media_storage.delete_media_file(
            cover.extra_metadata.get("cover_storage_key")
        )
        raise ValueError("Link conversation or message is unavailable")
    return _link_from_row(rows[0])


async def find_link_card(
    *,
    link_id: str,
    user_id: str,
    conversation_id: str,
    require_unbound: bool = False,
    snapshot_id: str | None = None,
) -> ChatLinkCard | None:
    # A supplied version is authoritative: never silently replace a missing
    # or foreign snapshot with the mutable cache's latest content.
    rows = await db.query_raw(
        """
        SELECT l.*, s.id AS snapshot_id, s.recorded_at AT TIME ZONE 'UTC' AS snapshot_recorded_at,
               s.payload AS snapshot_payload
        FROM chat_link_cards l
        JOIN conversations c ON c.id=l.conversation_id AND c.user_id=l.user_id
        LEFT JOIN LATERAL (
            SELECT id, payload, recorded_at FROM chat_link_snapshots
            WHERE link_card_id=l.id AND (
                ($4::text IS NOT NULL AND id=$4)
                OR ($4::text IS NULL AND payload=
                    to_jsonb(l)-'message_id'-'created_at'-'updated_at')
            ) ORDER BY recorded_at DESC, id DESC LIMIT 1
        ) s ON TRUE
        WHERE l.id=$1 AND l.user_id=$2 AND l.conversation_id=$3
          AND c.is_deleted=FALSE AND (NOT $5::boolean OR l.message_id IS NULL)
          AND ($4::text IS NULL OR s.id IS NOT NULL)
        LIMIT 1
        """,
        link_id,
        user_id,
        conversation_id,
        snapshot_id,
        require_unbound,
    )
    if not rows:
        return None
    row = rows[0]
    frozen = _json_or_none(row.get("snapshot_payload"))
    return _link_from_row(
        {
            **row,
            **(frozen or {}),
            "snapshot_id": row.get("snapshot_id"),
            "snapshot_recorded_at": row.get("snapshot_recorded_at"),
        }
    )


async def bind_link_card_to_message(
    *,
    link_id: str,
    message_id: str,
    user_id: str,
    conversation_id: str,
) -> None:
    await db.execute_raw(
        """
        UPDATE chat_link_cards l
        SET message_id = $1, updated_at = NOW()
        WHERE l.id = $2 AND l.user_id = $3 AND l.conversation_id = $4
          AND (l.message_id IS NULL OR l.message_id = $1)
          AND EXISTS (SELECT 1 FROM messages m JOIN conversations c ON c.id=m.conversation_id
              WHERE m.id=$1 AND m.conversation_id=$4 AND m.role=l.role
                AND c.user_id=$3 AND NOT c.is_deleted)
        """,
        message_id,
        link_id,
        user_id,
        conversation_id,
    )


async def list_user_link_groups(
    user_id: str,
    *,
    limit: int | None = None,
) -> list[DailyShareLinkGroup]:
    # Message snapshots are historical truth. The URL cache is only a fallback
    # for old bound rows that never recorded a message projection.
    rows = await db.query_raw(
        """
        WITH history AS (
            SELECT m.id AS occurrence_id, m.conversation_id, m.role,
                   m.created_at AS timeline_at, m.metadata->'link_card' AS link_payload
            FROM conversations c JOIN messages m ON m.conversation_id=c.id
            WHERE c.user_id=$1 AND NOT c.is_deleted
              AND jsonb_typeof(m.metadata->'link_card')='object'
            UNION ALL
            SELECT * FROM (SELECT DISTINCT ON (m.id) m.id, m.conversation_id, m.role, m.created_at, to_jsonb(l)
            FROM chat_link_cards l JOIN conversations c ON c.id=l.conversation_id
            JOIN messages m ON m.id=l.message_id AND m.conversation_id=c.id AND m.role=l.role
            WHERE l.user_id=$1 AND c.user_id=$1 AND NOT c.is_deleted
              AND jsonb_typeof(m.metadata->'link_card') IS DISTINCT FROM 'object'
            ORDER BY m.id, l.updated_at DESC, l.id DESC) legacy
        )
        SELECT * FROM history ORDER BY timeline_at DESC, occurrence_id DESC
        LIMIT $2
        """,
        user_id,
        max(1, min(limit if limit is not None else 1000, 1000)),
    )
    links = []
    for row in rows:
        payload = _json_or_none(row.get("link_payload"))
        if not payload or not all(
            isinstance(payload.get(k), str) and payload[k]
            for k in ("id", "source_url", "final_url")
        ):
            continue
        # Existing link_card metadata embeds the actual component-card body.
        # Do not replace it with the current cache or another message's text.
        card = payload.get("component_card")
        item = _daily_link_from_row(
            {
                **payload,
                "conversation_id": row["conversation_id"],
                "role": row["role"],
                "message_id": row["occurrence_id"],
                "timeline_at": row["timeline_at"],
            }
        )
        if isinstance(card, dict) and card.get("type") == "external_link":
            item.component_card = card
        item.occurrence_id = row["occurrence_id"]
        links.append(item)
    return _group_links_by_day(links)


def _group_links_by_day(links: list[DailyShareLink]) -> list[DailyShareLinkGroup]:
    buckets: OrderedDict[str, list[DailyShareLink]] = OrderedDict()
    for link in links:
        key, title = _day_key_title(link.created_at)
        buckets.setdefault(f"{key}|{title}", []).append(link)
    groups: list[DailyShareLinkGroup] = []
    for bucket_key, group_links in buckets.items():
        key, title = bucket_key.split("|", 1)
        platforms = []
        for link in group_links:
            if link.platform not in platforms:
                platforms.append(link.platform)
        subtitle = "、".join(platforms[:4]) or "链接"
        groups.append(
            DailyShareLinkGroup(
                id=f"links-{key}",
                title=title,
                subtitle=subtitle,
                count=len(group_links),
                links=group_links,
            )
        )
    return groups


def _daily_link_from_row(row: Any) -> DailyShareLink:
    link = _link_from_row(row)
    timeline_at = _value(row, "timeline_at", "timelineAt") or link.created_at
    created = _format_dt(timeline_at)
    return DailyShareLink(
        id=link.id,
        message_id=link.message_id,
        conversation_id=link.conversation_id,
        role=link.role,
        source_app=link.source_app,
        source_url=link.source_url,
        final_url=link.final_url,
        platform=link.platform,
        title=link.title,
        description=link.description,
        author=link.author,
        image_url=link.image_url,
        summary=link.summary,
        created_at=created,
        component_card=component_card_for_link(link),
    )


def _link_from_row(row: Any) -> ChatLinkCard:
    return ChatLinkCard(
        id=str(_value(row, "id") or ""),
        user_id=str(_value(row, "user_id", "userId") or ""),
        conversation_id=str(_value(row, "conversation_id", "conversationId") or ""),
        message_id=_value(row, "message_id", "messageId"),
        role=str(_value(row, "role") or "user"),
        source_app=_value(row, "source_app", "sourceApp"),
        source_url=str(_value(row, "source_url", "sourceUrl") or ""),
        final_url=str(_value(row, "final_url", "finalUrl") or ""),
        platform=str(_value(row, "platform") or "链接"),
        title=str(_value(row, "title") or "未命名链接"),
        description=str(_value(row, "description") or ""),
        author=_value(row, "author"),
        image_url=_value(row, "image_url", "imageUrl"),
        content_text=str(_value(row, "content_text", "contentText") or ""),
        original_text=str(_value(row, "original_text", "originalText") or ""),
        summary=str(_value(row, "summary") or ""),
        status=str(_value(row, "status") or "ready"),
        error=_value(row, "error"),
        metadata=_json_or_none(_value(row, "metadata")),
        created_at=_value(row, "created_at", "createdAt"),
        updated_at=_value(row, "updated_at", "updatedAt"),
        snapshot_id=_value(row, "snapshot_id"),
        snapshot_recorded_at=_value(row, "snapshot_recorded_at"),
    )


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


def _json_or_none(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def _clean_optional(value: str | None, limit: int) -> str | None:
    cleaned = (value or "").strip()
    return cleaned[:limit] if cleaned else None


def _format_dt(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None:
        return None
    return str(value)


def _day_key_title(value: str | None) -> tuple[str, str]:
    parsed = None
    if value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            parsed = None
    if parsed is None:
        return "unknown", "未标记时间"
    return parsed.strftime("%Y-%m-%d"), parsed.strftime("%m月%d日")
