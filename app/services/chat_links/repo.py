from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any

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
    cover = await cache_link_cover(user_id=user_id, metadata=metadata)
    metadata = cover.metadata
    merged_extra_metadata = {
        **cover.extra_metadata,
        **(extra_metadata or {}),
    }
    try:
        rows = await db.query_raw(
            """
            INSERT INTO chat_link_cards (
                user_id, conversation_id, message_id, role, source_app,
                source_url, final_url, platform, title, description, author,
                image_url, content_text, original_text, summary, status, error,
                metadata
            )
            VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9, $10, $11,
                $12, $13, $14, $15, $16, $17,
                $18::jsonb
            )
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
            RETURNING *
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
        )
    except Exception:
        chat_media_storage.delete_media_file(cover.extra_metadata.get("cover_storage_key"))
        raise
    return _link_from_row(rows[0])


async def find_link_card(
    *,
    link_id: str,
    user_id: str,
    conversation_id: str,
    require_unbound: bool = False,
) -> ChatLinkCard | None:
    where = """
        id = $1
        AND user_id = $2
        AND conversation_id = $3
    """
    if require_unbound:
        where += "\n        AND message_id IS NULL"
    rows = await db.query_raw(f"SELECT * FROM chat_link_cards WHERE {where} LIMIT 1", link_id, user_id, conversation_id)
    if not rows:
        return None
    return _link_from_row(rows[0])


async def bind_link_card_to_message(
    *,
    link_id: str,
    message_id: str,
    user_id: str,
    conversation_id: str,
) -> None:
    await db.execute_raw(
        """
        UPDATE chat_link_cards
        SET message_id = $1, updated_at = NOW()
        WHERE id = $2
          AND user_id = $3
          AND conversation_id = $4
          AND (message_id IS NULL OR message_id = $1)
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
    query = """
    SELECT
      l.*,
      COALESCE(m.created_at, l.created_at) AS timeline_at
    FROM chat_link_cards l
    JOIN conversations c ON c.id = l.conversation_id
    LEFT JOIN messages m ON m.id = l.message_id
    WHERE l.user_id = $1
      AND c.user_id = $1
      AND c.is_deleted = FALSE
    ORDER BY COALESCE(m.created_at, l.created_at) DESC, l.created_at DESC
    """
    if limit is None:
        rows = await db.query_raw(query, user_id)
    else:
        rows = await db.query_raw(f"{query}\nLIMIT $2", user_id, max(1, min(limit, 1000)))
    links = [_daily_link_from_row(row) for row in rows or []]
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
