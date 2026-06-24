from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.db import db


@dataclass(frozen=True)
class OfflineActivityMedia:
    id: str
    recommendation_id: str
    user_id: str
    kind: str
    name: str | None
    mime: str
    size: int
    width: int | None
    height: int | None
    duration_seconds: int | None
    storage_key: str
    url: str
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


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def media_from_row(row: Any) -> OfflineActivityMedia:
    return OfflineActivityMedia(
        id=str(_value(row, "id")),
        recommendation_id=str(
            _value(row, "recommendation_id", "recommendationId"),
        ),
        user_id=str(_value(row, "user_id", "userId")),
        kind=str(_value(row, "kind") or "image"),
        name=_value(row, "name"),
        mime=str(_value(row, "mime") or "image/jpeg"),
        size=int(_value(row, "size") or 0),
        width=_int_or_none(_value(row, "width")),
        height=_int_or_none(_value(row, "height")),
        duration_seconds=_int_or_none(_value(row, "duration_seconds", "durationSeconds")),
        storage_key=str(_value(row, "storage_key", "storageKey") or ""),
        url=str(_value(row, "url") or ""),
        created_at=_value(row, "created_at", "createdAt"),
    )


async def activity_belongs_to_user(activity_id: str, user_id: str) -> bool:
    rows = await db.query_raw(
        """
        SELECT id
        FROM offline_activity_recommendations
        WHERE id = $1 AND user_id = $2
        LIMIT 1
        """,
        activity_id,
        user_id,
    )
    return bool(rows)


async def create_media(
    *,
    recommendation_id: str,
    user_id: str,
    storage_key: str,
    url: str,
    mime: str,
    size: int,
    kind: str = "image",
    name: str | None = None,
    width: int | None = None,
    height: int | None = None,
    duration_seconds: int | None = None,
) -> OfflineActivityMedia:
    rows = await db.query_raw(
        """
        INSERT INTO offline_activity_media (
            recommendation_id, user_id, kind, name, mime, size,
            width, height, duration_seconds, storage_key, url
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        RETURNING *
        """,
        recommendation_id,
        user_id,
        kind,
        name,
        mime,
        size,
        width,
        height,
        duration_seconds,
        storage_key,
        url,
    )
    return media_from_row(rows[0])


async def get_activity_media(
    *,
    media_ids: list[str],
    user_id: str,
    recommendation_id: str,
) -> list[OfflineActivityMedia]:
    if not media_ids:
        return []
    rows = await db.query_raw(
        """
        SELECT *
        FROM offline_activity_media
        WHERE id = ANY($1::text[])
          AND user_id = $2
          AND recommendation_id = $3
        ORDER BY created_at ASC
        """,
        media_ids,
        user_id,
        recommendation_id,
    )
    found = [media_from_row(row) for row in rows or []]
    by_id = {item.id: item for item in found}
    return [by_id[item_id] for item_id in media_ids if item_id in by_id]


async def delete_user_activity_media(user_id: str) -> list[OfflineActivityMedia]:
    rows = await db.query_raw(
        """
        DELETE FROM offline_activity_media
        WHERE user_id = $1
        RETURNING *
        """,
        user_id,
    )
    return [media_from_row(row) for row in rows or []]
