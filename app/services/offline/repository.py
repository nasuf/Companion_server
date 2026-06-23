from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from app.db import db
from app.services import profile_tags
from app.services.offline.user_tags import derive_user_tags

logger = logging.getLogger(__name__)


def new_id() -> str:
    return uuid4().hex


def now_utc() -> datetime:
    return datetime.now(UTC)


def _field(row: Any, snake: str, camel: str | None = None, default: Any = None) -> Any:
    if isinstance(row, dict):
        if snake in row:
            return row[snake]
        if camel and camel in row:
            return row[camel]
        return default
    if hasattr(row, snake):
        return getattr(row, snake)
    if camel and hasattr(row, camel):
        return getattr(row, camel)
    return default


def _json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    data = getattr(value, "data", None)
    if isinstance(data, (dict, list)):
        return data
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, type(default)) else default
        except Exception:
            return default
    return default


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    text = str(value)
    return text or None


def _timestamp_or_none(value: Any) -> datetime | str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        datetime.fromisoformat(normalized)
        return normalized
    except ValueError:
        logger.warning("Dropping invalid offline activity timestamp: %r", text[:120])
        return None


def activity_from_row(row: Any, *, reveal_task: bool = False) -> dict[str, Any]:
    status = str(_field(row, "status") or "pending")
    return {
        "id": str(_field(row, "id")),
        "user_id": _field(row, "user_id", "userId"),
        "agent_id": _field(row, "agent_id", "agentId"),
        "workspace_id": _field(row, "workspace_id", "workspaceId"),
        "conversation_id": _field(row, "conversation_id", "conversationId"),
        "status": status,
        "title": str(_field(row, "title") or ""),
        "summary": str(_field(row, "summary") or ""),
        "description": str(_field(row, "description") or ""),
        "category": _field(row, "category"),
        "city": _field(row, "city"),
        "location_name": _field(row, "location_name", "locationName"),
        "address": _field(row, "address"),
        "starts_at": _iso(_field(row, "starts_at", "startsAt")),
        "ends_at": _iso(_field(row, "ends_at", "endsAt")),
        "official_url": _field(row, "official_url", "officialUrl"),
        "image_urls": list(_json(_field(row, "image_urls", "imageUrls"), [])),
        "task_hint": _field(row, "task_hint", "taskHint"),
        "easter_egg_task": (
            _json(_field(row, "easter_egg_task", "easterEggTask"), None)
            if reveal_task or status in {"accepted", "completed"}
            else None
        ),
        "search_sources": list(_json(_field(row, "search_sources", "searchSources"), [])),
        "accepted_at": _iso(_field(row, "accepted_at", "acceptedAt")),
        "ignored_at": _iso(_field(row, "ignored_at", "ignoredAt")),
        "completed_at": _iso(_field(row, "completed_at", "completedAt")),
        "expires_at": _iso(_field(row, "expires_at", "expiresAt")),
        "completion_feedback": None,
        "created_at": _iso(_field(row, "created_at", "createdAt")) or "",
        "updated_at": _iso(_field(row, "updated_at", "updatedAt")) or "",
    }


def gift_from_row(row: Any) -> dict[str, Any]:
    return {
        "id": str(_field(row, "id")),
        "user_id": _field(row, "user_id", "userId"),
        "agent_id": _field(row, "agent_id", "agentId"),
        "workspace_id": _field(row, "workspace_id", "workspaceId"),
        "conversation_id": _field(row, "conversation_id", "conversationId"),
        "status": str(_field(row, "status") or "pending_address"),
        "trigger_type": str(_field(row, "trigger_type", "triggerType") or "daily_probability"),
        "gift_name": str(_field(row, "gift_name", "giftName") or ""),
        "gift_reason": _field(row, "gift_reason", "giftReason"),
        "gift_note": _field(row, "gift_note", "giftNote"),
        "product_image_url": _field(row, "product_image_url", "productImageUrl"),
        "paid_amount_cents": int(_field(row, "paid_amount_cents", "paidAmountCents", 0) or 0),
        "tracking_number": _field(row, "tracking_number", "trackingNumber"),
        "thanks_sent_at": _iso(_field(row, "thanks_sent_at", "thanksSentAt")),
        "ordered_at": _iso(_field(row, "ordered_at", "orderedAt")),
        "shipped_at": _iso(_field(row, "shipped_at", "shippedAt")),
        "delivered_at": _iso(_field(row, "delivered_at", "deliveredAt")),
        "created_at": _iso(_field(row, "created_at", "createdAt")) or "",
        "updated_at": _iso(_field(row, "updated_at", "updatedAt")) or "",
    }


def address_from_row(row: Any | None, *, masked: bool = True) -> dict[str, Any] | None:
    if not row:
        return None
    phone = str(_field(row, "phone") or "")
    detail = str(_field(row, "detail") or "")
    masked_phone = phone if not masked or len(phone) < 7 else f"{phone[:3]}****{phone[-4:]}"
    masked_detail = detail if not masked or len(detail) <= 6 else f"{detail[:6]}***"
    province = str(_field(row, "province") or "")
    city = str(_field(row, "city") or "")
    district = str(_field(row, "district") or "")
    display = " ".join(part for part in [province, city, district, masked_detail] if part)
    return {
        "id": str(_field(row, "id")),
        "recipient_name": _field(row, "recipient_name", "recipientName"),
        "phone": masked_phone,
        "province": province,
        "city": city,
        "district": district,
        "detail": masked_detail,
        "display": display or None,
    }


async def resolve_user_context(user_id: str, workspace_id: str | None = None) -> dict[str, Any] | None:
    if workspace_id:
        rows = await db.query_raw(
            """
            SELECT w.id AS workspace_id, w.user_id, w.agent_id, a.name AS agent_name,
                   a.city AS agent_city, c.id AS conversation_id, u.created_at AS user_created_at,
                   u.location_latitude AS user_location_latitude,
                   u.location_longitude AS user_location_longitude,
                   u.location_city AS user_location_city,
                   u.location_region AS user_location_region,
                   u.location_country AS user_location_country,
                   u.location_permission_status AS user_location_permission_status,
                   u.location_updated_at AS user_location_updated_at
            FROM chat_workspaces w
            JOIN ai_agents a ON a.id = w.agent_id
            JOIN users u ON u.id = w.user_id
            LEFT JOIN LATERAL (
                SELECT id FROM conversations
                WHERE workspace_id = w.id AND is_deleted = FALSE
                ORDER BY updated_at DESC LIMIT 1
            ) c ON TRUE
            WHERE w.id = $1 AND w.user_id = $2 AND w.status = 'active'
            LIMIT 1
            """,
            workspace_id,
            user_id,
        )
    else:
        rows = await db.query_raw(
            """
            SELECT w.id AS workspace_id, w.user_id, w.agent_id, a.name AS agent_name,
                   a.city AS agent_city, c.id AS conversation_id, u.created_at AS user_created_at,
                   u.location_latitude AS user_location_latitude,
                   u.location_longitude AS user_location_longitude,
                   u.location_city AS user_location_city,
                   u.location_region AS user_location_region,
                   u.location_country AS user_location_country,
                   u.location_permission_status AS user_location_permission_status,
                   u.location_updated_at AS user_location_updated_at
            FROM chat_workspaces w
            JOIN ai_agents a ON a.id = w.agent_id
            JOIN users u ON u.id = w.user_id
            LEFT JOIN LATERAL (
                SELECT id FROM conversations
                WHERE workspace_id = w.id AND is_deleted = FALSE
                ORDER BY updated_at DESC LIMIT 1
            ) c ON TRUE
            WHERE w.user_id = $1 AND w.status = 'active'
            ORDER BY w.created_at DESC
            LIMIT 1
            """,
            user_id,
        )
    if not rows:
        return None
    row = rows[0]
    latitude = _field(row, "user_location_latitude", "userLocationLatitude")
    longitude = _field(row, "user_location_longitude", "userLocationLongitude")
    city = _field(row, "user_location_city", "userLocationCity")
    region = _field(row, "user_location_region", "userLocationRegion")
    permission_status = _field(
        row, "user_location_permission_status", "userLocationPermissionStatus"
    )
    return {
        "workspace_id": _field(row, "workspace_id", "workspaceId"),
        "user_id": _field(row, "user_id", "userId"),
        "agent_id": _field(row, "agent_id", "agentId"),
        "agent_name": _field(row, "agent_name", "agentName") or "伴生",
        "agent_city": _field(row, "agent_city", "agentCity"),
        "conversation_id": _field(row, "conversation_id", "conversationId"),
        "user_created_at": _field(row, "user_created_at", "userCreatedAt"),
        "user_location_latitude": latitude,
        "user_location_longitude": longitude,
        "user_location_city": city,
        "user_location_region": region,
        "user_location_country": _field(row, "user_location_country", "userLocationCountry"),
        "user_location_permission_status": permission_status,
        "user_location_updated_at": _field(
            row, "user_location_updated_at", "userLocationUpdatedAt"
        ),
        "has_location": latitude is not None
        and longitude is not None
        and permission_status in {"whileInUse", "always"},
    }


async def ensure_trigger_state(user_id: str, agent_id: str, workspace_id: str | None) -> dict[str, Any]:
    rows = await db.query_raw(
        """
        INSERT INTO real_world_trigger_states (id, user_id, agent_id, workspace_id)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (user_id, agent_id) DO UPDATE
        SET workspace_id = COALESCE(EXCLUDED.workspace_id, real_world_trigger_states.workspace_id),
            updated_at = CURRENT_TIMESTAMP
        RETURNING *
        """,
        new_id(),
        user_id,
        agent_id,
        workspace_id,
    )
    return dict(rows[0])


async def list_real_world_contexts(*, limit: int = 500) -> list[dict[str, Any]]:
    rows = await db.query_raw(
        """
        SELECT w.id AS workspace_id, w.user_id, w.agent_id, a.name AS agent_name,
               a.city AS agent_city, c.id AS conversation_id, u.created_at AS user_created_at,
               u.location_latitude AS user_location_latitude,
               u.location_longitude AS user_location_longitude,
               u.location_city AS user_location_city,
               u.location_region AS user_location_region,
               u.location_country AS user_location_country,
               u.location_permission_status AS user_location_permission_status,
               u.location_updated_at AS user_location_updated_at,
               s.next_activity_recommendation_at, s.last_activity_recommendation_at,
               s.last_gift_paid_at
        FROM chat_workspaces w
        JOIN ai_agents a ON a.id = w.agent_id
        JOIN users u ON u.id = w.user_id
        LEFT JOIN real_world_trigger_states s ON s.user_id = w.user_id AND s.agent_id = w.agent_id
        LEFT JOIN LATERAL (
            SELECT id FROM conversations
            WHERE workspace_id = w.id AND is_deleted = FALSE
            ORDER BY updated_at DESC LIMIT 1
        ) c ON TRUE
        WHERE w.status = 'active'
        ORDER BY w.updated_at DESC
        LIMIT $1
        """,
        limit,
    )
    return [
        {
            "workspace_id": _field(row, "workspace_id", "workspaceId"),
            "user_id": _field(row, "user_id", "userId"),
            "agent_id": _field(row, "agent_id", "agentId"),
            "agent_name": _field(row, "agent_name", "agentName") or "伴生",
            "agent_city": _field(row, "agent_city", "agentCity"),
            "conversation_id": _field(row, "conversation_id", "conversationId"),
            "user_created_at": _field(row, "user_created_at", "userCreatedAt"),
            "user_location_latitude": _field(
                row, "user_location_latitude", "userLocationLatitude"
            ),
            "user_location_longitude": _field(
                row, "user_location_longitude", "userLocationLongitude"
            ),
            "user_location_city": _field(row, "user_location_city", "userLocationCity"),
            "user_location_region": _field(row, "user_location_region", "userLocationRegion"),
            "user_location_country": _field(row, "user_location_country", "userLocationCountry"),
            "user_location_permission_status": _field(
                row, "user_location_permission_status", "userLocationPermissionStatus"
            ),
            "user_location_updated_at": _field(
                row, "user_location_updated_at", "userLocationUpdatedAt"
            ),
            "has_location": _field(row, "user_location_latitude", "userLocationLatitude")
            is not None
            and _field(row, "user_location_longitude", "userLocationLongitude") is not None
            and _field(
                row, "user_location_permission_status", "userLocationPermissionStatus"
            )
            in {"whileInUse", "always"},
            "next_activity_recommendation_at": _field(
                row, "next_activity_recommendation_at", "nextActivityRecommendationAt"
            ),
            "last_activity_recommendation_at": _field(
                row, "last_activity_recommendation_at", "lastActivityRecommendationAt"
            ),
            "last_gift_paid_at": _field(row, "last_gift_paid_at", "lastGiftPaidAt"),
        }
        for row in rows or []
    ]


async def list_user_tags(
    user_id: str,
    workspace_id: str | None,
    *,
    agent_id: str | None = None,
    limit: int = 9,
) -> list[str]:
    if agent_id:
        try:
            persisted = await profile_tags.list_profile_tags(
                user_id,
                workspace_id,
                agent_id=agent_id,
                limit=limit,
            )
            if persisted:
                return persisted
        except Exception as exc:
            logger.warning("Falling back to rule profile tags: %s", exc)
    rows = await db.query_raw(
        """
        SELECT content, summary, main_category, sub_category, importance, updated_at
        FROM memories_user
        WHERE user_id = $1
          AND ($2::text IS NULL OR workspace_id = $2)
          AND is_archived = FALSE
          AND COALESCE(summary, content, '') <> ''
          AND COALESCE(sub_category, '') <> '提醒'
        ORDER BY importance DESC, updated_at DESC
        LIMIT $3 * 4
        """,
        user_id,
        workspace_id,
        limit,
    )
    return derive_user_tags(list(rows or []), limit=limit)


async def memory_brief(user_id: str, workspace_id: str | None, *, limit: int = 60) -> str:
    rows = await db.query_raw(
        """
        SELECT content, summary, main_category, sub_category
        FROM (
            SELECT content, summary, main_category, sub_category, importance, updated_at
            FROM memories_user
            WHERE user_id = $1
              AND ($2::text IS NULL OR workspace_id = $2)
              AND is_archived = FALSE
            UNION ALL
            SELECT content, summary, main_category, sub_category, importance, updated_at
            FROM memories_ai
            WHERE user_id = $1
              AND ($2::text IS NULL OR workspace_id = $2)
              AND is_archived = FALSE
        ) m
        ORDER BY importance DESC, updated_at DESC
        LIMIT $3
        """,
        user_id,
        workspace_id,
        limit,
    )
    parts: list[str] = []
    for row in rows or []:
        label = " / ".join(
            part for part in [
                str(_field(row, "main_category", "mainCategory") or ""),
                str(_field(row, "sub_category", "subCategory") or ""),
            ] if part
        )
        text = str(_field(row, "summary") or _field(row, "content") or "").strip()
        if text:
            parts.append(f"- {label}: {text}" if label else f"- {text}")
    return "\n".join(parts)[:3000]


async def create_activity(data: dict[str, Any]) -> dict[str, Any]:
    activity_id = data.get("id") or new_id()
    starts_at = _timestamp_or_none(data.get("starts_at"))
    ends_at = _timestamp_or_none(data.get("ends_at"))
    expires_at = _timestamp_or_none(data.get("expires_at"))
    rows = await db.query_raw(
        """
        INSERT INTO offline_activity_recommendations (
            id, user_id, agent_id, workspace_id, conversation_id, status, source,
            title, summary, description, category, city, location_name, address,
            starts_at, ends_at, official_url, image_urls, search_sources,
            easter_egg_task, task_hint, expires_at
        )
        VALUES (
            $1, $2, $3, $4, $5, $6, $7,
            $8, $9, $10, $11, $12, $13, $14,
            $15::timestamptz, $16::timestamptz, $17, $18::jsonb, $19::jsonb,
            $20::jsonb, $21, $22::timestamptz
        )
        RETURNING *
        """,
        activity_id,
        data["user_id"],
        data["agent_id"],
        data.get("workspace_id"),
        data.get("conversation_id"),
        data.get("status", "pending"),
        data.get("source", "scheduled"),
        data["title"],
        data.get("summary", ""),
        data.get("description", ""),
        data.get("category"),
        data.get("city"),
        data.get("location_name"),
        data.get("address"),
        starts_at,
        ends_at,
        data.get("official_url"),
        json.dumps(data.get("image_urls") or [], ensure_ascii=False),
        json.dumps(data.get("search_sources") or [], ensure_ascii=False),
        json.dumps(data.get("easter_egg_task") or {}, ensure_ascii=False),
        data.get("task_hint"),
        expires_at,
    )
    return activity_from_row(rows[0])


async def list_activities(user_id: str, workspace_id: str | None = None) -> list[dict[str, Any]]:
    rows = await db.query_raw(
        """
        SELECT *
        FROM offline_activity_recommendations
        WHERE user_id = $1
          AND ($2::text IS NULL OR workspace_id = $2)
        ORDER BY created_at DESC
        LIMIT 100
        """,
        user_id,
        workspace_id,
    )
    return [activity_from_row(row) for row in rows or []]


async def list_recent_activity_fingerprints(
    user_id: str,
    workspace_id: str | None = None,
    *,
    limit: int = 20,
) -> list[dict[str, str]]:
    rows = await db.query_raw(
        """
        SELECT title, location_name, address, category
        FROM offline_activity_recommendations
        WHERE user_id = $1
          AND ($2::text IS NULL OR workspace_id = $2)
        ORDER BY created_at DESC
        LIMIT $3
        """,
        user_id,
        workspace_id,
        limit,
    )
    items: list[dict[str, str]] = []
    for row in rows or []:
        items.append(
            {
                "title": str(_field(row, "title") or "").strip(),
                "location_name": str(
                    _field(row, "location_name", "locationName") or ""
                ).strip(),
                "address": str(_field(row, "address") or "").strip(),
                "category": str(_field(row, "category") or "").strip(),
            }
        )
    return items


async def clear_user_activities(user_id: str) -> dict[str, int]:
    feedback_rows = await db.query_raw(
        """
        DELETE FROM offline_activity_feedback
        WHERE recommendation_id IN (
            SELECT id FROM offline_activity_recommendations WHERE user_id = $1
        )
        RETURNING id
        """,
        user_id,
    )
    activity_rows = await db.query_raw(
        """
        DELETE FROM offline_activity_recommendations
        WHERE user_id = $1
        RETURNING id
        """,
        user_id,
    )
    return {
        "deleted_activities": len(activity_rows or []),
        "deleted_feedback": len(feedback_rows or []),
    }


async def get_activity(activity_id: str, user_id: str, *, reveal_task: bool = False) -> dict[str, Any] | None:
    rows = await db.query_raw(
        "SELECT * FROM offline_activity_recommendations WHERE id = $1 AND user_id = $2 LIMIT 1",
        activity_id,
        user_id,
    )
    return activity_from_row(rows[0], reveal_task=reveal_task) if rows else None


async def update_activity_status(
    activity_id: str,
    user_id: str,
    status: str,
    *,
    completed: bool = False,
) -> dict[str, Any] | None:
    column = {
        "accepted": "accepted_at",
        "ignored": "ignored_at",
        "completed": "completed_at",
        "expired": "updated_at",
    }.get(status, "updated_at")
    rows = await db.query_raw(
        f"""
        UPDATE offline_activity_recommendations
        SET status = $3,
            {column} = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1 AND user_id = $2
        RETURNING *
        """,
        activity_id,
        user_id,
        status,
    )
    return activity_from_row(rows[0], reveal_task=True) if rows else None


async def create_activity_feedback(
    *,
    recommendation_id: str,
    user_id: str,
    kind: str,
    text: str = "",
    photo_attachment_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    await db.execute_raw(
        """
        INSERT INTO offline_activity_feedback (
            id, recommendation_id, user_id, kind, text, photo_attachment_ids, metadata
        )
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb)
        """,
        new_id(),
        recommendation_id,
        user_id,
        kind,
        text,
        json.dumps(photo_attachment_ids or [], ensure_ascii=False),
        json.dumps(metadata or {}, ensure_ascii=False),
    )


async def get_activity_completion_feedback(
    *,
    recommendation_id: str,
    user_id: str,
) -> dict[str, Any] | None:
    rows = await db.query_raw(
        """
        SELECT *
        FROM offline_activity_feedback
        WHERE recommendation_id = $1
          AND user_id = $2
          AND kind = 'completion'
        ORDER BY created_at DESC
        LIMIT 1
        """,
        recommendation_id,
        user_id,
    )
    if not rows:
        return None
    row = rows[0]
    attachment_ids = [
        str(item)
        for item in _json(
            _field(row, "photo_attachment_ids", "photoAttachmentIds"),
            [],
        )
        if str(item).strip()
    ][:3]
    attachments: list[dict[str, Any]] = []
    if attachment_ids:
        attachment_rows = await db.query_raw(
            """
            SELECT id, kind, name, mime, size, width, height, url,
                   created_at
            FROM offline_activity_media
            WHERE id = ANY($1::text[])
              AND user_id = $2
              AND recommendation_id = $3
            """,
            attachment_ids,
            user_id,
            recommendation_id,
        )
        by_id = {str(_field(item, "id")): item for item in attachment_rows or []}
        for attachment_id in attachment_ids:
            item = by_id.get(attachment_id)
            if not item:
                continue
            attachments.append(
                {
                    "id": str(_field(item, "id")),
                    "kind": str(_field(item, "kind") or "image"),
                    "name": _field(item, "name"),
                    "mime": str(_field(item, "mime") or "image/jpeg"),
                    "size": int(_field(item, "size") or 0),
                    "width": _field(item, "width"),
                    "height": _field(item, "height"),
                    "url": str(_field(item, "url") or ""),
                    "vision_status": "ready",
                    "vision_summary": None,
                    "created_at": _iso(_field(item, "created_at", "createdAt")),
                }
            )
    return {
        "text": str(_field(row, "text") or ""),
        "photo_attachments": attachments,
        "created_at": _iso(_field(row, "created_at", "createdAt")),
    }


async def default_address(user_id: str, *, masked: bool = True) -> dict[str, Any] | None:
    rows = await db.query_raw(
        """
        SELECT *
        FROM gift_addresses
        WHERE user_id = $1 AND is_default = TRUE
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        user_id,
    )
    return address_from_row(rows[0], masked=masked) if rows else None


async def upsert_address(user_id: str, data: dict[str, Any]) -> dict[str, Any]:
    existing = await default_address(user_id, masked=False)
    address_id = existing["id"] if existing else new_id()
    rows = await db.query_raw(
        """
        INSERT INTO gift_addresses (
            id, user_id, recipient_name, phone, province, city, district, detail, is_default
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, TRUE)
        ON CONFLICT (id) DO UPDATE
        SET recipient_name = EXCLUDED.recipient_name,
            phone = EXCLUDED.phone,
            province = EXCLUDED.province,
            city = EXCLUDED.city,
            district = EXCLUDED.district,
            detail = EXCLUDED.detail,
            is_default = TRUE,
            updated_at = CURRENT_TIMESTAMP
        RETURNING *
        """,
        address_id,
        user_id,
        data["recipient_name"],
        data["phone"],
        data.get("province", ""),
        data["city"],
        data.get("district", ""),
        data["detail"],
    )
    return address_from_row(rows[0], masked=True) or {}


async def create_gift(data: dict[str, Any]) -> dict[str, Any]:
    gift_id = data.get("id") or new_id()
    rows = await db.query_raw(
        """
        INSERT INTO real_world_gifts (
            id, user_id, agent_id, workspace_id, conversation_id, status, trigger_type,
            gift_name, gift_reason, gift_note, product_image_url,
            target_amount_cents, paid_amount_cents, provider, provider_order_id,
            tracking_number, address_snapshot, failure_reason, ordered_at, shipped_at, delivered_at
        )
        VALUES (
            $1, $2, $3, $4, $5, $6, $7,
            $8, $9, $10, $11,
            $12, $13, $14, $15,
            $16, $17::jsonb, $18,
            $19::timestamptz, $20::timestamptz, $21::timestamptz
        )
        RETURNING *
        """,
        gift_id,
        data["user_id"],
        data["agent_id"],
        data.get("workspace_id"),
        data.get("conversation_id"),
        data.get("status", "pending_address"),
        data.get("trigger_type", "daily_probability"),
        data.get("gift_name", ""),
        data.get("gift_reason"),
        data.get("gift_note"),
        data.get("product_image_url"),
        int(data.get("target_amount_cents") or 0),
        int(data.get("paid_amount_cents") or 0),
        data.get("provider", "mock"),
        data.get("provider_order_id"),
        data.get("tracking_number"),
        json.dumps(data.get("address_snapshot") or {}, ensure_ascii=False),
        data.get("failure_reason"),
        data.get("ordered_at"),
        data.get("shipped_at"),
        data.get("delivered_at"),
    )
    return gift_from_row(rows[0])


async def list_gifts(user_id: str, workspace_id: str | None = None) -> list[dict[str, Any]]:
    rows = await db.query_raw(
        """
        SELECT *
        FROM real_world_gifts
        WHERE user_id = $1
          AND ($2::text IS NULL OR workspace_id = $2)
        ORDER BY created_at DESC
        LIMIT 100
        """,
        user_id,
        workspace_id,
    )
    return [gift_from_row(row) for row in rows or []]


async def get_gift(gift_id: str, user_id: str) -> dict[str, Any] | None:
    rows = await db.query_raw(
        "SELECT * FROM real_world_gifts WHERE id = $1 AND user_id = $2 LIMIT 1",
        gift_id,
        user_id,
    )
    return gift_from_row(rows[0]) if rows else None


async def mark_gift_thanked(gift_id: str, user_id: str, message: str) -> dict[str, Any] | None:
    rows = await db.query_raw(
        """
        UPDATE real_world_gifts
        SET thanks_message = COALESCE(thanks_message, $3),
            thanks_sent_at = COALESCE(thanks_sent_at, CURRENT_TIMESTAMP),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1 AND user_id = $2
        RETURNING *
        """,
        gift_id,
        user_id,
        message,
    )
    return gift_from_row(rows[0]) if rows else None


async def add_tracking_events(gift_id: str, events: list[dict[str, Any]]) -> None:
    for event in events:
        await db.execute_raw(
            """
            INSERT INTO gift_tracking_events (
                id, gift_id, status, title, description, location, occurred_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7::timestamptz)
            """,
            new_id(),
            gift_id,
            event["status"],
            event["title"],
            event.get("description"),
            event.get("location"),
            event.get("occurred_at") or now_utc(),
        )


async def gift_tracking(gift_id: str, user_id: str) -> list[dict[str, Any]]:
    rows = await db.query_raw(
        """
        SELECT e.*
        FROM gift_tracking_events e
        JOIN real_world_gifts g ON g.id = e.gift_id
        WHERE e.gift_id = $1 AND g.user_id = $2
        ORDER BY e.occurred_at ASC
        """,
        gift_id,
        user_id,
    )
    return [
        {
            "id": str(_field(row, "id")),
            "status": str(_field(row, "status") or ""),
            "title": str(_field(row, "title") or ""),
            "description": _field(row, "description"),
            "location": _field(row, "location"),
            "occurred_at": _iso(_field(row, "occurred_at", "occurredAt")) or "",
        }
        for row in rows or []
    ]


async def recharge_total_cents(user_id: str) -> int:
    rows = await db.query_raw(
        """
        SELECT COALESCE(SUM(amount_cents), 0) AS total
        FROM real_world_recharge_ledger
        WHERE user_id = $1
        """,
        user_id,
    )
    return int(_field(rows[0], "total", default=0) or 0) if rows else 0


async def historical_gift_spend_cents(user_id: str) -> int:
    rows = await db.query_raw(
        """
        SELECT COALESCE(SUM(paid_amount_cents), 0) AS total
        FROM real_world_gifts
        WHERE user_id = $1 AND status IN ('ordered', 'shipping', 'delivered')
        """,
        user_id,
    )
    return int(_field(rows[0], "total", default=0) or 0) if rows else 0


def next_activity_due(base: datetime | None = None, *, accepted_delta_days: int = 0) -> datetime:
    import random

    start = base or now_utc()
    days = random.randint(20, 40) + accepted_delta_days
    return start + timedelta(days=max(8, min(days, 70)))


async def update_next_activity_due(
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    due_at: datetime,
) -> None:
    await db.execute_raw(
        """
        INSERT INTO real_world_trigger_states (
            id, user_id, agent_id, workspace_id, next_activity_recommendation_at,
            last_activity_recommendation_at
        )
        VALUES ($1, $2, $3, $4, $5::timestamptz, CURRENT_TIMESTAMP)
        ON CONFLICT (user_id, agent_id) DO UPDATE
        SET workspace_id = COALESCE(EXCLUDED.workspace_id, real_world_trigger_states.workspace_id),
            next_activity_recommendation_at = EXCLUDED.next_activity_recommendation_at,
            last_activity_recommendation_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        """,
        new_id(),
        user_id,
        agent_id,
        workspace_id,
        due_at,
    )


async def update_last_gift_paid(
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
) -> None:
    await db.execute_raw(
        """
        INSERT INTO real_world_trigger_states (
            id, user_id, agent_id, workspace_id, last_gift_paid_at
        )
        VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
        ON CONFLICT (user_id, agent_id) DO UPDATE
        SET workspace_id = COALESCE(EXCLUDED.workspace_id, real_world_trigger_states.workspace_id),
            last_gift_paid_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        """,
        new_id(),
        user_id,
        agent_id,
        workspace_id,
    )
