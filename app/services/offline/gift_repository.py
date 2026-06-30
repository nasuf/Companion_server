from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from app.db import db
from app.services.offline.repository import _field, _iso, new_id, now_utc


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
        "provider_product_id": _field(row, "provider_product_id", "providerProductId"),
        "product_url": _field(row, "product_url", "productUrl"),
        "product_snapshot": _field(row, "product_snapshot", "productSnapshot") or {},
        "provider": str(_field(row, "provider") or "mock"),
        "provider_order_id": _field(row, "provider_order_id", "providerOrderId"),
        "logistics_provider": _field(row, "logistics_provider", "logisticsProvider"),
        "provider_payload": _field(row, "provider_payload", "providerPayload") or {},
        "logistics_payload": _field(row, "logistics_payload", "logisticsPayload") or {},
        "last_tracking_synced_at": _iso(_field(row, "last_tracking_synced_at", "lastTrackingSyncedAt")),
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
            provider_product_id, product_url, product_snapshot,
            target_amount_cents, paid_amount_cents, provider, provider_order_id,
            tracking_number, logistics_provider, provider_payload, logistics_payload,
            last_tracking_synced_at, address_snapshot, failure_reason,
            ordered_at, shipped_at, delivered_at
        )
        VALUES (
            $1, $2, $3, $4, $5, $6, $7,
            $8, $9, $10, $11,
            $12, $13, $14::jsonb,
            $15, $16, $17, $18,
            $19, $20, $21::jsonb, $22::jsonb,
            $23::timestamptz, $24::jsonb, $25,
            $26::timestamptz, $27::timestamptz, $28::timestamptz
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
        data.get("provider_product_id"),
        data.get("product_url"),
        json.dumps(data.get("product_snapshot") or {}, ensure_ascii=False),
        int(data.get("target_amount_cents") or 0),
        int(data.get("paid_amount_cents") or 0),
        data.get("provider", "mock"),
        data.get("provider_order_id"),
        data.get("tracking_number"),
        data.get("logistics_provider"),
        json.dumps(data.get("provider_payload") or {}, ensure_ascii=False),
        json.dumps(data.get("logistics_payload") or {}, ensure_ascii=False),
        data.get("last_tracking_synced_at"),
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


async def update_gift_order_details(gift_id: str, user_id: str, data: dict[str, Any]) -> dict[str, Any] | None:
    rows = await db.query_raw(
        """
        UPDATE real_world_gifts
        SET status = $3,
            gift_name = $4,
            gift_reason = $5,
            gift_note = $6,
            product_image_url = $7,
            provider_product_id = $8,
            product_url = $9,
            product_snapshot = $10::jsonb,
            target_amount_cents = $11,
            paid_amount_cents = $12,
            provider = $13,
            provider_order_id = $14,
            tracking_number = $15,
            logistics_provider = $16,
            provider_payload = $17::jsonb,
            logistics_payload = $18::jsonb,
            last_tracking_synced_at = $19::timestamptz,
            address_snapshot = $20::jsonb,
            failure_reason = NULL,
            ordered_at = $21::timestamptz,
            shipped_at = $22::timestamptz,
            delivered_at = $23::timestamptz,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1 AND user_id = $2
        RETURNING *
        """,
        gift_id,
        user_id,
        data.get("status", "shipping"),
        data.get("gift_name", ""),
        data.get("gift_reason"),
        data.get("gift_note"),
        data.get("product_image_url"),
        data.get("provider_product_id"),
        data.get("product_url"),
        json.dumps(data.get("product_snapshot") or {}, ensure_ascii=False),
        int(data.get("target_amount_cents") or 0),
        int(data.get("paid_amount_cents") or 0),
        data.get("provider", "mock"),
        data.get("provider_order_id"),
        data.get("tracking_number"),
        data.get("logistics_provider"),
        json.dumps(data.get("provider_payload") or {}, ensure_ascii=False),
        json.dumps(data.get("logistics_payload") or {}, ensure_ascii=False),
        data.get("last_tracking_synced_at"),
        json.dumps(data.get("address_snapshot") or {}, ensure_ascii=False),
        data.get("ordered_at"),
        data.get("shipped_at"),
        data.get("delivered_at"),
    )
    return gift_from_row(rows[0]) if rows else None


async def update_gift_status(
    gift_id: str,
    user_id: str,
    status: str,
    *,
    failure_reason: str | None = None,
) -> dict[str, Any] | None:
    rows = await db.query_raw(
        """
        UPDATE real_world_gifts
        SET status = $3,
            failure_reason = $4,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1 AND user_id = $2
        RETURNING *
        """,
        gift_id,
        user_id,
        status,
        failure_reason,
    )
    return gift_from_row(rows[0]) if rows else None


async def mark_gift_delivered(
    gift_id: str,
    user_id: str,
    delivered_at: datetime | None = None,
) -> dict[str, Any] | None:
    rows = await db.query_raw(
        """
        UPDATE real_world_gifts
        SET status = 'delivered',
            delivered_at = COALESCE($3::timestamptz, delivered_at, CURRENT_TIMESTAMP),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1 AND user_id = $2
        RETURNING *
        """,
        gift_id,
        user_id,
        delivered_at,
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
        occurred_at = event.get("occurred_at") or now_utc()
        description = event.get("description")
        location = event.get("location")
        exists = await db.query_raw(
            """
            SELECT 1
            FROM gift_tracking_events
            WHERE gift_id = $1
              AND status = $2
              AND title = $3
              AND COALESCE(description, '') = COALESCE($4::text, '')
              AND COALESCE(location, '') = COALESCE($5::text, '')
            LIMIT 1
            """,
            gift_id,
            event["status"],
            event["title"],
            description,
            location,
        )
        if exists:
            continue
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
            description,
            location,
            occurred_at,
        )


async def update_tracking_snapshot(
    gift_id: str,
    user_id: str,
    *,
    status: str | None = None,
    tracking_number: str | None = None,
    logistics_provider: str | None = None,
    logistics_payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    rows = await db.query_raw(
        """
        UPDATE real_world_gifts
        SET status = COALESCE($3, status),
            tracking_number = COALESCE($4, tracking_number),
            logistics_provider = COALESCE($5, logistics_provider),
            logistics_payload = COALESCE($6::jsonb, logistics_payload),
            last_tracking_synced_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1 AND user_id = $2
        RETURNING *
        """,
        gift_id,
        user_id,
        status,
        tracking_number,
        logistics_provider,
        json.dumps(logistics_payload, ensure_ascii=False) if logistics_payload is not None else None,
    )
    return gift_from_row(rows[0]) if rows else None


async def gift_tracking(gift_id: str, user_id: str) -> list[dict[str, Any]]:
    rows = await db.query_raw(
        """
        SELECT *
        FROM (
            SELECT DISTINCT ON (
                e.status,
                e.title,
                COALESCE(e.description, ''),
                COALESCE(e.location, '')
            ) e.*
            FROM gift_tracking_events e
            JOIN real_world_gifts g ON g.id = e.gift_id
            WHERE e.gift_id = $1 AND g.user_id = $2
            ORDER BY
                e.status,
                e.title,
                COALESCE(e.description, ''),
                COALESCE(e.location, ''),
                e.occurred_at ASC
        ) deduped
        ORDER BY occurred_at ASC
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


async def tracking_status_exists(gift_id: str, status: str) -> bool:
    rows = await db.query_raw(
        """
        SELECT 1
        FROM gift_tracking_events
        WHERE gift_id = $1 AND status = $2
        LIMIT 1
        """,
        gift_id,
        status,
    )
    return bool(rows)


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


async def user_birthday_mmdd(user_id: str, workspace_id: str | None) -> tuple[int, int] | None:
    rows = await db.query_raw(
        """
        SELECT content, summary
        FROM memories_user
        WHERE user_id = $1
          AND ($2::text IS NULL OR workspace_id = $2)
          AND is_archived = FALSE
          AND main_category = '身份'
          AND sub_category = '生日'
        ORDER BY level ASC, created_at ASC
        LIMIT 1
        """,
        user_id,
        workspace_id,
    )
    if not rows:
        return None
    text = f"{_field(rows[0], 'summary') or ''} {_field(rows[0], 'content') or ''}"
    match = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})", text)
    if not match:
        match = re.search(r"(\d{1,2})[/-](\d{1,2})", text)
    if not match:
        return None
    month, day = int(match.group(1)), int(match.group(2))
    if 1 <= month <= 12 and 1 <= day <= 31:
        return month, day
    return None


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
