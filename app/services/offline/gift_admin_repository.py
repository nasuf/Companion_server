"""后台礼物管理的数据访问。跨用户查询只在这里出现，调用方必须是管理员接口。"""

from __future__ import annotations

import json
from typing import Any

from app.db import db
from app.services.offline.gift_repository import gift_from_row
from app.services.offline.gift_status import ACTIVE_STATUSES, CLOSED_STATUSES, IN_TRANSIT_STATUSES
from app.services.offline.repository import _field, _iso, new_id

ADMIN_LIST_MAX_LIMIT = 100
_MESSAGE_LIMIT = 30
_MEMORY_SCAN_LIMIT = 60

_STATUS_FILTERS: dict[str, list[str] | None] = {
    "all": None,
    "in_transit": list(IN_TRANSIT_STATUSES),
    "out_for_delivery": ["out_for_delivery"],
    "delivered": ["delivered"],
    "closed": [s for s in CLOSED_STATUSES if s != "delivered"],
}

# 后台列表/详情共用的 SELECT：礼物 + 用户 + Agent + 最近一条物流节点。
_GIFT_SELECT = """
    SELECT g.*,
           u.username,
           u.display_name,
           (
               SELECT ai.raw_profile->>'nickname'
               FROM auth_identities ai
               WHERE ai.user_id = u.id AND ai.provider = 'wechat'
               ORDER BY ai.updated_at DESC
               LIMIT 1
           ) AS nickname,
           a.name AS agent_name,
           le.title AS last_event_title,
           le.occurred_at AS last_event_at
    FROM real_world_gifts g
    JOIN users u ON u.id = g.user_id
    LEFT JOIN ai_agents a ON a.id = g.agent_id
    LEFT JOIN LATERAL (
        SELECT e.title, e.occurred_at
        FROM gift_tracking_events e
        WHERE e.gift_id = g.id
        ORDER BY e.occurred_at DESC, e.created_at DESC
        LIMIT 1
    ) le ON TRUE
"""

_LIST_WHERE = """
    WHERE ($1::text[] IS NULL OR g.status = ANY($1::text[]))
      AND ($2::text IS NULL OR g.user_id = $2)
      AND (
          $3::text IS NULL
          OR g.id = $4
          OR g.gift_name ILIKE $3
          OR g.tracking_number ILIKE $3
          OR u.username ILIKE $3
      )
"""


def admin_gift_from_row(row: Any) -> dict[str, Any]:
    gift = gift_from_row(row)
    snapshot = _json_dict(_field(row, "product_snapshot", "productSnapshot"))
    return {
        **gift,
        "username": _field(row, "username"),
        "nickname": _field(row, "nickname") or _field(row, "display_name"),
        "agent_name": _field(row, "agent_name"),
        "gift_reason": _field(row, "gift_reason", "giftReason"),
        "carrier": gift.get("logistics_provider"),
        "address_snapshot": _json_dict(_field(row, "address_snapshot", "addressSnapshot")),
        "recommendation": snapshot.get("recommendation") if isinstance(snapshot.get("recommendation"), dict) else None,
        "failure_reason": _field(row, "failure_reason", "failureReason"),
        "thanks_message": _field(row, "thanks_message", "thanksMessage"),
        "last_event_title": _field(row, "last_event_title"),
        "last_event_at": _iso(_field(row, "last_event_at")),
    }


async def list_gifts(
    *,
    status_filter: str = "all",
    user_id: str | None = None,
    query: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    statuses = _STATUS_FILTERS.get(status_filter)
    q = (query or "").strip()
    pattern = f"%{q}%" if q else None
    limit = min(max(limit, 1), ADMIN_LIST_MAX_LIMIT)
    offset = max(offset, 0)
    params = (statuses, user_id, pattern, q or None)
    rows = await db.query_raw(
        f"{_GIFT_SELECT} {_LIST_WHERE} ORDER BY g.updated_at DESC, g.id LIMIT $5 OFFSET $6",
        *params,
        limit,
        offset,
    )
    count_rows = await db.query_raw(
        f"""
        SELECT COUNT(*)::int AS total
        FROM real_world_gifts g
        JOIN users u ON u.id = g.user_id
        {_LIST_WHERE}
        """,
        *params,
    )
    # 计数失败要报错，不能当成 0 份礼物。
    total = int(_field(count_rows[0], "total"))
    return [admin_gift_from_row(row) for row in rows or []], total


async def get_gift(gift_id: str) -> dict[str, Any] | None:
    rows = await db.query_raw(f"{_GIFT_SELECT} WHERE g.id = $1 LIMIT 1", gift_id)
    return admin_gift_from_row(rows[0]) if rows else None


async def list_tracking_events(gift_id: str) -> list[dict[str, Any]]:
    # 后台看原始轨迹（不去重），便于发现重复录入。
    rows = await db.query_raw(
        """
        SELECT id, status, title, description, location, occurred_at, created_at
        FROM gift_tracking_events
        WHERE gift_id = $1
        ORDER BY occurred_at ASC, created_at ASC
        LIMIT 200
        """,
        gift_id,
    )
    return [_event_from_row(row) for row in rows or []]


async def gift_messages(gift_id: str, user_id: str) -> list[dict[str, Any]]:
    """这份礼物在聊天里产生的消息（寄出/派送/签收/感谢），用于核对用户实际看到了什么。"""
    rows = await db.query_raw(
        """
        SELECT m.id, m.role, m.content, m.created_at,
               m.metadata->>'trigger_type' AS trigger_type,
               (m.metadata ? 'trace_id') AS has_trace
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        WHERE c.user_id = $2
          AND m.metadata->>'real_world_type' = 'gift'
          AND m.metadata->>'source_id' = $1
        ORDER BY m.created_at ASC
        LIMIT $3
        """,
        gift_id,
        user_id,
        _MESSAGE_LIMIT,
    )
    return [
        {
            "id": str(_field(row, "id")),
            "role": str(_field(row, "role") or ""),
            "trigger_type": _field(row, "trigger_type"),
            "text_preview": str(_field(row, "content") or "")[:120],
            "has_trace": bool(_field(row, "has_trace")),
            "created_at": _iso(_field(row, "created_at", "createdAt")) or "",
        }
        for row in rows or []
    ]


async def insert_manual_gift(data: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    """插入后台手动礼物。返回 (礼物, 是否新建)；同一个 request id 重复提交不会再建一份。"""
    rows = await db.query_raw(
        """
        INSERT INTO real_world_gifts (
            id, user_id, agent_id, workspace_id, conversation_id, status, trigger_type,
            gift_name, gift_reason, gift_note, product_image_url, product_snapshot,
            target_amount_cents, paid_amount_cents, provider, tracking_number,
            logistics_provider, address_snapshot, ordered_at, shipped_at
        )
        VALUES (
            $1, $2, $3, $4, $5, $6, 'admin_manual',
            $7, $8, $9, $10, $11::jsonb,
            $12, $12, 'manual', $13,
            $14, $15::jsonb, CURRENT_TIMESTAMP,
            CASE WHEN $6::text = 'shipping' THEN CURRENT_TIMESTAMP ELSE NULL END
        )
        ON CONFLICT (id) DO NOTHING
        RETURNING *
        """,
        data["id"],
        data["user_id"],
        data["agent_id"],
        data.get("workspace_id"),
        data.get("conversation_id"),
        data["status"],
        data["gift_name"],
        data.get("gift_reason"),
        data.get("gift_note"),
        data.get("product_image_url"),
        json.dumps(data.get("product_snapshot") or {}, ensure_ascii=False),
        int(data.get("paid_amount_cents") or 0),
        data.get("tracking_number"),
        data.get("logistics_provider"),
        json.dumps(data.get("address_snapshot") or {}, ensure_ascii=False),
    )
    if rows:
        return gift_from_row(rows[0]), True
    existing = await get_gift(data["id"])
    if existing and existing["user_id"] == data["user_id"]:
        return existing, False
    return None, False


_UPDATABLE_COLUMNS = {
    "gift_name": "gift_name",
    "gift_note": "gift_note",
    "gift_reason": "gift_reason",
    "product_image_url": "product_image_url",
    "paid_amount_cents": "paid_amount_cents",
    "logistics_provider": "logistics_provider",
    "tracking_number": "tracking_number",
}


async def update_gift_details(gift_id: str, fields: dict[str, Any]) -> dict[str, Any] | None:
    columns = [(column, fields[key]) for key, column in _UPDATABLE_COLUMNS.items() if key in fields]
    if not columns:
        return await get_gift(gift_id)
    assignments = ", ".join(f"{column} = ${index + 2}" for index, (column, _) in enumerate(columns))
    rows = await db.query_raw(
        f"""
        UPDATE real_world_gifts
        SET {assignments}, updated_at = CURRENT_TIMESTAMP
        WHERE id = $1 AND status <> 'cancelled'
        RETURNING id
        """,
        gift_id,
        *[value for _, value in columns],
    )
    return await get_gift(gift_id) if rows else None


async def set_transit_status(gift_id: str, status: str) -> bool:
    """在途状态之间手动调整（含纠错回退）。进入派送中走 gift_service 的 CAS + 通知。"""
    rows = await db.query_raw(
        """
        UPDATE real_world_gifts
        SET status = $2::text,
            shipped_at = CASE WHEN $2::text = 'shipping' THEN COALESCE(shipped_at, CURRENT_TIMESTAMP)
                              ELSE shipped_at END,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1 AND status = ANY($3::text[])
        RETURNING id
        """,
        gift_id,
        status,
        list(IN_TRANSIT_STATUSES),
    )
    return bool(rows)


async def insert_tracking_event(gift_id: str, event: dict[str, Any]) -> dict[str, Any] | None:
    """插入一条物流节点；与已有节点完全相同时返回 None（视为重复录入）。"""
    rows = await db.query_raw(
        """
        INSERT INTO gift_tracking_events (id, gift_id, status, title, description, location, occurred_at)
        SELECT $1, $2, $3::text, $4::text, $5::text, $6::text, COALESCE($7::timestamptz, CURRENT_TIMESTAMP)
        WHERE NOT EXISTS (
            SELECT 1 FROM gift_tracking_events
            WHERE gift_id = $2
              AND status = $3
              AND title = $4
              AND COALESCE(description, '') = COALESCE($5::text, '')
              AND COALESCE(location, '') = COALESCE($6::text, '')
        )
        RETURNING id, status, title, description, location, occurred_at, created_at
        """,
        new_id(),
        gift_id,
        event["status"],
        event["title"],
        event.get("description"),
        event.get("location"),
        event.get("occurred_at"),
    )
    if rows:
        await db.execute_raw(
            "UPDATE real_world_gifts SET updated_at = CURRENT_TIMESTAMP WHERE id = $1",
            gift_id,
        )
    return _event_from_row(rows[0]) if rows else None


async def delete_events_after_stage(gift_id: str, statuses: list[str]) -> int:
    """回退物流进度时删掉更靠后的节点（签收记录不在其中，永远保留）。"""
    rows = await db.query_raw(
        """
        DELETE FROM gift_tracking_events
        WHERE gift_id = $1 AND status = ANY($2::text[]) AND status <> 'delivered'
        RETURNING id
        """,
        gift_id,
        statuses,
    )
    return len(rows or [])


async def cancel_gift(gift_id: str, reason: str) -> bool:
    rows = await db.query_raw(
        """
        UPDATE real_world_gifts
        SET status = 'cancelled',
            failure_reason = NULLIF($2, ''),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1 AND status = ANY($3::text[])
        RETURNING id
        """,
        gift_id,
        reason,
        list(ACTIVE_STATUSES),
    )
    return bool(rows)


async def get_user(user_id: str) -> dict[str, Any] | None:
    rows = await db.query_raw(
        """
        SELECT u.id, u.username, u.display_name, u.created_at,
               (
                   SELECT ai.raw_profile->>'nickname'
                   FROM auth_identities ai
                   WHERE ai.user_id = u.id AND ai.provider = 'wechat'
                   ORDER BY ai.updated_at DESC
                   LIMIT 1
               ) AS nickname
        FROM users u
        WHERE u.id = $1
        LIMIT 1
        """,
        user_id,
    )
    if not rows:
        return None
    row = rows[0]
    return {
        "user_id": str(_field(row, "id")),
        "username": str(_field(row, "username") or ""),
        "nickname": _field(row, "nickname"),
        "display_name": _field(row, "display_name"),
        "created_at": _iso(_field(row, "created_at")),
    }


async def list_user_workspaces(user_id: str) -> list[dict[str, Any]]:
    rows = await db.query_raw(
        """
        SELECT w.id AS workspace_id, w.agent_id, a.name AS agent_name,
               (
                   SELECT c.id FROM conversations c
                   WHERE c.workspace_id = w.id AND c.is_deleted = FALSE
                   ORDER BY c.updated_at DESC
                   LIMIT 1
               ) AS conversation_id
        FROM chat_workspaces w
        LEFT JOIN ai_agents a ON a.id = w.agent_id
        WHERE w.user_id = $1 AND w.status = 'active'
        ORDER BY w.updated_at DESC
        LIMIT 20
        """,
        user_id,
    )
    return [
        {
            "workspace_id": str(_field(row, "workspace_id")),
            "agent_id": _field(row, "agent_id"),
            "agent_name": _field(row, "agent_name"),
            "conversation_id": _field(row, "conversation_id"),
        }
        for row in rows or []
    ]


async def user_memories_for_gift(user_id: str, workspace_id: str | None) -> list[dict[str, Any]]:
    """礼物分析用的用户记忆（只取用户侧，不混入 AI 人设记忆；提醒类不参与）。"""
    rows = await db.query_raw(
        """
        SELECT id, content, main_category, sub_category, updated_at
        FROM memories_user
        WHERE user_id = $1
          AND ($2::text IS NULL OR workspace_id = $2)
          AND is_archived = FALSE
          AND COALESCE(content, '') <> ''
          AND COALESCE(sub_category, '') <> '提醒'
        ORDER BY importance DESC, updated_at DESC
        LIMIT $3
        """,
        user_id,
        workspace_id,
        _MEMORY_SCAN_LIMIT,
    )
    return [
        {
            "id": str(_field(row, "id")),
            "content": str(_field(row, "content") or "").strip(),
            "category": " / ".join(
                part for part in (
                    str(_field(row, "main_category") or ""),
                    str(_field(row, "sub_category") or ""),
                ) if part
            ) or None,
            "updated_at": _iso(_field(row, "updated_at")),
        }
        for row in rows or []
    ]


async def count_user_memories(user_id: str, workspace_id: str | None) -> int:
    rows = await db.query_raw(
        """
        SELECT COUNT(*)::int AS total
        FROM memories_user
        WHERE user_id = $1 AND ($2::text IS NULL OR workspace_id = $2) AND is_archived = FALSE
        """,
        user_id,
        workspace_id,
    )
    return int(_field(rows[0], "total"))


async def gift_names_for_user(user_id: str) -> list[str]:
    """所有寄过 / 在途的礼物名（跨工作区），推荐时排除。"""
    rows = await db.query_raw(
        """
        SELECT gift_name
        FROM real_world_gifts
        WHERE user_id = $1 AND status <> 'cancelled' AND gift_name <> ''
        ORDER BY created_at DESC
        LIMIT 50
        """,
        user_id,
    )
    return [str(_field(row, "gift_name")) for row in rows or []]


def _event_from_row(row: Any) -> dict[str, Any]:
    return {
        "id": str(_field(row, "id")),
        "status": str(_field(row, "status") or ""),
        "title": str(_field(row, "title") or ""),
        "description": _field(row, "description"),
        "location": _field(row, "location"),
        "occurred_at": _iso(_field(row, "occurred_at", "occurredAt")) or "",
        "created_at": _iso(_field(row, "created_at", "createdAt")),
    }


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}
