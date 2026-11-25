"""Web 后台的微信支付订单查询与人工退款。

可观测性要求（AGENTS.md 每批必须评估）：这一页要能回答运维三个问题——
1. 用户付了钱但没到账，卡在哪一步？→ status + 回调是否到达 + 处理错误。
2. 到账了吗、发了什么？→ granted_at、vip_days / ticket_amount、实付金额。
3. 能不能退？退到哪一步了？→ refunded_fen、退款单状态。

刻意区分四种"没到账"：
* `created`：下单了但用户没付（或没付完）——回调根本不会来。
* `paid`：微信说付了但我方还没发放（极短暂，或发放中途崩）。
* 有回调但 `process_error` 非空：收到了、处理失败了。
* 完全没有回调行：回调没到（地址错、被拦、微信还没推）。
这四种原因不同，不能混成一句"失败"。
"""

from __future__ import annotations

import logging
from typing import Any

from app.db import db
from app.services.payments.wechat import refunds

logger = logging.getLogger(__name__)

_ALLOWED_STATUSES = frozenset(
    {"created", "paid", "granted", "closed", "refunded", "failed"}
)


def _field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _iso(value: Any) -> str | None:
    # query_raw 把 timestamp 列返回成 ISO 字符串；ORM/测试里可能是 datetime。
    if isinstance(value, str):
        return value or None
    return value.isoformat() if hasattr(value, "isoformat") else None


async def list_orders(
    *,
    status: str | None = None,
    user_id: str | None = None,
    query: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """分页列订单。``limit`` 有界（见调用方的 Query 约束），避免拉全表。"""
    normalized_status = status if status in _ALLOWED_STATUSES else None
    search = f"%{query.strip()}%" if query and query.strip() else None

    rows = await db.query_raw(
        """
        SELECT o.out_trade_no, o.transaction_id, o.user_id, u.username,
               u.display_name AS nickname, o.product_id, o.product_title,
               o.amount_fen, o.paid_amount_fen, o.vip_days, o.ticket_amount,
               o.status, o.trade_state, o.refunded_fen, o.fail_reason,
               o.success_time, o.granted_at, o.created_at,
               n.notification_count, n.last_notification_at, n.last_notification_error
        FROM wechat_pay_orders o
        LEFT JOIN users u ON u.id = o.user_id
        LEFT JOIN LATERAL (
            SELECT COUNT(*)::int AS notification_count,
                   MAX(received_at) AS last_notification_at,
                   (ARRAY_AGG(process_error ORDER BY received_at DESC))[1]
                       AS last_notification_error
            FROM wechat_pay_notifications
            WHERE out_trade_no = o.out_trade_no
        ) n ON TRUE
        WHERE ($1::text IS NULL OR o.status = $1)
          AND ($2::text IS NULL OR o.user_id = $2)
          AND (
            $3::text IS NULL
            OR o.out_trade_no ILIKE $3
            OR o.transaction_id ILIKE $3
            OR u.username ILIKE $3
          )
        ORDER BY o.created_at DESC
        LIMIT $4 OFFSET $5
        """,
        normalized_status,
        user_id,
        search,
        limit,
        offset,
    )

    totals = await db.query_raw(
        """
        SELECT COUNT(*)::int AS total,
               COUNT(*) FILTER (WHERE o.status = 'granted')::int AS granted_count,
               COALESCE(SUM(o.paid_amount_fen) FILTER (WHERE o.status = 'granted'), 0)::int
                   AS paid_amount_fen_total
        FROM wechat_pay_orders o
        LEFT JOIN users u ON u.id = o.user_id
        WHERE ($1::text IS NULL OR o.status = $1)
          AND ($2::text IS NULL OR o.user_id = $2)
          AND (
            $3::text IS NULL
            OR o.out_trade_no ILIKE $3
            OR o.transaction_id ILIKE $3
            OR u.username ILIKE $3
          )
        """,
        normalized_status,
        user_id,
        search,
    )
    summary = totals[0] if totals else {}

    return {
        "items": [
            {
                "out_trade_no": str(_field(row, "out_trade_no")),
                "transaction_id": _field(row, "transaction_id"),
                "user_id": str(_field(row, "user_id")),
                "username": _field(row, "username"),
                "nickname": _field(row, "nickname"),
                "product_id": str(_field(row, "product_id")),
                "product_title": str(_field(row, "product_title")),
                "amount_fen": int(_field(row, "amount_fen") or 0),
                "paid_amount_fen": _field(row, "paid_amount_fen"),
                "vip_days": int(_field(row, "vip_days") or 0),
                "ticket_amount": int(_field(row, "ticket_amount") or 0),
                "status": str(_field(row, "status")),
                "trade_state": _field(row, "trade_state"),
                "refunded_fen": int(_field(row, "refunded_fen") or 0),
                "fail_reason": _field(row, "fail_reason"),
                "notification_count": int(_field(row, "notification_count") or 0),
                "last_notification_at": _iso(_field(row, "last_notification_at")),
                "last_notification_error": _field(row, "last_notification_error"),
                "success_time": _iso(_field(row, "success_time")),
                "granted_at": _iso(_field(row, "granted_at")),
                "created_at": _iso(_field(row, "created_at")) or "",
            }
            for row in rows
        ],
        "total": int(_field(summary, "total") or 0),
        "granted_count": int(_field(summary, "granted_count") or 0),
        "paid_amount_fen_total": int(_field(summary, "paid_amount_fen_total") or 0),
    }


async def refund_order(
    *, out_trade_no: str, actor_id: str, reason: str | None
) -> dict[str, Any]:
    """人工全额退款。实际出款与权益回收在 `refunds` 模块里，这里只转交。"""
    return await refunds.create_refund(
        out_trade_no=out_trade_no, actor_id=actor_id, reason=reason
    )
