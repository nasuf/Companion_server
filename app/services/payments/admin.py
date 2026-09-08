"""Admin 支付/交易审计查询（复用 wallet.py 的 user-join + iso 序列化惯例）。"""

from __future__ import annotations

from typing import Any

from app.db import db
from app.services.payments.membership import product_label

_NICKNAME_SUBQUERY = """
    (
        SELECT ai.raw_profile->>'nickname'
        FROM auth_identities ai
        WHERE ai.user_id = t.user_id AND ai.provider = 'wechat'
        ORDER BY ai.updated_at DESC
        LIMIT 1
    ) AS nickname
"""


def _nickname_subquery(alias: str) -> str:
    """微信昵称子查询，按给定表别名的 user_id 关联（_NICKNAME_SUBQUERY 的通用版）。"""
    return f"""
    (
        SELECT ai.raw_profile->>'nickname'
        FROM auth_identities ai
        WHERE ai.user_id = {alias}.user_id AND ai.provider = 'wechat'
        ORDER BY ai.updated_at DESC
        LIMIT 1
    ) AS nickname
"""


def _ticket_amount_from_product(product_id: str) -> int:
    """从 com.bansheng.ticket.80 解析出充值钞票数（末段整数）。"""
    try:
        return int(product_id.rsplit(".", 1)[-1])
    except (ValueError, IndexError):
        return 0


def _field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


async def list_transactions(
    *,
    transaction_id: str | None = None,
    user_id: str | None = None,
    status: str | None = None,
    environment: str | None = None,
    kind: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    limit = min(max(limit, 1), 200)
    offset = max(offset, 0)
    clauses: list[str] = []
    params: list[Any] = [limit, offset]
    for column, value in (
        ("transaction_id", transaction_id),
        ("user_id", user_id),
        ("status", status),
        ("environment", environment),
        ("kind", kind),
    ):
        if value:
            params.append(value)
            clauses.append(f"t.{column} = ${len(params)}")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = await db.query_raw(
        f"""
        SELECT t.id, t.provider, t.transaction_id, t.original_transaction_id,
               t.product_id, t.kind, t.environment, t.user_id, t.quantity,
               t.status, t.purchase_date, t.expires_date, t.created_at,
               u.username, u.display_name, {_NICKNAME_SUBQUERY}
        FROM iap_transactions t
        LEFT JOIN users u ON u.id = t.user_id
        {where}
        ORDER BY t.created_at DESC, t.id DESC
        LIMIT $1 OFFSET $2
        """,
        *params,
    )
    return [
        {
            "id": str(_field(r, "id", "")),
            "provider": str(_field(r, "provider", "")),
            "transaction_id": str(_field(r, "transaction_id", "")),
            "original_transaction_id": _field(r, "original_transaction_id"),
            "product_id": str(_field(r, "product_id", "")),
            "kind": str(_field(r, "kind", "")),
            "environment": str(_field(r, "environment", "")),
            "user_id": str(_field(r, "user_id", "")),
            "username": _field(r, "username") or _field(r, "display_name") or _field(r, "nickname"),
            "nickname": _field(r, "nickname"),
            "quantity": int(_field(r, "quantity", 1) or 1),
            "status": str(_field(r, "status", "")),
            "purchase_date": _iso(_field(r, "purchase_date")),
            "expires_date": _iso(_field(r, "expires_date")),
            "created_at": _iso(_field(r, "created_at")) or "",
        }
        for r in rows
    ]


async def list_subscriptions(
    *, user_id: str | None = None, status: str | None = None, limit: int = 50, offset: int = 0
) -> list[dict[str, Any]]:
    limit = min(max(limit, 1), 200)
    offset = max(offset, 0)
    clauses: list[str] = []
    params: list[Any] = [limit, offset]
    if user_id:
        params.append(user_id)
        clauses.append(f"user_id = ${len(params)}")
    if status:
        params.append(status)
        clauses.append(f"status = ${len(params)}")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = await db.query_raw(
        f"""
        SELECT original_transaction_id, user_id, product_id, environment, status,
               auto_renew_status, auto_renew_product_id, expires_date,
               grace_period_expires_date, updated_at
        FROM iap_subscription_state
        {where}
        ORDER BY updated_at DESC
        LIMIT $1 OFFSET $2
        """,
        *params,
    )
    return [
        {
            "original_transaction_id": str(_field(r, "original_transaction_id", "")),
            "user_id": str(_field(r, "user_id", "")),
            "product_id": str(_field(r, "product_id", "")),
            "environment": str(_field(r, "environment", "")),
            "status": str(_field(r, "status", "")),
            "auto_renew_status": _field(r, "auto_renew_status"),
            "auto_renew_product_id": _field(r, "auto_renew_product_id"),
            "expires_date": _iso(_field(r, "expires_date")),
            "grace_period_expires_date": _iso(_field(r, "grace_period_expires_date")),
            "updated_at": _iso(_field(r, "updated_at")) or "",
        }
        for r in rows
    ]


async def list_notifications(
    *,
    notification_type: str | None = None,
    unprocessed_only: bool = False,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    limit = min(max(limit, 1), 200)
    offset = max(offset, 0)
    clauses: list[str] = []
    params: list[Any] = [limit, offset]
    if notification_type:
        params.append(notification_type)
        clauses.append(f"notification_type = ${len(params)}")
    if unprocessed_only:
        clauses.append("processed_at IS NULL")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = await db.query_raw(
        f"""
        SELECT id, notification_uuid, notification_type, subtype, environment,
               original_transaction_id, transaction_id, processed_at,
               process_error, received_at
        FROM iap_notifications
        {where}
        ORDER BY received_at DESC
        LIMIT $1 OFFSET $2
        """,
        *params,
    )
    return [
        {
            "id": str(_field(r, "id", "")),
            "notification_uuid": str(_field(r, "notification_uuid", "")),
            "notification_type": str(_field(r, "notification_type", "")),
            "subtype": _field(r, "subtype"),
            "environment": _field(r, "environment"),
            "original_transaction_id": _field(r, "original_transaction_id"),
            "transaction_id": _field(r, "transaction_id"),
            "processed_at": _iso(_field(r, "processed_at")),
            "process_error": _field(r, "process_error"),
            "received_at": _iso(_field(r, "received_at")) or "",
        }
        for r in rows
    ]


# wallet_ledger 的钞票 delta 存的是 0.1 钞票的子单位（×10），展示前要除回来。
_TICKET_SUBUNIT_SCALE = 10


async def list_vip_members(
    *,
    q: str | None = None,
    status: str | None = None,  # active | expired | None(全部)
    product_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """有 VIP（vip_until 非空）的用户列表：当前状态 + 套餐 + 过期/续期 + 最近交易时间。

    数据源：user_wallets.vip_until 为"是否有会员"的唯一真相；LEFT JOIN 最近一条订阅
    状态（自动续订=下次续期时间）+ 最近一笔 VIP 交易（套餐/发起/到账时间）。管理员手动
    设置的 VIP（无内购记录）也会出现，套餐显示"手动设置"。
    """
    from app.services.agent_template.registry import TEMPLATE_SYSTEM_USERNAME

    limit = min(max(limit, 1), 200)
    offset = max(offset, 0)

    clauses: list[str] = ["w.vip_until IS NOT NULL"]
    params: list[Any] = [TEMPLATE_SYSTEM_USERNAME]
    clauses.append(f"u.username <> ${len(params)}")
    if status == "active":
        clauses.append("w.vip_until > CURRENT_TIMESTAMP")
    elif status == "expired":
        clauses.append("w.vip_until <= CURRENT_TIMESTAMP")
    if product_id:
        params.append(product_id)
        clauses.append(f"lt.product_id = ${len(params)}")
    if q:
        params.append(f"%{q}%")
        i = len(params)
        clauses.append(
            f"(u.username ILIKE ${i} OR u.display_name ILIKE ${i} OR w.user_id ILIKE ${i})"
        )
    where = "WHERE " + " AND ".join(clauses)
    from_join = f"""
        FROM user_wallets w
        JOIN users u ON u.id = w.user_id
        LEFT JOIN LATERAL (
            SELECT ss.product_id, ss.status, ss.auto_renew_status, ss.auto_renew_product_id,
                   ss.expires_date, ss.grace_period_expires_date, ss.original_transaction_id
            FROM iap_subscription_state ss
            WHERE ss.user_id = w.user_id AND ss.provider = 'apple'
            ORDER BY ss.updated_at DESC
            LIMIT 1
        ) s ON TRUE
        LEFT JOIN LATERAL (
            SELECT t.product_id, t.kind, t.purchase_date, t.created_at,
                   t.environment, t.transaction_id
            FROM iap_transactions t
            WHERE t.user_id = w.user_id AND t.status = 'granted'
              AND (t.kind = 'subscription' OR t.product_id LIKE 'com.bansheng.vip.%')
            ORDER BY t.created_at DESC
            LIMIT 1
        ) lt ON TRUE
        {where}
    """

    count_rows = await db.query_raw(
        f"""
        SELECT COUNT(*) AS n,
               COUNT(*) FILTER (WHERE w.vip_until > CURRENT_TIMESTAMP) AS active
        {from_join}
        """,
        *params,
    )
    total = int(_field(count_rows[0], "n", 0) or 0) if count_rows else 0
    active_count = int(_field(count_rows[0], "active", 0) or 0) if count_rows else 0

    page_params = [*params, limit, offset]
    rows = await db.query_raw(
        f"""
        SELECT w.user_id, w.vip_until, w.vip_trial_used, w.vip_last_grant_at,
               (w.vip_until > CURRENT_TIMESTAMP) AS is_active,
               u.username, u.display_name,
               s.product_id AS sub_product, s.status AS sub_status,
               s.auto_renew_status, s.auto_renew_product_id,
               s.expires_date AS sub_expires, s.grace_period_expires_date,
               lt.product_id AS last_product, lt.kind AS last_kind,
               lt.purchase_date AS last_purchase, lt.created_at AS last_credited,
               lt.environment AS last_env, lt.transaction_id AS last_txn,
               {_nickname_subquery("w")}
        {from_join}
        ORDER BY (w.vip_until > CURRENT_TIMESTAMP) DESC, w.vip_until DESC NULLS LAST
        LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
        """,
        *page_params,
    )

    items: list[dict[str, Any]] = []
    for r in rows:
        sub_product = _field(r, "sub_product")
        sub_status = _field(r, "sub_status")
        auto_renew = bool(_field(r, "auto_renew_status"))
        is_auto_renew = bool(sub_product) and auto_renew and sub_status in ("active", "in_grace")
        plan_product = _field(r, "last_product") or sub_product
        items.append(
            {
                "user_id": str(_field(r, "user_id", "")),
                "username": _field(r, "username")
                or _field(r, "display_name")
                or _field(r, "nickname"),
                "nickname": _field(r, "nickname"),
                "is_active": bool(_field(r, "is_active")),
                "plan_label": product_label(plan_product) if plan_product else "手动设置",
                "product_id": plan_product,
                "is_auto_renew": is_auto_renew,
                "subscription_status": sub_status,
                "vip_until": _iso(_field(r, "vip_until")),
                # 下次续期仅对"自动续订生效中"有意义；一次性时长包/手动设置无续期。
                "next_renewal_date": _iso(_field(r, "sub_expires")) if is_auto_renew else None,
                "grace_period_expires_date": _iso(_field(r, "grace_period_expires_date")),
                "vip_trial_used": bool(_field(r, "vip_trial_used")),
                "environment": _field(r, "last_env"),
                "started_at": _iso(_field(r, "last_purchase")),
                "credited_at": _iso(_field(r, "last_credited")),
                "last_transaction_id": _field(r, "last_txn"),
            }
        )
    return {"items": items, "total": total, "active_count": active_count}


async def list_recharges(
    *,
    q: str | None = None,
    status: str | None = None,
    environment: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """钞票充值（消耗型内购）流水：到账钞票 + 实付金额 + 发起/到账时间 + 状态。

    只列 com.bansheng.ticket.* 商品（商城充值 tab 的实际充值行为）。到账钞票以
    wallet_ledger.delta（子单位 /10）为准，缺失时按商品档位推算；实付金额取交易
    payload 里 Apple 回的 price/currency（仅新交易有）。
    """
    limit = min(max(limit, 1), 200)
    offset = max(offset, 0)

    clauses: list[str] = ["t.product_id LIKE 'com.bansheng.ticket.%'"]
    params: list[Any] = []
    if status:
        params.append(status)
        clauses.append(f"t.status = ${len(params)}")
    if environment:
        params.append(environment)
        clauses.append(f"t.environment = ${len(params)}")
    if q:
        params.append(f"%{q}%")
        i = len(params)
        clauses.append(
            f"(u.username ILIKE ${i} OR u.display_name ILIKE ${i} "
            f"OR t.user_id ILIKE ${i} OR t.transaction_id ILIKE ${i})"
        )
    where = "WHERE " + " AND ".join(clauses)
    from_join = f"""
        FROM iap_transactions t
        LEFT JOIN users u ON u.id = t.user_id
        LEFT JOIN wallet_ledger l
            ON l.source = 'iap_apple' AND l.source_id = t.transaction_id AND l.currency = 'ticket'
        {where}
    """

    count_rows = await db.query_raw(
        f"""
        SELECT COUNT(*) AS n,
               COUNT(DISTINCT t.user_id) AS users,
               COALESCE(SUM(l.delta) FILTER (WHERE t.status = 'granted'), 0) AS tickets_sub
        {from_join}
        """,
        *params,
    )
    total = int(_field(count_rows[0], "n", 0) or 0) if count_rows else 0
    distinct_users = int(_field(count_rows[0], "users", 0) or 0) if count_rows else 0
    tickets_sub = int(_field(count_rows[0], "tickets_sub", 0) or 0) if count_rows else 0
    total_tickets = round(tickets_sub / _TICKET_SUBUNIT_SCALE)

    page_params = [*params, limit, offset]
    rows = await db.query_raw(
        f"""
        SELECT t.transaction_id, t.user_id, t.product_id, t.quantity,
               t.environment, t.status, t.purchase_date, t.created_at, t.updated_at,
               u.username, u.display_name,
               l.delta AS delta_sub,
               t.raw_transaction_payload->>'price' AS price_milli,
               t.raw_transaction_payload->>'currency' AS currency,
               t.raw_transaction_payload->>'storefront' AS storefront,
               {_NICKNAME_SUBQUERY}
        {from_join}
        ORDER BY t.created_at DESC, t.transaction_id DESC
        LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
        """,
        *page_params,
    )

    items: list[dict[str, Any]] = []
    for r in rows:
        product_id = str(_field(r, "product_id", ""))
        quantity = int(_field(r, "quantity", 1) or 1)
        delta_sub = _field(r, "delta_sub")
        if delta_sub is not None:
            tickets = round(int(delta_sub) / _TICKET_SUBUNIT_SCALE)
        else:
            tickets = _ticket_amount_from_product(product_id) * quantity
        price_milli = _field(r, "price_milli")
        amount: float | None = None
        if price_milli not in (None, ""):
            try:
                amount = int(price_milli) / 1000
            except (TypeError, ValueError):
                amount = None
        pack = _ticket_amount_from_product(product_id)
        items.append(
            {
                "transaction_id": str(_field(r, "transaction_id", "")),
                "user_id": str(_field(r, "user_id", "")),
                "username": _field(r, "username")
                or _field(r, "display_name")
                or _field(r, "nickname"),
                "nickname": _field(r, "nickname"),
                "product_id": product_id,
                "product_label": f"{pack}钞票" if pack else product_id,
                "tickets": tickets,
                "quantity": quantity,
                "amount": amount,
                "currency": _field(r, "currency"),
                "storefront": _field(r, "storefront"),
                "environment": str(_field(r, "environment", "")),
                "status": str(_field(r, "status", "")),
                "initiated_at": _iso(_field(r, "purchase_date")),
                "credited_at": _iso(_field(r, "created_at")),
                "updated_at": _iso(_field(r, "updated_at")),
            }
        )
    return {
        "items": items,
        "total": total,
        "total_tickets": total_tickets,
        "distinct_users": distinct_users,
    }
