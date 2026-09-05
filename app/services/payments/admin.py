"""Admin 支付/交易审计查询（复用 wallet.py 的 user-join + iso 序列化惯例）。"""

from __future__ import annotations

from typing import Any

from app.db import db

_NICKNAME_SUBQUERY = """
    (
        SELECT ai.raw_profile->>'nickname'
        FROM auth_identities ai
        WHERE ai.user_id = t.user_id AND ai.provider = 'wechat'
        ORDER BY ai.updated_at DESC
        LIMIT 1
    ) AS nickname
"""


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
