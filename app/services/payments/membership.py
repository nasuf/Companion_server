"""User-facing IAP membership summary: VIP state, auto-renew subscription, purchase history."""

from __future__ import annotations

from typing import Any

from app.db import db
from app.services import wallet
from app.services.payments import grant
from app.services.payments.catalog import APPLE_PRODUCTS

_VIP_PRODUCT_PREFIX = "com.bansheng.vip"

_PRODUCT_LABELS: dict[str, str] = {
    "com.bansheng.vip.monthly.auto": "连续包月",
    "com.bansheng.vip.month": "月卡",
    "com.bansheng.vip.quarter": "季卡",
    "com.bansheng.vip.year": "年卡",
    "com.bansheng.vip.trial": "体验会员",
}


def _field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def product_label(product_id: str) -> str:
    return _PRODUCT_LABELS.get(product_id) or product_id


def _is_vip_product(product_id: str) -> bool:
    product = APPLE_PRODUCTS.get(product_id)
    return product is not None and product.grants_vip


async def get_membership(user_id: str, *, history_limit: int = 50) -> dict[str, Any]:
    """Membership hub for store UI: VIP snapshot + subscription + VIP purchase history."""
    await grant.reconcile_vip_entitlements(user_id)
    vip = await wallet.full_wallet(user_id)

    sub_rows = await db.query_raw(
        """
        SELECT product_id, status, auto_renew_status, auto_renew_product_id,
               expires_date, grace_period_expires_date, updated_at
        FROM iap_subscription_state
        WHERE user_id = $1
        ORDER BY
            CASE status
                WHEN 'active' THEN 0
                WHEN 'in_grace' THEN 1
                ELSE 2
            END,
            updated_at DESC
        LIMIT 1
        """,
        user_id,
    )
    subscription = None
    if sub_rows:
        row = sub_rows[0]
        subscription = {
            "product_id": str(_field(row, "product_id", "")),
            "product_label": product_label(str(_field(row, "product_id", ""))),
            "status": str(_field(row, "status", "")),
            "auto_renew_enabled": bool(_field(row, "auto_renew_status")),
            "auto_renew_product_id": _field(row, "auto_renew_product_id"),
            "expires_date": _iso(_field(row, "expires_date")),
            "grace_period_expires_date": _iso(_field(row, "grace_period_expires_date")),
            "updated_at": _iso(_field(row, "updated_at")) or "",
        }

    limit = min(max(history_limit, 1), 100)
    history_rows = await db.query_raw(
        """
        SELECT transaction_id, product_id, kind, status, purchase_date, expires_date
        FROM iap_transactions
        WHERE user_id = $1
          AND product_id LIKE $2
          AND status IN ('granted', 'refunded', 'revoked')
        ORDER BY purchase_date DESC NULLS LAST, created_at DESC
        LIMIT $3
        """,
        user_id,
        f"{_VIP_PRODUCT_PREFIX}%",
        limit,
    )
    history = [
        {
            "transaction_id": str(_field(r, "transaction_id", "")),
            "product_id": str(_field(r, "product_id", "")),
            "product_label": product_label(str(_field(r, "product_id", ""))),
            "kind": str(_field(r, "kind", "")),
            "status": str(_field(r, "status", "")),
            "purchase_date": _iso(_field(r, "purchase_date")),
            "expires_date": _iso(_field(r, "expires_date")),
        }
        for r in history_rows
        if _is_vip_product(str(_field(r, "product_id", "")))
    ]

    auto_renew_active = (
        subscription is not None
        and subscription["status"] in ("active", "in_grace")
        and subscription["auto_renew_enabled"]
    )

    return {
        "vip": vip,
        "subscription": subscription,
        "auto_renew_active": auto_renew_active,
        "history": history,
    }
