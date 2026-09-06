from __future__ import annotations

import json
from typing import Any

from app.db import db
from app.services import wallet
from app.services.store_catalog import (
    EXCHANGE_PRODUCTS,
    MAKEUP_CARD_KIND,
    MUSIC_COUPON_KIND,
    catalog_payload,
)

_BATCH_KINDS: tuple[str, ...] = (MUSIC_COUPON_KIND, MAKEUP_CARD_KIND)


def _field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    text = str(value)
    return text or None


def _inventory_row(row: Any) -> dict[str, Any]:
    return {
        "product_kind": str(_field(row, "product_kind", "")),
        "quantity": int(_field(row, "quantity", 0) or 0),
        "acquired_at": _iso(_field(row, "acquired_at")),
        "updated_at": _iso(_field(row, "updated_at")),
        "expires_at": None,
        "is_gift": False,
    }


def _wallet_row(row: Any) -> dict[str, int]:
    return wallet.wallet_balances(row)


async def add_inventory(
    user_id: str,
    product_kind: str,
    *,
    quantity: int = 1,
    client: Any,
) -> dict[str, Any]:
    if quantity <= 0:
        raise ValueError("invalid_amount")
    inventory_rows = await client.query_raw(
        """
        INSERT INTO user_store_inventory
            (user_id, product_kind, quantity, acquired_at, updated_at)
        VALUES ($1, $2, $3, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT (user_id, product_kind) DO UPDATE
        SET quantity = user_store_inventory.quantity + $3,
            updated_at = CURRENT_TIMESTAMP
        RETURNING product_kind, quantity, acquired_at, updated_at
        """,
        user_id,
        product_kind,
        quantity,
    )
    return _inventory_row(inventory_rows[0])


async def list_inventory(user_id: str) -> dict[str, list[dict[str, Any]]]:
    rows = await db.query_raw(
        """
        SELECT product_kind, quantity, acquired_at, updated_at
        FROM user_store_inventory
        WHERE user_id = $1
          AND quantity > 0
        ORDER BY updated_at DESC, product_kind ASC
        """,
        user_id,
    )
    items = [_inventory_row(row) for row in rows]
    unbound_rows = await db.query_raw(
        """
        SELECT metadata->>'product_kind' AS product_kind,
               COUNT(*)::int AS n
        FROM user_offerings
        WHERE user_id = $1
          AND kind = 'gift'
          AND message_id IS NULL
          AND status = 'sent'
          AND COALESCE(metadata->>'product_kind', '') <> ''
        GROUP BY metadata->>'product_kind'
        """,
        user_id,
    )
    extra = {
        str(_field(row, "product_kind", "") or ""): int(_field(row, "n", 0) or 0)
        for row in unbound_rows
        if str(_field(row, "product_kind", "") or "")
    }
    if extra:
        merged: list[dict[str, Any]] = []
        for item in items:
            kind = str(item.get("product_kind") or "")
            bump = extra.pop(kind, 0)
            if bump:
                item = {**item, "quantity": int(item.get("quantity") or 0) + bump}
            merged.append(item)
        for kind, bump in extra.items():
            merged.append({
                "product_kind": kind,
                "quantity": bump,
                "acquired_at": None,
                "updated_at": None,
                "expires_at": None,
                "is_gift": False,
            })
        items = merged

    # 音乐畅听券/补签卡自 20260824 迁移到批次表（带过期），单独聚合合并展示。
    for kind in _BATCH_KINDS:
        summary = await batch_summary(user_id, kind)
        if summary["quantity"] > 0:
            items.append({
                "product_kind": kind,
                "quantity": summary["quantity"],
                "acquired_at": None,
                "updated_at": None,
                "expires_at": summary["earliest_expires_at"],
                "is_gift": summary["is_gift"],
            })
    return {"items": items}


async def get_catalog(user_id: str) -> dict[str, Any]:
    await wallet.ensure_wallet(user_id)
    rows = await db.query_raw(
        """
        SELECT vip_until, vip_trial_used
        FROM user_wallets
        WHERE user_id = $1
        """,
        user_id,
    )
    row = rows[0] if rows else {}
    return catalog_payload(
        is_vip=wallet.is_vip_from_row(row),
        vip_trial_available=wallet.vip_trial_available_from_row(row),
    )


async def exchange_product(user_id: str, product_kind: str) -> dict[str, Any]:
    product = EXCHANGE_PRODUCTS.get(product_kind)
    if product is None:
        raise ValueError("unknown_product")

    await wallet.ensure_wallet(user_id)
    async with db.tx() as tx:
        status_rows = await tx.query_raw(
            """
            SELECT point_balance, vip_until, vip_trial_used
            FROM user_wallets
            WHERE user_id = $1
            FOR UPDATE
            """,
            user_id,
        )
        is_vip = wallet.is_vip_from_row(status_rows[0] if status_rows else {})
        price = product.price_for(is_vip)
        wallet_rows = await tx.query_raw(
            """
            UPDATE user_wallets
            SET point_balance = point_balance - $2,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = $1
              AND point_balance >= $2
            RETURNING ticket_balance, point_balance, achievement_points_synced
            """,
            user_id,
            price,
        )
        if not wallet_rows:
            raise ValueError("insufficient_point_balance")
        balance = _wallet_row(wallet_rows[0])
        inventory_item = await add_inventory(
            user_id, product_kind, quantity=1, client=tx
        )
        await tx.execute_raw(
            """
            INSERT INTO wallet_ledger
                (user_id, currency, delta, balance_after, source, source_id, metadata)
            VALUES ($1, 'point', $2, $3, 'store_exchange', $4, $5::jsonb)
            """,
            user_id,
            -price,
            balance["point_balance"],
            product_kind,
            json.dumps(
                {
                    "product_kind": product_kind,
                    "price": price,
                    "member_price": product.member_price,
                    "list_price": product.list_price,
                    "is_vip": is_vip,
                },
                ensure_ascii=False,
            ),
        )

    return {
        "wallet": balance,
        "inventory_item": inventory_item,
    }


async def add_batch(
    user_id: str,
    product_kind: str,
    *,
    quantity: int,
    source: str,
    expires_at: Any = None,
    client: Any,
) -> dict[str, Any]:
    """Grant a new expiring/permanent batch of a consumable (音乐畅听券/补签卡).

    Unlike ``add_inventory`` (which merges into one row per product_kind), each
    grant is its own row so distinct expiry dates can coexist and be drained
    earliest-first — see :func:`consume_batch_units`.
    """
    if quantity <= 0:
        raise ValueError("invalid_amount")
    rows = await client.query_raw(
        """
        INSERT INTO user_consumable_batch
            (user_id, product_kind, quantity, source, expires_at)
        VALUES ($1, $2, $3, $4, $5::timestamp)
        RETURNING id, product_kind, quantity, source, expires_at, created_at
        """,
        user_id,
        product_kind,
        quantity,
        source,
        expires_at,
    )
    return _batch_row(rows[0])


async def consume_batch_units(
    user_id: str,
    product_kind: str,
    units: int,
    *,
    client: Any,
) -> int:
    """Drain ``units`` from unexpired batches, earliest-``expires_at`` first
    (NULL/permanent batches drained last). Returns units actually consumed;
    raises ``insufficient_inventory`` if the unexpired total is short.
    """
    if units <= 0:
        raise ValueError("invalid_amount")
    locked = await client.query_raw(
        """
        SELECT id, quantity
        FROM user_consumable_batch
        WHERE user_id = $1
          AND product_kind = $2
          AND quantity > 0
          AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
        ORDER BY expires_at IS NULL, expires_at ASC, created_at ASC
        FOR UPDATE
        """,
        user_id,
        product_kind,
    )
    available = sum(int(_field(row, "quantity", 0) or 0) for row in locked)
    if available < units:
        raise ValueError("insufficient_inventory")

    remaining = units
    for row in locked:
        if remaining <= 0:
            break
        row_id = str(_field(row, "id", ""))
        take = min(remaining, int(_field(row, "quantity", 0) or 0))
        if take <= 0:
            continue
        await client.execute_raw(
            "UPDATE user_consumable_batch SET quantity = quantity - $2 WHERE id = $1",
            row_id,
            take,
        )
        remaining -= take
    return units


async def batch_summary(user_id: str, product_kind: str) -> dict[str, Any]:
    """Aggregate unexpired batches for backpack display: total quantity, the
    soonest expiry among them, and whether any of that quantity is a VIP gift.
    """
    rows = await db.query_raw(
        """
        SELECT quantity, source, expires_at
        FROM user_consumable_batch
        WHERE user_id = $1
          AND product_kind = $2
          AND quantity > 0
          AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
        ORDER BY expires_at IS NULL, expires_at ASC
        """,
        user_id,
        product_kind,
    )
    total = sum(int(_field(row, "quantity", 0) or 0) for row in rows)
    earliest = _iso(_field(rows[0], "expires_at")) if rows else None
    has_gift = any(str(_field(row, "source", "")) == "vip_grant" for row in rows)
    return {
        "product_kind": product_kind,
        "quantity": total,
        "earliest_expires_at": earliest,
        "is_gift": has_gift,
    }


def _batch_row(row: Any) -> dict[str, Any]:
    return {
        "id": str(_field(row, "id", "")),
        "product_kind": str(_field(row, "product_kind", "")),
        "quantity": int(_field(row, "quantity", 0) or 0),
        "source": str(_field(row, "source", "")),
        "expires_at": _iso(_field(row, "expires_at")),
        "created_at": _iso(_field(row, "created_at")),
    }


async def consume_inventory(
    user_id: str,
    product_kind: str,
    *,
    quantity: int = 1,
    client: Any,
) -> dict[str, Any]:
    """Atomically decrement owned inventory. Raises if the stack is too small."""
    if quantity <= 0:
        raise ValueError("invalid_amount")
    rows = await client.query_raw(
        """
        UPDATE user_store_inventory
        SET quantity = quantity - $3,
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = $1
          AND product_kind = $2
          AND quantity >= $3
        RETURNING product_kind, quantity, acquired_at, updated_at
        """,
        user_id,
        product_kind,
        quantity,
    )
    if not rows:
        raise ValueError("insufficient_inventory")
    return _inventory_row(rows[0])
