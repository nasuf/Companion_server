from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.db import db
from app.services import wallet


@dataclass(frozen=True)
class StoreProduct:
    product_kind: str
    price: int


EXCHANGE_PRODUCTS: dict[str, StoreProduct] = {
    "tea": StoreProduct("tea", 99),
    "cake": StoreProduct("cake", 99),
    "coffee": StoreProduct("coffee", 288),
    "cola": StoreProduct("cola", 512),
    "flower": StoreProduct("flower", 1314),
    "plush": StoreProduct("plush", 9999),
    "capsuleSkin": StoreProduct("capsuleSkin", 188),
    "chatFrame": StoreProduct("chatFrame", 388),
    "bubble": StoreProduct("bubble", 288),
    "backdrop": StoreProduct("backdrop", 588),
    "theme": StoreProduct("theme", 888),
    "stationery": StoreProduct("stationery", 688),
    "checkinSkin": StoreProduct("checkinSkin", 1888),
    "signCard": StoreProduct("signCard", 100),
    "musicCoupon": StoreProduct("musicCoupon", 1888),
    "gameCoupon": StoreProduct("gameCoupon", 1888),
    "movieCoupon": StoreProduct("movieCoupon", 1888),
}


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
    }


def _wallet_row(row: Any) -> dict[str, int]:
    return {
        "ticket_balance": int(_field(row, "ticket_balance", 0) or 0),
        "point_balance": int(_field(row, "point_balance", 0) or 0),
        "achievement_points_synced": int(
            _field(row, "achievement_points_synced", 0) or 0
        ),
    }


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
    return {"items": [_inventory_row(row) for row in rows]}


async def exchange_product(user_id: str, product_kind: str) -> dict[str, Any]:
    product = EXCHANGE_PRODUCTS.get(product_kind)
    if product is None:
        raise ValueError("unknown_product")

    await wallet.ensure_wallet(user_id)
    async with db.tx() as tx:
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
            product.price,
        )
        if not wallet_rows:
            raise ValueError("insufficient_point_balance")
        balance = _wallet_row(wallet_rows[0])

        inventory_rows = await tx.query_raw(
            """
            INSERT INTO user_store_inventory
                (user_id, product_kind, quantity, acquired_at, updated_at)
            VALUES ($1, $2, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id, product_kind) DO UPDATE
            SET quantity = user_store_inventory.quantity + 1,
                updated_at = CURRENT_TIMESTAMP
            RETURNING product_kind, quantity, acquired_at, updated_at
            """,
            user_id,
            product_kind,
        )
        await tx.execute_raw(
            """
            INSERT INTO wallet_ledger
                (user_id, currency, delta, balance_after, source, source_id, metadata)
            VALUES ($1, 'point', $2, $3, 'store_exchange', $4, $5::jsonb)
            """,
            user_id,
            -product.price,
            balance["point_balance"],
            product_kind,
            json.dumps(
                {"product_kind": product_kind, "price": product.price},
                ensure_ascii=False,
            ),
        )

    return {
        "wallet": balance,
        "inventory_item": _inventory_row(inventory_rows[0]),
    }
