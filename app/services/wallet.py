from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from app.db import db
from app.services.achievements.mode import achievement_display_enabled
from app.services.achievements.service import list_achievements

TICKET_TO_POINTS_RATE = 10
# Same ceiling as user-facing red-packet sends; one admin action cannot exceed it.
MAX_TICKET_ADJUST = 1_000_000
MAX_POINT_ADJUST = 1_000_000

# Ledger source for manual admin ticket adjustments. The ledger row itself is
# the audit record — its metadata carries the acting admin id and the note.
SOURCE_ADMIN_GRANT = "admin_grant"


def _field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _json(row_value: Any) -> dict[str, Any]:
    if isinstance(row_value, dict):
        return row_value
    if isinstance(row_value, str) and row_value:
        try:
            parsed = json.loads(row_value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _as_aware_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def is_vip_from_row(row: Any) -> bool:
    until = _as_aware_dt(_field(row, "vip_until"))
    if until is None:
        return False
    return until > datetime.now(timezone.utc)


def vip_trial_available_from_row(row: Any) -> bool:
    if bool(_field(row, "vip_trial_used", False)):
        return False
    return not is_vip_from_row(row)


async def is_vip(user_id: str, *, client: Any | None = None) -> bool:
    """Read-only VIP check, usable inside an existing transaction via ``client``."""
    executor = client or db
    rows = await executor.query_raw(
        "SELECT vip_until FROM user_wallets WHERE user_id = $1",
        user_id,
    )
    if not rows:
        return False
    return is_vip_from_row(rows[0])


def wallet_balances(row: Any) -> dict[str, int]:
    return {
        "ticket_balance": int(_field(row, "ticket_balance", 0) or 0),
        "point_balance": int(_field(row, "point_balance", 0) or 0),
        "achievement_points_synced": int(
            _field(row, "achievement_points_synced", 0) or 0
        ),
    }


async def debit_tickets(
    user_id: str,
    amount: int,
    *,
    source: str,
    source_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    client: Any,
) -> dict[str, int]:
    """Spend shop tickets inside an existing transaction."""
    if amount <= 0:
        raise ValueError("invalid_amount")
    rows = await client.query_raw(
        """
        UPDATE user_wallets
        SET ticket_balance = ticket_balance - $2,
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = $1 AND ticket_balance >= $2
        RETURNING ticket_balance, point_balance, achievement_points_synced
        """,
        user_id,
        amount,
    )
    if not rows:
        raise ValueError("insufficient_ticket_balance")
    balance = wallet_balances(rows[0])
    await _record_ledger(
        user_id=user_id,
        currency="ticket",
        delta=-amount,
        balance_after=balance["ticket_balance"],
        source=source,
        source_id=source_id,
        metadata=metadata,
        client=client,
    )
    return balance


async def credit_tickets(
    user_id: str,
    amount: int,
    *,
    source: str,
    source_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    client: Any,
) -> dict[str, int]:
    """Add shop tickets inside an existing transaction (mirror of debit_tickets).

    The wallet row must already exist (call ensure_wallet first). Writes both the
    balance update and the ledger row on the same client so the audit trail can
    never diverge from the balance.
    """
    if amount <= 0:
        raise ValueError("invalid_amount")
    rows = await client.query_raw(
        """
        UPDATE user_wallets
        SET ticket_balance = ticket_balance + $2,
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = $1
        RETURNING ticket_balance, point_balance, achievement_points_synced
        """,
        user_id,
        amount,
    )
    if not rows:
        raise ValueError("wallet_not_found")
    balance = wallet_balances(rows[0])
    await _record_ledger(
        user_id=user_id,
        currency="ticket",
        delta=amount,
        balance_after=balance["ticket_balance"],
        source=source,
        source_id=source_id,
        metadata=metadata,
        client=client,
    )
    return balance


async def debit_tickets_prioritized(
    user_id: str,
    amount: int,
    *,
    source: str,
    source_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    client: Any,
) -> dict[str, int]:
    """Spend tickets, draining gift (limited) balance before permanent balance.

    Gift tickets expire on VIP lapse, so they must be consumed first. Runs inside
    an existing transaction with a row lock (the split needs both balances read
    first). Writes one ledger row per bucket actually touched so the audit trail
    reconciles per currency.
    """
    if amount <= 0:
        raise ValueError("invalid_amount")
    locked = await client.query_raw(
        """
        SELECT gift_ticket_balance, ticket_balance
        FROM user_wallets
        WHERE user_id = $1
        FOR UPDATE
        """,
        user_id,
    )
    if not locked:
        raise ValueError("wallet_not_found")
    gift = int(_field(locked[0], "gift_ticket_balance", 0) or 0)
    perm = int(_field(locked[0], "ticket_balance", 0) or 0)
    if gift + perm < amount:
        raise ValueError("insufficient_ticket_balance")
    from_gift = min(gift, amount)
    from_perm = amount - from_gift
    rows = await client.query_raw(
        """
        UPDATE user_wallets
        SET gift_ticket_balance = gift_ticket_balance - $2,
            ticket_balance = ticket_balance - $3,
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = $1
        RETURNING gift_ticket_balance, ticket_balance, point_balance,
                  achievement_points_synced
        """,
        user_id,
        from_gift,
        from_perm,
    )
    balance = _spendable_balances(rows[0])
    if from_gift > 0:
        await _record_ledger(
            user_id=user_id,
            currency="gift_ticket",
            delta=-from_gift,
            balance_after=balance["gift_ticket_balance"],
            source=source,
            source_id=source_id,
            metadata=metadata,
            client=client,
        )
    if from_perm > 0:
        await _record_ledger(
            user_id=user_id,
            currency="ticket",
            delta=-from_perm,
            balance_after=balance["ticket_balance"],
            source=source,
            source_id=source_id,
            metadata=metadata,
            client=client,
        )
    return balance


async def credit_gift_tickets(
    user_id: str,
    amount: int,
    *,
    source: str,
    source_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    client: Any,
) -> dict[str, int]:
    """Add limited (gift) tickets inside an existing transaction (VIP monthly grant)."""
    if amount <= 0:
        raise ValueError("invalid_amount")
    rows = await client.query_raw(
        """
        UPDATE user_wallets
        SET gift_ticket_balance = gift_ticket_balance + $2,
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = $1
        RETURNING gift_ticket_balance, ticket_balance, point_balance,
                  achievement_points_synced
        """,
        user_id,
        amount,
    )
    if not rows:
        raise ValueError("wallet_not_found")
    balance = _spendable_balances(rows[0])
    await _record_ledger(
        user_id=user_id,
        currency="gift_ticket",
        delta=amount,
        balance_after=balance["gift_ticket_balance"],
        source=source,
        source_id=source_id,
        metadata=metadata,
        client=client,
    )
    return balance


async def zero_gift_tickets(
    user_id: str,
    *,
    source: str,
    metadata: dict[str, Any] | None = None,
    client: Any,
) -> dict[str, int]:
    """Clear all limited (gift) tickets on VIP lapse. No ledger row if already 0."""
    locked = await client.query_raw(
        """
        SELECT gift_ticket_balance, ticket_balance, point_balance,
               achievement_points_synced
        FROM user_wallets
        WHERE user_id = $1
        FOR UPDATE
        """,
        user_id,
    )
    if not locked:
        raise ValueError("wallet_not_found")
    cleared = int(_field(locked[0], "gift_ticket_balance", 0) or 0)
    if cleared <= 0:
        return _spendable_balances(locked[0])
    rows = await client.query_raw(
        """
        UPDATE user_wallets
        SET gift_ticket_balance = 0,
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = $1
        RETURNING gift_ticket_balance, ticket_balance, point_balance,
                  achievement_points_synced
        """,
        user_id,
    )
    balance = _spendable_balances(rows[0])
    await _record_ledger(
        user_id=user_id,
        currency="gift_ticket",
        delta=-cleared,
        balance_after=0,
        source=source,
        metadata=metadata,
        client=client,
    )
    return balance


def _spendable_balances(row: Any) -> dict[str, int]:
    """Balance dict including the gift-ticket bucket (rows must SELECT it)."""
    return {
        "gift_ticket_balance": int(_field(row, "gift_ticket_balance", 0) or 0),
        "ticket_balance": int(_field(row, "ticket_balance", 0) or 0),
        "point_balance": int(_field(row, "point_balance", 0) or 0),
        "achievement_points_synced": int(
            _field(row, "achievement_points_synced", 0) or 0
        ),
    }


async def full_wallet(user_id: str, *, client: Any | None = None) -> dict[str, Any]:
    """Complete wallet snapshot for display (balances + VIP state + accrual).

    Pass ``client`` (an open ``db.tx()``) when called from inside a
    transaction that already holds a lock on this user's row — otherwise
    this borrows a second connection from the pool while the caller's
    transaction still holds the first, risking pool starvation under the
    project's deliberately small connection limit (see ``app/db.py``).
    """
    executor = client or db
    await ensure_wallet(user_id, client=client)
    rows = await executor.query_raw(
        """
        SELECT gift_ticket_balance, ticket_balance, point_balance,
               achievement_points_synced, overage_accrued,
               vip_until, vip_trial_used, vip_last_grant_at
        FROM user_wallets
        WHERE user_id = $1
        """,
        user_id,
    )
    row = rows[0] if rows else {}
    balance = _spendable_balances(row)
    return {
        **balance,
        "spendable_tickets": balance["gift_ticket_balance"] + balance["ticket_balance"],
        "is_vip": is_vip_from_row(row),
        "vip_until": _iso(_field(row, "vip_until")) or None,
        "vip_trial_available": vip_trial_available_from_row(row),
        "vip_last_grant_at": _iso(_field(row, "vip_last_grant_at")) or None,
    }


async def admin_adjust_tickets(
    user_id: str,
    amount: int,
    *,
    admin_id: str,
    note: str | None = None,
) -> dict[str, int]:
    """Manual admin ticket grant/adjustment (positive adds, negative deducts).

    Runs in its own transaction with a row lock so concurrent adjustments cannot
    lose an update. The balance is floored at 0 — a negative adjustment can never
    push it below zero, and the ledger records the *applied* delta (post-floor),
    not the requested one, so the running balance always reconciles.

    The ledger row is the audit record: source=admin_grant, metadata carries the
    acting admin id, the requested amount, and the free-text note.
    """
    if amount == 0 or abs(amount) > MAX_TICKET_ADJUST:
        raise ValueError("invalid_amount")
    await _ensure_user_exists(user_id)
    await ensure_wallet(user_id)
    async with db.tx() as tx:
        locked = await tx.query_raw(
            """
            SELECT ticket_balance, point_balance, achievement_points_synced
            FROM user_wallets
            WHERE user_id = $1
            FOR UPDATE
            """,
            user_id,
        )
        if not locked:
            raise ValueError("wallet_not_found")
        current = int(_field(locked[0], "ticket_balance", 0) or 0)
        new_balance = max(0, current + amount)
        applied = new_balance - current
        if applied == 0:
            # Deducting from an empty wallet is a no-op; surface it so the caller
            # doesn't report a phantom success.
            raise ValueError("no_change")
        rows = await tx.query_raw(
            """
            UPDATE user_wallets
            SET ticket_balance = $2,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = $1
            RETURNING ticket_balance, point_balance, achievement_points_synced
            """,
            user_id,
            new_balance,
        )
        balance = wallet_balances(rows[0])
        await _record_ledger(
            user_id=user_id,
            currency="ticket",
            delta=applied,
            balance_after=balance["ticket_balance"],
            source=SOURCE_ADMIN_GRANT,
            metadata={
                "requested": amount,
                "applied": applied,
                "admin_id": admin_id,
                "note": (note or "").strip(),
            },
            client=tx,
        )
    return {"user_id": user_id, "delta": applied, **balance}


async def admin_adjust_points(
    user_id: str,
    amount: int,
    *,
    admin_id: str,
    note: str | None = None,
) -> dict[str, int]:
    """Manual admin shop-point grant/adjustment (positive adds, negative deducts).

    Mirrors ``admin_adjust_tickets`` but writes ``currency='point'`` ledger rows.
    """
    if amount == 0 or abs(amount) > MAX_POINT_ADJUST:
        raise ValueError("invalid_amount")
    await _ensure_user_exists(user_id)
    await ensure_wallet(user_id)
    async with db.tx() as tx:
        locked = await tx.query_raw(
            """
            SELECT ticket_balance, point_balance, achievement_points_synced
            FROM user_wallets
            WHERE user_id = $1
            FOR UPDATE
            """,
            user_id,
        )
        if not locked:
            raise ValueError("wallet_not_found")
        current = int(_field(locked[0], "point_balance", 0) or 0)
        new_balance = max(0, current + amount)
        applied = new_balance - current
        if applied == 0:
            raise ValueError("no_change")
        rows = await tx.query_raw(
            """
            UPDATE user_wallets
            SET point_balance = $2,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = $1
            RETURNING ticket_balance, point_balance, achievement_points_synced
            """,
            user_id,
            new_balance,
        )
        balance = wallet_balances(rows[0])
        await _record_ledger(
            user_id=user_id,
            currency="point",
            delta=applied,
            balance_after=balance["point_balance"],
            source=SOURCE_ADMIN_GRANT,
            metadata={
                "requested": amount,
                "applied": applied,
                "admin_id": admin_id,
                "note": (note or "").strip(),
            },
            client=tx,
        )
    return {"user_id": user_id, "delta": applied, **balance}


async def _ensure_user_exists(user_id: str) -> None:
    rows = await db.query_raw(
        "SELECT 1 FROM users WHERE id = $1 LIMIT 1",
        user_id,
    )
    if not rows:
        raise ValueError("user_not_found")


async def ensure_wallet(
    user_id: str, *, client: Any | None = None
) -> dict[str, int]:
    executor = client or db
    rows = await executor.query_raw(
        """
        INSERT INTO user_wallets (user_id)
        VALUES ($1)
        ON CONFLICT (user_id) DO UPDATE
        SET updated_at = user_wallets.updated_at
        RETURNING ticket_balance, point_balance, achievement_points_synced
        """,
        user_id,
    )
    row = rows[0]
    return {
        "ticket_balance": int(_field(row, "ticket_balance", 0) or 0),
        "point_balance": int(_field(row, "point_balance", 0) or 0),
        "achievement_points_synced": int(
            _field(row, "achievement_points_synced", 0) or 0
        ),
    }


async def sync_achievement_points(user_id: str, agent_id: str) -> dict[str, int]:
    # Silent mode keeps the achievement page (and its points) fully usable, so
    # point sync stays on; only "off" skips it. The ledger-delta logic below is
    # cumulative, so the first sync after re-enabling credits any gap at once.
    if not await achievement_display_enabled():
        return await ensure_wallet(user_id)
    await ensure_wallet(user_id)
    achievements = await list_achievements(user_id=user_id, agent_id=agent_id)
    total = int(achievements.get("score") or 0)
    async with db.tx() as tx:
        locked_rows = await tx.query_raw(
            """
            SELECT ticket_balance, point_balance, achievement_points_synced
            FROM user_wallets
            WHERE user_id = $1
            FOR UPDATE
            """,
            user_id,
        )
        locked = locked_rows[0]
        wallet = {
            "ticket_balance": int(_field(locked, "ticket_balance", 0) or 0),
            "point_balance": int(_field(locked, "point_balance", 0) or 0),
            "achievement_points_synced": int(
                _field(locked, "achievement_points_synced", 0) or 0
            ),
        }
        synced_rows = await tx.query_raw(
            """
            SELECT COALESCE(SUM(delta), 0) AS synced
            FROM wallet_ledger
            WHERE user_id = $1
              AND currency = 'point'
              AND source = 'achievement_sync'
              AND source_id = $2
            """,
            user_id,
            agent_id,
        )
        already_synced = int(_field(synced_rows[0], "synced", 0) or 0) if synced_rows else 0
        delta = max(0, total - already_synced)
        if delta <= 0:
            return wallet

        rows = await tx.query_raw(
            """
            UPDATE user_wallets
            SET point_balance = point_balance + $2,
                achievement_points_synced = achievement_points_synced + $2,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = $1
            RETURNING point_balance, achievement_points_synced
            """,
            user_id,
            delta,
        )
        balance_after = int(_field(rows[0], "point_balance", 0) or 0)
        await _record_ledger(
            user_id=user_id,
            currency="point",
            delta=delta,
            balance_after=balance_after,
            source="achievement_sync",
            source_id=agent_id,
            metadata={"achievement_score": total},
            client=tx,
        )
        wallet["point_balance"] = balance_after
        wallet["achievement_points_synced"] = int(
            _field(rows[0], "achievement_points_synced", 0) or 0
        )
        return wallet


async def get_balance(user_id: str, *, agent_id: str | None = None) -> dict[str, int]:
    if agent_id:
        return await sync_achievement_points(user_id, agent_id)
    return await ensure_wallet(user_id)


async def exchange_ticket_to_points(user_id: str, ticket_amount: int) -> dict[str, int]:
    if ticket_amount <= 0:
        raise ValueError("invalid_amount")
    point_delta = ticket_amount * TICKET_TO_POINTS_RATE
    await ensure_wallet(user_id)
    rows = await db.query_raw(
        """
        UPDATE user_wallets
        SET ticket_balance = ticket_balance - $2,
            point_balance = point_balance + $3,
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = $1 AND ticket_balance >= $2
        RETURNING ticket_balance, point_balance
        """,
        user_id,
        ticket_amount,
        point_delta,
    )
    if not rows:
        raise ValueError("insufficient_ticket_balance")

    row = rows[0]
    ticket_balance = int(_field(row, "ticket_balance", 0) or 0)
    point_balance = int(_field(row, "point_balance", 0) or 0)
    await _record_ledger(
        user_id=user_id,
        currency="ticket",
        delta=-ticket_amount,
        balance_after=ticket_balance,
        source="ticket_to_point_exchange",
        metadata={"point_delta": point_delta},
    )
    await _record_ledger(
        user_id=user_id,
        currency="point",
        delta=point_delta,
        balance_after=point_balance,
        source="ticket_to_point_exchange",
        metadata={"ticket_delta": -ticket_amount},
    )
    return {
        "ticket_balance": ticket_balance,
        "point_balance": point_balance,
        "achievement_points_synced": (
            await ensure_wallet(user_id)
        )["achievement_points_synced"],
    }


async def list_ledger(
    user_id: str,
    *,
    currency: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    limit = min(max(limit, 1), 200)
    offset = max(offset, 0)
    if currency:
        rows = await db.query_raw(
            """
            SELECT id, currency, delta, balance_after, source, source_id, metadata, created_at
            FROM wallet_ledger
            WHERE user_id = $1 AND currency = $2
            ORDER BY created_at DESC
            LIMIT $3 OFFSET $4
            """,
            user_id,
            currency,
            limit,
            offset,
        )
    else:
        rows = await db.query_raw(
            """
            SELECT id, currency, delta, balance_after, source, source_id, metadata, created_at
            FROM wallet_ledger
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
            """,
            user_id,
            limit,
            offset,
        )
    return [_ledger_row(row) for row in rows]


async def _record_ledger(
    *,
    user_id: str,
    currency: str,
    delta: int,
    balance_after: int,
    source: str,
    source_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    client: Any | None = None,
) -> None:
    executor = client or db
    await executor.execute_raw(
        """
        INSERT INTO wallet_ledger
            (user_id, currency, delta, balance_after, source, source_id, metadata)
        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
        """,
        user_id,
        currency,
        delta,
        balance_after,
        source,
        source_id,
        json.dumps(metadata or {}, ensure_ascii=False),
    )


def _ledger_row(row: Any) -> dict[str, Any]:
    created_at = _field(row, "created_at")
    return {
        "id": str(_field(row, "id", "")),
        "currency": str(_field(row, "currency", "")),
        "delta": int(_field(row, "delta", 0) or 0),
        "balance_after": int(_field(row, "balance_after", 0) or 0),
        "source": str(_field(row, "source", "")),
        "source_id": _field(row, "source_id"),
        "metadata": _json(_field(row, "metadata")),
        "created_at": created_at.isoformat()
        if hasattr(created_at, "isoformat")
        else str(created_at or ""),
    }


def _iso(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value or "")


async def list_admin_balances(
    *,
    search: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """All users' ticket balances for the admin console (paginated + searchable).

    LEFT JOINs the wallet so a user who never opened one still shows a 0 balance.
    Hides the template system user like the other admin listings do.
    """
    from app.services.agent_template.registry import TEMPLATE_SYSTEM_USERNAME

    limit = min(max(limit, 1), 200)
    offset = max(offset, 0)
    term = (search or "").strip()
    pattern = f"%{term}%"
    where = "WHERE u.username <> $1"
    params: list[Any] = [TEMPLATE_SYSTEM_USERNAME]
    if term:
        where += (
            " AND (u.username ILIKE $2 OR u.id ILIKE $2"
            " OR u.display_name ILIKE $2"
            " OR EXISTS (SELECT 1 FROM auth_identities ai"
            " WHERE ai.user_id = u.id AND ai.provider = 'wechat'"
            " AND ai.raw_profile->>'nickname' ILIKE $2))"
        )
        params.append(pattern)

    count_rows = await db.query_raw(
        f"SELECT COUNT(*) AS n FROM users u {where}",
        *params,
    )
    total = int(_field(count_rows[0], "n", 0) or 0) if count_rows else 0

    rows = await db.query_raw(
        f"""
        SELECT u.id, u.username, u.display_name,
               COALESCE(w.ticket_balance, 0) AS ticket_balance,
               COALESCE(w.point_balance, 0) AS point_balance,
               COALESCE(w.gift_ticket_balance, 0) AS gift_ticket_balance,
               w.vip_until,
               w.updated_at,
               (
                   SELECT ai.raw_profile->>'nickname'
                   FROM auth_identities ai
                   WHERE ai.user_id = u.id AND ai.provider = 'wechat'
                   ORDER BY ai.updated_at DESC
                   LIMIT 1
               ) AS nickname
        FROM users u
        LEFT JOIN user_wallets w ON w.user_id = u.id
        {where}
        ORDER BY COALESCE(w.ticket_balance, 0) DESC, u.created_at DESC
        LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
        """,
        *params,
        limit,
        offset,
    )
    items = [
        {
            "user_id": str(_field(row, "id", "")),
            "username": str(_field(row, "username", "") or ""),
            "display_name": _field(row, "display_name"),
            "nickname": _field(row, "nickname"),
            "ticket_balance": int(_field(row, "ticket_balance", 0) or 0),
            "point_balance": int(_field(row, "point_balance", 0) or 0),
            "gift_ticket_balance": int(_field(row, "gift_ticket_balance", 0) or 0),
            "is_vip": is_vip_from_row(row),
            "vip_until": _iso(_field(row, "vip_until")) or None,
            "updated_at": _iso(_field(row, "updated_at")) or None,
        }
        for row in rows
    ]
    return {"items": items, "total": total}


async def admin_set_vip_until(
    user_id: str,
    vip_until: Any,
) -> dict[str, Any]:
    """Directly set (or clear, with ``None``) a user's VIP expiry.

    No ledger row: VIP isn't a currency balance, and the wallet row itself is
    the record. The caller (API layer) is responsible for logging admin_id/
    note, mirroring how admin_adjust_tickets/points log via the standard
    logger in addition to their ledger row.
    """
    await _ensure_user_exists(user_id)
    await ensure_wallet(user_id)
    rows = await db.query_raw(
        """
        UPDATE user_wallets
        SET vip_until = $2,
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = $1
        RETURNING vip_until
        """,
        user_id,
        vip_until,
    )
    row = rows[0]
    return {
        "user_id": user_id,
        "is_vip": is_vip_from_row(row),
        "vip_until": _iso(_field(row, "vip_until")) or None,
    }


async def list_admin_gift_ticket_ledger(
    *,
    user_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Gift(限时)-ticket-currency ledger for the VIP admin audit view."""
    limit = min(max(limit, 1), 200)
    offset = max(offset, 0)
    where = "WHERE l.currency = 'gift_ticket'"
    params: list[Any] = [limit, offset]
    if user_id:
        where += " AND l.user_id = $3"
        params.append(user_id)
    rows = await db.query_raw(
        f"""
        SELECT l.id, l.user_id, u.username, u.display_name,
               l.currency, l.delta, l.balance_after, l.source, l.source_id,
               l.metadata, l.created_at,
               (
                   SELECT ai.raw_profile->>'nickname'
                   FROM auth_identities ai
                   WHERE ai.user_id = l.user_id AND ai.provider = 'wechat'
                   ORDER BY ai.updated_at DESC
                   LIMIT 1
               ) AS nickname
        FROM wallet_ledger l
        LEFT JOIN users u ON u.id = l.user_id
        {where}
        ORDER BY l.created_at DESC, l.id DESC
        LIMIT $1 OFFSET $2
        """,
        *params,
    )
    return [
        {
            "id": str(_field(row, "id", "")),
            "user_id": str(_field(row, "user_id", "")),
            "username": _field(row, "username"),
            "display_name": _field(row, "display_name"),
            "nickname": _field(row, "nickname"),
            "currency": str(_field(row, "currency", "")),
            "delta": int(_field(row, "delta", 0) or 0),
            "balance_after": int(_field(row, "balance_after", 0) or 0),
            "source": str(_field(row, "source", "")),
            "source_id": _field(row, "source_id"),
            "metadata": _json(_field(row, "metadata")),
            "created_at": _iso(_field(row, "created_at")),
        }
        for row in rows
    ]


async def list_admin_ticket_ledger(
    *,
    user_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Ticket-currency ledger across all users for the admin audit view."""
    limit = min(max(limit, 1), 200)
    offset = max(offset, 0)
    where = "WHERE l.currency = 'ticket'"
    params: list[Any] = [limit, offset]
    if user_id:
        where += " AND l.user_id = $3"
        params.append(user_id)
    rows = await db.query_raw(
        f"""
        SELECT l.id, l.user_id, u.username, u.display_name,
               l.currency, l.delta, l.balance_after, l.source, l.source_id,
               l.metadata, l.created_at,
               (
                   SELECT ai.raw_profile->>'nickname'
                   FROM auth_identities ai
                   WHERE ai.user_id = l.user_id AND ai.provider = 'wechat'
                   ORDER BY ai.updated_at DESC
                   LIMIT 1
               ) AS nickname
        FROM wallet_ledger l
        LEFT JOIN users u ON u.id = l.user_id
        {where}
        ORDER BY l.created_at DESC, l.id DESC
        LIMIT $1 OFFSET $2
        """,
        *params,
    )
    items: list[dict[str, Any]] = []
    for row in rows:
        created_at = _field(row, "created_at")
        items.append(
            {
                "id": str(_field(row, "id", "")),
                "user_id": str(_field(row, "user_id", "")),
                "username": _field(row, "username"),
                "display_name": _field(row, "display_name"),
                "nickname": _field(row, "nickname"),
                "currency": str(_field(row, "currency", "")),
                "delta": int(_field(row, "delta", 0) or 0),
                "balance_after": int(_field(row, "balance_after", 0) or 0),
                "source": str(_field(row, "source", "")),
                "source_id": _field(row, "source_id"),
                "metadata": _json(_field(row, "metadata")),
                "created_at": _iso(created_at),
            }
        )
    return items


async def list_admin_point_ledger(
    *,
    user_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Point-currency ledger across all users for the admin audit view."""
    limit = min(max(limit, 1), 200)
    offset = max(offset, 0)
    where = "WHERE l.currency = 'point'"
    params: list[Any] = [limit, offset]
    if user_id:
        where += " AND l.user_id = $3"
        params.append(user_id)
    rows = await db.query_raw(
        f"""
        SELECT l.id, l.user_id, u.username, u.display_name,
               l.currency, l.delta, l.balance_after, l.source, l.source_id,
               l.metadata, l.created_at,
               (
                   SELECT ai.raw_profile->>'nickname'
                   FROM auth_identities ai
                   WHERE ai.user_id = l.user_id AND ai.provider = 'wechat'
                   ORDER BY ai.updated_at DESC
                   LIMIT 1
               ) AS nickname
        FROM wallet_ledger l
        LEFT JOIN users u ON u.id = l.user_id
        {where}
        ORDER BY l.created_at DESC, l.id DESC
        LIMIT $1 OFFSET $2
        """,
        *params,
    )
    items: list[dict[str, Any]] = []
    for row in rows:
        created_at = _field(row, "created_at")
        items.append(
            {
                "id": str(_field(row, "id", "")),
                "user_id": str(_field(row, "user_id", "")),
                "username": _field(row, "username"),
                "display_name": _field(row, "display_name"),
                "nickname": _field(row, "nickname"),
                "currency": str(_field(row, "currency", "")),
                "delta": int(_field(row, "delta", 0) or 0),
                "balance_after": int(_field(row, "balance_after", 0) or 0),
                "source": str(_field(row, "source", "")),
                "source_id": _field(row, "source_id"),
                "metadata": _json(_field(row, "metadata")),
                "created_at": _iso(created_at),
            }
        )
    return items
