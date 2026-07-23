"""Game points wallet: daily grant, per-game settlement, levels, conversion.

This is a separate spendable currency from the shop wallet (``user_wallets``):

* Each user is granted +20 game points on their first activity of a new day,
  but only when their current balance is below 20 (so a healthy balance is not
  inflated every day). See :func:`ensure_daily_grant`.
* When the balance hits 0 the user cannot start a new game until the next day's
  grant restores it. Play is gated in ``games.native.create_session``.
* Winning / losing / quitting a native game adjusts the balance by the amount
  configured per game (:data:`game_point_rules`); settlement runs inside the
  same DB transaction that records the terminal game event, so retries can never
  double-count a match (guarded further by the ledger's partial unique index).
* The level (白手套 / 蓝手套 …) is derived purely from the current balance
  against the admin-editable ``game_level_tiers`` ladder.
* Game points can be converted 1:1 into shop points (``user_wallets.point_balance``),
  but only the portion above 20 and never in reverse.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any

from app.db import db
from app.services.games.balance import GAME_TITLES

# Business timezone is fixed UTC+8, matching the rest of the schedule domain.
_TZ = timezone(timedelta(hours=8))

# Daily grant amount and the balance floor below which the grant tops up.
DAILY_GRANT = 20
# Only the game points above this floor may be converted to shop points; the
# floor stays behind so the user can always keep playing.
CONVERT_FLOOR = 20
# Game point -> shop point conversion rate (1:1 per product spec).
CONVERT_RATE = 1

# Ledger sources.
_SOURCE_DAILY_GRANT = "daily_grant"
_SOURCE_GAME_SETTLE = "game_settle"
_SOURCE_CONVERT = "convert_to_shop"
# Official (admin) balance grant/adjustment; never touches lifetime_earned so
# the level is unaffected.
_SOURCE_ADMIN_GRANT = "admin_grant"
# Shop-side ledger source for the credited shop points.
_SHOP_SOURCE_CONVERT = "game_point_conversion"


def _field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _load_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _today() -> date:
    return datetime.now(_TZ).date()


def _normalize_date(value: Any) -> date | None:
    """Coerce a raw-query date column (date / datetime / ISO string) to a date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


async def ensure_wallet(user_id: str) -> dict[str, Any]:
    rows = await db.query_raw(
        """
        INSERT INTO user_game_wallets (user_id)
        VALUES ($1)
        ON CONFLICT (user_id) DO UPDATE
        SET updated_at = user_game_wallets.updated_at
        RETURNING balance, lifetime_earned, last_grant_date
        """,
        user_id,
    )
    row = rows[0]
    return {
        "balance": int(_field(row, "balance", 0) or 0),
        "lifetime_earned": int(_field(row, "lifetime_earned", 0) or 0),
        "last_grant_date": _field(row, "last_grant_date"),
    }


async def _record_ledger(
    *,
    user_id: str,
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
        INSERT INTO game_point_ledger
            (user_id, delta, balance_after, source, source_id, metadata)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb)
        """,
        user_id,
        delta,
        balance_after,
        source,
        source_id,
        json.dumps(metadata or {}, ensure_ascii=False),
    )


async def ensure_daily_grant(user_id: str) -> int:
    """Apply the once-per-day grant if due and return the current balance.

    The grant fires at most once per UTC+8 day and only tops up when the balance
    is below :data:`DAILY_GRANT`. The wallet row is locked ``FOR UPDATE`` so two
    concurrent first-of-day requests cannot both credit the grant.
    """

    await ensure_wallet(user_id)
    today = _today()
    # Prisma's raw-query builder cannot serialize a bare ``datetime.date``, so
    # dates are passed as ISO strings and cast to ``date`` in SQL ($n::date).
    today_iso = today.isoformat()
    async with db.tx() as tx:
        locked = await tx.query_raw(
            """
            SELECT balance, last_grant_date
            FROM user_game_wallets
            WHERE user_id = $1
            FOR UPDATE
            """,
            user_id,
        )
        row = locked[0]
        balance = int(_field(row, "balance", 0) or 0)
        last_grant = _normalize_date(_field(row, "last_grant_date"))

        if last_grant == today:
            return balance

        # A new day: mark it granted regardless, but only add points when the
        # balance is still below the daily floor (spec rule 4).
        if balance < DAILY_GRANT:
            new_balance = balance + DAILY_GRANT
            await tx.execute_raw(
                """
                UPDATE user_game_wallets
                SET balance = $2, last_grant_date = $3::date,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = $1
                """,
                user_id,
                new_balance,
                today_iso,
            )
            await _record_ledger(
                user_id=user_id,
                delta=DAILY_GRANT,
                balance_after=new_balance,
                source=_SOURCE_DAILY_GRANT,
                source_id=today_iso,
                metadata={"granted_on": today_iso},
                client=tx,
            )
            return new_balance

        await tx.execute_raw(
            """
            UPDATE user_game_wallets
            SET last_grant_date = $2::date, updated_at = CURRENT_TIMESTAMP
            WHERE user_id = $1
            """,
            user_id,
            today_iso,
        )
        return balance


async def get_state(user_id: str, *, game_key: str | None = None) -> dict[str, Any]:
    """Return the full wallet state (applies the daily grant first).

    When ``game_key`` is provided, also include ``game_points_for_game`` — the
    net points this specific game has settled for the user (win/milestone credits
    minus lose/quit debits), for the per-game display on each game screen.
    """

    await ensure_daily_grant(user_id)
    wallet = await ensure_wallet(user_id)
    balance = int(wallet["balance"])
    lifetime_earned = int(wallet["lifetime_earned"])
    tiers = await list_level_tiers()
    # The level reflects points actually earned by playing (wins / milestones),
    # not the spendable balance, so it never drops when the user loses or
    # converts points to the shop.
    level, next_tier = _resolve_level(lifetime_earned, tiers)
    game_points_for_game: int | None = None
    if game_key:
        game_points_for_game = await _points_for_game(user_id, game_key)
    return {
        "balance": balance,
        "lifetime_earned": lifetime_earned,
        "can_play": balance > 0,
        "daily_grant": DAILY_GRANT,
        "convert_floor": CONVERT_FLOOR,
        "convert_rate": CONVERT_RATE,
        "convertible": max(0, balance - CONVERT_FLOOR),
        "level": level,
        "next_tier": next_tier,
        "game_points_for_game": game_points_for_game,
    }


async def _points_for_game(user_id: str, game_key: str) -> int:
    """Net game points settled for one game (from the settlement ledger)."""
    rows = await db.query_raw(
        """
        SELECT COALESCE(SUM(delta), 0) AS total
        FROM game_point_ledger
        WHERE user_id = $1
          AND source = $2
          AND metadata->>'game_key' = $3
        """,
        user_id,
        _SOURCE_GAME_SETTLE,
        game_key,
    )
    return int(_field(rows[0], "total", 0) or 0) if rows else 0


def _resolve_level(
    lifetime_earned: int,
    tiers: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not tiers:
        return None, None
    ordered = sorted(tiers, key=lambda item: int(item.get("cumulative_points") or 0))
    current: dict[str, Any] | None = None
    upcoming: dict[str, Any] | None = None
    for tier in ordered:
        if int(tier.get("cumulative_points") or 0) <= lifetime_earned:
            current = tier
        else:
            upcoming = tier
            break
    # Below the very first threshold still maps to the first tier.
    if current is None:
        current = ordered[0]
    return current, upcoming


# ─────────────────────────── settlement ───────────────────────────


def _outcome_delta(rules: dict[str, Any], outcome: str) -> int:
    mapping = {
        "win": int(rules.get("win") or 0),
        "lose": int(rules.get("lose") or 0),
        "draw": int(rules.get("draw") or 0),
        "aborted": int(rules.get("quit") or 0),
    }
    return mapping.get(outcome, 0)


def _milestone_delta(rules: dict[str, Any], outcome: str, max_tile: int) -> int:
    if outcome == "aborted":
        quit_rule = rules.get("quit_below_threshold") or {}
        threshold = int(quit_rule.get("threshold") or 0)
        if max_tile < threshold:
            return int(quit_rule.get("below") or 0)
        return int(quit_rule.get("at_or_above") or 0)
    # Finished (win or lose): award the points of the highest milestone reached.
    milestones = rules.get("milestones") or []
    points = 0
    for entry in sorted(milestones, key=lambda item: int(item.get("tile") or 0)):
        if max_tile >= int(entry.get("tile") or 0):
            points = int(entry.get("points") or 0)
    return points


async def settle_session(session: Any, *, database: Any) -> None:
    """Adjust the game-point balance for one terminal native game session.

    Must run inside the transaction that records the terminal game event so the
    ``inserted`` idempotency guard in ``native.handle_event`` protects it; the
    ledger's partial unique index on ``(user_id, source, source_id)`` is an
    additional safety net against double settlement.
    """

    game_key = str(getattr(session, "game_key", "") or "")
    user_id = str(getattr(session, "user_id", "") or "")
    session_id = str(getattr(session, "id", "") or "")
    if not game_key or not user_id or not session_id:
        return

    result = _load_json(getattr(session, "result", None))
    outcome = str(result.get("user_outcome") or "").lower()
    if outcome not in {"win", "lose", "draw", "aborted"}:
        return

    rows = await database.query_raw(
        "SELECT rules FROM game_point_rules WHERE game_key = $1 LIMIT 1",
        game_key,
    )
    if not rows:
        return
    rules = _load_json(_field(rows[0], "rules"))
    rule_type = str(rules.get("type") or "outcome")

    max_tile = 0
    if rule_type == "milestone":
        final_payload = _load_json(result.get("final_payload"))
        max_tile = int(final_payload.get("max_tile") or 0)
        delta = _milestone_delta(rules, outcome, max_tile)
    else:
        delta = _outcome_delta(rules, outcome)

    if delta == 0:
        return

    # Ensure the wallet row exists, then lock it and apply the delta with a hard
    # floor at 0 (losing/quitting can never push the balance negative).
    await database.execute_raw(
        """
        INSERT INTO user_game_wallets (user_id)
        VALUES ($1)
        ON CONFLICT (user_id) DO UPDATE SET updated_at = user_game_wallets.updated_at
        """,
        user_id,
    )
    locked = await database.query_raw(
        "SELECT balance FROM user_game_wallets WHERE user_id = $1 FOR UPDATE",
        user_id,
    )
    current = int(_field(locked[0], "balance", 0) or 0)
    new_balance = max(0, current + delta)
    applied = new_balance - current
    # Only points actually earned by winning / reaching a milestone (a positive
    # settlement) count toward the level; losses and quits never reduce it.
    earned = max(0, delta)
    if applied == 0 and earned == 0:
        return
    await database.execute_raw(
        """
        UPDATE user_game_wallets
        SET balance = $2,
            lifetime_earned = lifetime_earned + $3,
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = $1
        """,
        user_id,
        new_balance,
        earned,
    )
    metadata = {
        "game_key": game_key,
        "outcome": outcome,
        "intended_delta": delta,
        "earned": earned,
    }
    if rule_type == "milestone":
        metadata["max_tile"] = max_tile
    await _record_ledger(
        user_id=user_id,
        delta=applied,
        balance_after=new_balance,
        source=_SOURCE_GAME_SETTLE,
        source_id=session_id,
        metadata=metadata,
        client=database,
    )


# ─────────────────────────── conversion ───────────────────────────


async def convert_to_shop(user_id: str, amount: int) -> dict[str, Any]:
    """Convert ``amount`` game points into shop points (1:1, irreversible).

    Only the balance above :data:`CONVERT_FLOOR` may be converted, and the
    resulting balance must remain at or above the floor.
    """

    if amount <= 0:
        raise ValueError("invalid_amount")
    await ensure_wallet(user_id)
    shop_delta = amount * CONVERT_RATE
    async with db.tx() as tx:
        locked = await tx.query_raw(
            "SELECT balance FROM user_game_wallets WHERE user_id = $1 FOR UPDATE",
            user_id,
        )
        balance = int(_field(locked[0], "balance", 0) or 0)
        if balance - amount < CONVERT_FLOOR:
            raise ValueError("insufficient_convertible")
        new_game_balance = balance - amount
        await tx.execute_raw(
            """
            UPDATE user_game_wallets
            SET balance = $2, updated_at = CURRENT_TIMESTAMP
            WHERE user_id = $1
            """,
            user_id,
            new_game_balance,
        )
        await _record_ledger(
            user_id=user_id,
            delta=-amount,
            balance_after=new_game_balance,
            source=_SOURCE_CONVERT,
            metadata={"shop_point_delta": shop_delta},
            client=tx,
        )
        # Credit the shop wallet (creating it if absent) in the same transaction.
        await tx.execute_raw(
            """
            INSERT INTO user_wallets (user_id, point_balance)
            VALUES ($1, $2)
            ON CONFLICT (user_id) DO UPDATE
            SET point_balance = user_wallets.point_balance + $2,
                updated_at = CURRENT_TIMESTAMP
            """,
            user_id,
            shop_delta,
        )
        shop_rows = await tx.query_raw(
            "SELECT ticket_balance, point_balance FROM user_wallets WHERE user_id = $1",
            user_id,
        )
        shop_point_balance = int(_field(shop_rows[0], "point_balance", 0) or 0)
        await _record_shop_ledger(
            user_id=user_id,
            delta=shop_delta,
            balance_after=shop_point_balance,
            metadata={"game_point_delta": -amount},
            client=tx,
        )
    return {
        "game_balance": new_game_balance,
        "shop_point_balance": shop_point_balance,
        "converted": amount,
        "shop_point_delta": shop_delta,
    }


async def _record_shop_ledger(
    *,
    user_id: str,
    delta: int,
    balance_after: int,
    metadata: dict[str, Any] | None = None,
    client: Any | None = None,
) -> None:
    executor = client or db
    await executor.execute_raw(
        """
        INSERT INTO wallet_ledger
            (user_id, currency, delta, balance_after, source, source_id, metadata)
        VALUES ($1, 'point', $2, $3, $4, NULL, $5::jsonb)
        """,
        user_id,
        delta,
        balance_after,
        _SHOP_SOURCE_CONVERT,
        json.dumps(metadata or {}, ensure_ascii=False),
    )


# ─────────────────────────── admin grant + ledger ───────────────────────────


async def admin_grant(
    user_id: str,
    amount: int,
    *,
    note: str | None = None,
) -> dict[str, Any]:
    """Official balance grant/adjustment (adds ``amount`` to the balance).

    Only the spendable balance changes — ``lifetime_earned`` is untouched, so the
    game level never moves. The balance is floored at 0 (a negative adjustment
    cannot push it below zero).
    """
    if amount == 0:
        raise ValueError("invalid_amount")
    await ensure_wallet(user_id)
    async with db.tx() as tx:
        locked = await tx.query_raw(
            "SELECT balance FROM user_game_wallets WHERE user_id = $1 FOR UPDATE",
            user_id,
        )
        current = int(_field(locked[0], "balance", 0) or 0)
        new_balance = max(0, current + amount)
        applied = new_balance - current
        if applied == 0:
            return {"user_id": user_id, "balance": current, "delta": 0}
        await tx.execute_raw(
            """
            UPDATE user_game_wallets
            SET balance = $2, updated_at = CURRENT_TIMESTAMP
            WHERE user_id = $1
            """,
            user_id,
            new_balance,
        )
        await _record_ledger(
            user_id=user_id,
            delta=applied,
            balance_after=new_balance,
            source=_SOURCE_ADMIN_GRANT,
            metadata={"requested": amount, "note": (note or "").strip()},
            client=tx,
        )
    return {"user_id": user_id, "balance": new_balance, "delta": applied}


async def list_admin_ledger(
    *,
    user_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """All users' game-point change records for the admin console.

    Each row carries the running ``lifetime_earned`` at that point and the level
    it maps to, so level changes are visible inline (``level_up``). Only positive
    game settlements (win / milestone) advance the level.
    """
    limit = min(max(limit, 1), 200)
    offset = max(offset, 0)
    where = "WHERE l.user_id = $3" if user_id else ""
    # Window running-sum of earned points (drives the level) per user, computed
    # over the full history (before pagination) so each row's level is accurate.
    query = f"""
        SELECT l.id, l.user_id, u.username, l.delta, l.balance_after,
               l.source, l.metadata, l.created_at,
               SUM(
                   CASE WHEN l.source = 'game_settle'
                        THEN COALESCE((l.metadata->>'earned')::int, 0)
                        ELSE 0 END
               ) OVER (
                   PARTITION BY l.user_id
                   ORDER BY l.created_at ASC, l.id ASC
               ) AS lifetime_after
        FROM game_point_ledger l
        LEFT JOIN users u ON u.id = l.user_id
        {where}
        ORDER BY l.created_at DESC, l.id DESC
        LIMIT $1 OFFSET $2
    """
    params: list[Any] = [limit, offset]
    if user_id:
        params.append(user_id)
    rows = await db.query_raw(query, *params)
    tiers = await list_level_tiers()

    def _level_name(points: int) -> str | None:
        level, _ = _resolve_level(points, tiers)
        return level.get("tier_name") if level else None

    items: list[dict[str, Any]] = []
    for row in rows:
        metadata = _load_json(_field(row, "metadata"))
        lifetime_after = int(_field(row, "lifetime_after", 0) or 0)
        source = str(_field(row, "source", ""))
        earned = 0
        if source == _SOURCE_GAME_SETTLE:
            earned = int(metadata.get("earned") or 0)
        level_after = _level_name(lifetime_after)
        level_up = bool(
            earned > 0 and level_after != _level_name(lifetime_after - earned)
        )
        created_at = _field(row, "created_at")
        items.append(
            {
                "id": str(_field(row, "id", "")),
                "user_id": str(_field(row, "user_id", "")),
                "username": _field(row, "username"),
                "delta": int(_field(row, "delta", 0) or 0),
                "balance_after": int(_field(row, "balance_after", 0) or 0),
                "source": source,
                "metadata": metadata,
                "created_at": created_at.isoformat()
                if hasattr(created_at, "isoformat")
                else str(created_at or ""),
                "lifetime_after": lifetime_after,
                "level_name": level_after,
                "level_up": level_up,
            }
        )
    return items


# ─────────────────────────── admin config ───────────────────────────


async def list_level_tiers() -> list[dict[str, Any]]:
    rows = await db.query_raw(
        """
        SELECT sort_order, stage_name, tier_name, upgrade_points, cumulative_points
        FROM game_level_tiers
        ORDER BY sort_order ASC
        """
    )
    return [
        {
            "sort_order": int(_field(row, "sort_order", 0) or 0),
            "stage_name": str(_field(row, "stage_name", "") or ""),
            "tier_name": str(_field(row, "tier_name", "") or ""),
            "upgrade_points": int(_field(row, "upgrade_points", 0) or 0),
            "cumulative_points": int(_field(row, "cumulative_points", 0) or 0),
        }
        for row in rows
    ]


async def replace_level_tiers(tiers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not tiers:
        raise ValueError("empty_tiers")
    normalized: list[dict[str, Any]] = []
    for index, tier in enumerate(tiers, start=1):
        stage_name = str(tier.get("stage_name") or "").strip()
        tier_name = str(tier.get("tier_name") or "").strip()
        if not stage_name or not tier_name:
            raise ValueError("invalid_tier")
        upgrade_points = int(tier.get("upgrade_points") or 0)
        cumulative_points = int(tier.get("cumulative_points") or 0)
        if upgrade_points < 0 or cumulative_points < 0:
            raise ValueError("invalid_tier")
        normalized.append(
            {
                "sort_order": index,
                "stage_name": stage_name,
                "tier_name": tier_name,
                "upgrade_points": upgrade_points,
                "cumulative_points": cumulative_points,
            }
        )
    # Cumulative thresholds must be non-decreasing for the level lookup to work.
    previous = -1
    for tier in normalized:
        if tier["cumulative_points"] < previous:
            raise ValueError("cumulative_points_must_be_non_decreasing")
        previous = tier["cumulative_points"]
    async with db.tx() as tx:
        await tx.execute_raw("DELETE FROM game_level_tiers")
        for tier in normalized:
            await tx.execute_raw(
                """
                INSERT INTO game_level_tiers
                    (sort_order, stage_name, tier_name, upgrade_points, cumulative_points)
                VALUES ($1, $2, $3, $4, $5)
                """,
                tier["sort_order"],
                tier["stage_name"],
                tier["tier_name"],
                tier["upgrade_points"],
                tier["cumulative_points"],
            )
    return await list_level_tiers()


async def list_point_rules() -> list[dict[str, Any]]:
    rows = await db.query_raw(
        "SELECT game_key, rules FROM game_point_rules"
    )
    stored = {str(_field(row, "game_key")): _load_json(_field(row, "rules")) for row in rows}
    result: list[dict[str, Any]] = []
    for game_key, title in GAME_TITLES.items():
        result.append(
            {
                "game_key": game_key,
                "title": title,
                "rules": stored.get(game_key, {}),
            }
        )
    return result


def _validate_rules(rules: dict[str, Any]) -> dict[str, Any]:
    rule_type = str(rules.get("type") or "outcome")
    if rule_type == "outcome":
        cleaned: dict[str, Any] = {"type": "outcome"}
        for key in ("win", "lose", "draw", "quit"):
            value = rules.get(key, 0)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"invalid_rule_value:{key}")
            cleaned[key] = value
        if "pending_pm" in rules:
            cleaned["pending_pm"] = bool(rules["pending_pm"])
        return cleaned
    if rule_type == "milestone":
        raw_milestones = rules.get("milestones")
        if not isinstance(raw_milestones, list) or not raw_milestones:
            raise ValueError("invalid_milestones")
        milestones: list[dict[str, int]] = []
        previous_tile = -1
        for entry in raw_milestones:
            if not isinstance(entry, dict):
                raise ValueError("invalid_milestones")
            tile = entry.get("tile")
            points = entry.get("points")
            if isinstance(tile, bool) or not isinstance(tile, int) or tile <= 0:
                raise ValueError("invalid_milestones")
            if isinstance(points, bool) or not isinstance(points, int):
                raise ValueError("invalid_milestones")
            if tile <= previous_tile:
                raise ValueError("milestones_must_ascend")
            previous_tile = tile
            milestones.append({"tile": tile, "points": points})
        quit_rule = rules.get("quit_below_threshold") or {}
        if not isinstance(quit_rule, dict):
            raise ValueError("invalid_quit_rule")
        cleaned_quit = {
            "threshold": int(quit_rule.get("threshold") or 0),
            "below": int(quit_rule.get("below") or 0),
            "at_or_above": int(quit_rule.get("at_or_above") or 0),
        }
        cleaned = {
            "type": "milestone",
            "milestones": milestones,
            "quit_below_threshold": cleaned_quit,
        }
        if "pending_pm" in rules:
            cleaned["pending_pm"] = bool(rules["pending_pm"])
        return cleaned
    raise ValueError("invalid_rule_type")


async def update_point_rule(game_key: str, rules: dict[str, Any]) -> dict[str, Any]:
    if game_key not in GAME_TITLES:
        raise ValueError("unsupported_game")
    cleaned = _validate_rules(rules)
    await db.execute_raw(
        """
        INSERT INTO game_point_rules (game_key, rules, updated_at)
        VALUES ($1, $2::jsonb, CURRENT_TIMESTAMP)
        ON CONFLICT (game_key) DO UPDATE
        SET rules = EXCLUDED.rules, updated_at = CURRENT_TIMESTAMP
        """,
        game_key,
        json.dumps(cleaned, ensure_ascii=False),
    )
    return {"game_key": game_key, "title": GAME_TITLES[game_key], "rules": cleaned}
