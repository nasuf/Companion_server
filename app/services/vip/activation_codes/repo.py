from __future__ import annotations

from datetime import datetime
from typing import Any

from app.db import db
from app.services.vip.activation_codes.normalize import normalize_code
from app.services.vip.entitlements import REDEMPTION_GRANTED, REDEMPTION_REVOKED, naive_dt

STATUS_GRANTED = REDEMPTION_GRANTED
STATUS_REVOKED = REDEMPTION_REVOKED


def _field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _row_to_code(row: Any) -> dict[str, Any]:
    return {
        "id": str(_field(row, "id")),
        "code": str(_field(row, "code")),
        "duration_days": int(_field(row, "duration_days") or 0),
        "max_redemptions": _field(row, "max_redemptions"),
        "redemption_count": int(_field(row, "redemption_count") or 0),
        "enabled": bool(_field(row, "enabled", True)),
        "valid_from": _field(row, "valid_from"),
        "valid_until": _field(row, "valid_until"),
        "note": _field(row, "note"),
        "created_by": str(_field(row, "created_by")) if _field(row, "created_by") else None,
        "created_at": _field(row, "created_at"),
        "updated_at": _field(row, "updated_at"),
    }


async def fetch_code_by_normalized(normalized: str, *, client: Any | None = None) -> dict | None:
    executor = client or db
    rows = await executor.query_raw(
        """
        SELECT id, code, duration_days, max_redemptions, redemption_count,
               enabled, valid_from, valid_until, note, created_by, created_at, updated_at
        FROM vip_activation_codes
        WHERE code = $1
        LIMIT 1
        """,
        normalized,
    )
    return _row_to_code(rows[0]) if rows else None


async def fetch_code_for_update(code_id: str, *, client: Any | None = None) -> dict | None:
    executor = client or db
    rows = await executor.query_raw(
        """
        SELECT id, code, duration_days, max_redemptions, redemption_count,
               enabled, valid_from, valid_until, note, created_by, created_at, updated_at
        FROM vip_activation_codes
        WHERE id = $1::uuid
        FOR UPDATE
        """,
        code_id,
    )
    return _row_to_code(rows[0]) if rows else None


async def user_has_granted_redemption(
    code_id: str, user_id: str, *, client: Any | None = None
) -> bool:
    executor = client or db
    rows = await executor.query_raw(
        """
        SELECT 1 FROM vip_code_redemptions
        WHERE code_id = $1::uuid AND user_id = $2::uuid AND status = $3
        LIMIT 1
        """,
        code_id,
        user_id,
        STATUS_GRANTED,
    )
    return bool(rows)


async def insert_redemption(
    *,
    code_id: str,
    user_id: str,
    duration_days: int,
    effective_start: datetime,
    effective_end: datetime,
    client: Any,
) -> str:
    rows = await client.query_raw(
        """
        INSERT INTO vip_code_redemptions (
            code_id, user_id, duration_days, status,
            redeemed_at, effective_start, effective_end
        )
        VALUES ($1::uuid, $2::uuid, $3, $4, CURRENT_TIMESTAMP, $5::timestamp, $6::timestamp)
        RETURNING id
        """,
        code_id,
        user_id,
        duration_days,
        STATUS_GRANTED,
        naive_dt(effective_start),
        naive_dt(effective_end),
    )
    return str(_field(rows[0], "id"))


async def increment_redemption_count(code_id: str, *, client: Any) -> None:
    await client.execute_raw(
        """
        UPDATE vip_activation_codes
        SET redemption_count = redemption_count + 1,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1::uuid
        """,
        code_id,
    )


async def insert_code(
    *,
    code: str,
    duration_days: int,
    max_redemptions: int | None,
    valid_from: datetime | None,
    valid_until: datetime | None,
    note: str | None,
    created_by: str | None,
    client: Any | None = None,
) -> dict:
    normalized = normalize_code(code)
    executor = client or db
    rows = await executor.query_raw(
        """
        INSERT INTO vip_activation_codes (
            code, duration_days, max_redemptions, enabled,
            valid_from, valid_until, note, created_by
        )
        VALUES ($1, $2, $3, TRUE, $4::timestamp, $5::timestamp, $6, $7::uuid)
        RETURNING id, code, duration_days, max_redemptions, redemption_count,
                  enabled, valid_from, valid_until, note, created_by, created_at, updated_at
        """,
        normalized,
        duration_days,
        max_redemptions,
        naive_dt(valid_from),
        naive_dt(valid_until),
        note,
        created_by,
    )
    return _row_to_code(rows[0])
