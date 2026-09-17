from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.db import db
from app.observability.events import EVT_VIP_CODE_REVOKE
from app.services.vip.activation_codes.errors import VipActivationError
from app.services.vip.activation_codes.generator import generate_code_string
from app.services.vip.activation_codes import repo
from app.services.vip.activation_codes.normalize import format_code_display, strip_code_separators
from app.services.vip.entitlements import recompute_vip_entitlements

logger = logging.getLogger(__name__)


def _field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _mask_code(code: str) -> str:
    normalized = strip_code_separators(code)
    if not normalized:
        return "**"
    display = format_code_display(normalized)
    if len(normalized) <= 6:
        return display[:2] + "**"
    if "-" in display:
        left, _, right = display.partition("-")
        if len(left) >= 2 and len(right) >= 2:
            return f"{left[:2]}**-**{right[-2:]}"
    return display[:4] + "**" + display[-2:]


async def create_codes(
    *,
    duration_days: int,
    count: int = 1,
    max_redemptions: int | None = 1,
    valid_from: datetime | None = None,
    valid_until: datetime | None = None,
    note: str | None = None,
    created_by: str | None = None,
) -> list[dict[str, Any]]:
    if duration_days <= 0:
        raise ValueError("duration_days must be positive")
    if count <= 0 or count > 500:
        raise ValueError("count must be 1..500")
    if max_redemptions is not None and max_redemptions <= 0:
        raise ValueError("max_redemptions must be positive or null")

    created: list[dict[str, Any]] = []
    for _ in range(count):
        for _attempt in range(20):
            code_str = generate_code_string()
            try:
                row = await repo.insert_code(
                    code=code_str,
                    duration_days=duration_days,
                    max_redemptions=max_redemptions,
                    valid_from=valid_from,
                    valid_until=valid_until,
                    note=note,
                    created_by=created_by,
                )
                created.append(_serialize_code(row))
                break
            except Exception as exc:
                if "unique" in str(exc).lower() or "duplicate" in str(exc).lower():
                    continue
                raise
        else:
            raise RuntimeError("failed to generate unique activation code")

    return created


def _serialize_code(row: dict) -> dict[str, Any]:
    return {
        "id": row["id"],
        "code": format_code_display(str(row["code"])),
        "duration_days": row["duration_days"],
        "max_redemptions": row["max_redemptions"],
        "redemption_count": row["redemption_count"],
        "enabled": row["enabled"],
        "valid_from": _iso(row.get("valid_from")),
        "valid_until": _iso(row.get("valid_until")),
        "note": row.get("note"),
        "created_by": row.get("created_by"),
        "created_at": _iso(row.get("created_at")),
        "updated_at": _iso(row.get("updated_at")),
    }


async def list_codes(
    *,
    q: str | None = None,
    enabled: bool | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    limit = min(max(limit, 1), 200)
    offset = max(offset, 0)
    clauses = ["1=1"]
    args: list[Any] = []
    idx = 1

    if q:
        clauses.append(f"code ILIKE ${idx}")
        args.append(f"%{strip_code_separators(q)}%")
        idx += 1
    if enabled is not None:
        clauses.append(f"enabled = ${idx}")
        args.append(enabled)
        idx += 1

    where = " AND ".join(clauses)
    count_rows = await db.query_raw(
        f"SELECT COUNT(*) AS cnt FROM vip_activation_codes WHERE {where}",
        *args,
    )
    total = int(_field(count_rows[0], "cnt") or 0)

    args.extend([limit, offset])
    rows = await db.query_raw(
        f"""
        SELECT id, code, duration_days, max_redemptions, redemption_count,
               enabled, valid_from, valid_until, note, created_by, created_at, updated_at
        FROM vip_activation_codes
        WHERE {where}
        ORDER BY created_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
        """,
        *args,
    )
    items = [_serialize_code(repo._row_to_code(r)) for r in rows]
    return {"items": items, "total": total, "limit": limit, "offset": offset}


async def set_code_enabled(code_id: str, *, enabled: bool) -> dict[str, Any]:
    rows = await db.query_raw(
        """
        UPDATE vip_activation_codes
        SET enabled = $2, updated_at = CURRENT_TIMESTAMP
        WHERE id = $1::uuid
        RETURNING id, code, duration_days, max_redemptions, redemption_count,
                  enabled, valid_from, valid_until, note, created_by, created_at, updated_at
        """,
        code_id,
        enabled,
    )
    if not rows:
        raise VipActivationError("not_found", "激活码不存在")
    return _serialize_code(repo._row_to_code(rows[0]))


async def list_redemptions(
    *,
    user_id: str | None = None,
    code_id: str | None = None,
    code_q: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    limit = min(max(limit, 1), 200)
    offset = max(offset, 0)
    clauses = ["1=1"]
    args: list[Any] = []
    idx = 1

    if user_id:
        clauses.append(f"r.user_id = ${idx}")
        args.append(user_id)
        idx += 1
    if code_id:
        clauses.append(f"r.code_id = ${idx}::uuid")
        args.append(code_id)
        idx += 1
    if code_q:
        clauses.append(f"c.code ILIKE ${idx}")
        args.append(f"%{strip_code_separators(code_q)}%")
        idx += 1
    if status:
        clauses.append(f"r.status = ${idx}")
        args.append(status)
        idx += 1

    where = " AND ".join(clauses)
    count_rows = await db.query_raw(
        f"""
        SELECT COUNT(*) AS cnt
        FROM vip_code_redemptions r
        JOIN vip_activation_codes c ON c.id = r.code_id
        WHERE {where}
        """,
        *args,
    )
    total = int(_field(count_rows[0], "cnt") or 0)

    args.extend([limit, offset])
    rows = await db.query_raw(
        f"""
        SELECT r.id, r.code_id, r.user_id, r.duration_days, r.status,
               r.redeemed_at, r.effective_start, r.effective_end,
               r.revoked_at, r.revoked_by,
               c.code, c.duration_days AS code_duration_days,
               u.display_name AS user_display_name, u.email AS user_email
        FROM vip_code_redemptions r
        JOIN vip_activation_codes c ON c.id = r.code_id
        LEFT JOIN users u ON u.id = r.user_id
        WHERE {where}
        ORDER BY r.redeemed_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
        """,
        *args,
    )
    items = [
        {
            "id": str(_field(r, "id")),
            "code_id": str(_field(r, "code_id")),
            "code": format_code_display(str(_field(r, "code"))),
            "code_preview": _mask_code(str(_field(r, "code"))),
            "user_id": str(_field(r, "user_id")),
            "user_display_name": _field(r, "user_display_name"),
            "user_email": _field(r, "user_email"),
            "duration_days": int(_field(r, "duration_days") or 0),
            "status": str(_field(r, "status")),
            "redeemed_at": _iso(_field(r, "redeemed_at")),
            "effective_start": _iso(_field(r, "effective_start")),
            "effective_end": _iso(_field(r, "effective_end")),
            "revoked_at": _iso(_field(r, "revoked_at")),
            "revoked_by": str(_field(r, "revoked_by")) if _field(r, "revoked_by") else None,
        }
        for r in rows
    ]
    return {"items": items, "total": total, "limit": limit, "offset": offset}


async def revoke_redemption(redemption_id: str, *, admin_user_id: str) -> dict[str, Any]:
    async with db.tx() as tx:
        rows = await tx.query_raw(
            """
            SELECT r.id, r.user_id, r.status, r.code_id
            FROM vip_code_redemptions r
            WHERE r.id = $1::uuid
            FOR UPDATE
            """,
            redemption_id,
        )
        if not rows:
            raise VipActivationError("not_found", "兑换记录不存在")
        row = rows[0]
        if str(_field(row, "status")) == repo.STATUS_REVOKED:
            raise VipActivationError("already_revoked", "该兑换已撤销")

        user_id = str(_field(row, "user_id"))
        await tx.execute_raw(
            """
            UPDATE vip_code_redemptions
            SET status = $2,
                revoked_at = CURRENT_TIMESTAMP,
                revoked_by = $3
            WHERE id = $1::uuid
            """,
            redemption_id,
            repo.STATUS_REVOKED,
            admin_user_id,
        )
        await tx.execute_raw(
            """
            UPDATE vip_activation_codes
            SET redemption_count = GREATEST(redemption_count - 1, 0),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $1::uuid
            """,
            str(_field(row, "code_id")),
        )
        await recompute_vip_entitlements(user_id, client=tx, clear_lapse=True)

    logger.info(
        "vip code redemption revoked id=%s user=%s",
        redemption_id[:8],
        user_id[:8],
        extra={"event": EVT_VIP_CODE_REVOKE, "redemption_id": redemption_id},
    )
    return {"id": redemption_id, "status": repo.STATUS_REVOKED, "user_id": user_id}
