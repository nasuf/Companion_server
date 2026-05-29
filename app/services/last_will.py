from __future__ import annotations

from datetime import UTC, date, datetime
import hashlib
import json
import logging
from typing import Any

from app.db import db
from app.models.last_will import LastWillContact
from app.services.last_will_crypto import protect_contact, reveal_contact
from app.services.user_activity import local_activity_date

logger = logging.getLogger(__name__)


def _field(row: Any, name: str) -> Any:
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name, None)


def _json_contacts(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    data = getattr(value, "data", value)
    if not isinstance(data, list):
        return []
    contacts: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            try:
                contacts.append(
                    LastWillContact.model_validate(reveal_contact(item)).model_dump()
                )
            except Exception:
                continue
    return contacts[:3]


def _last_active_date(row: Any, today: date) -> date:
    raw = (
        _field(row, "lastActivityDate")
        or _field(row, "userLastSeenAt")
        or _field(row, "userUpdatedAt")
        or _field(row, "userCreatedAt")
    )
    if raw is None:
        return today
    if isinstance(raw, datetime):
        return local_activity_date(raw)
    if isinstance(raw, date):
        return raw
    return date.fromisoformat(str(raw)[:10])


async def scan_due_last_wills(now: datetime | None = None, *, limit: int = 500) -> dict[str, int]:
    """Mark active last wills as triggered after enough missed login days.

    Delivery providers are intentionally decoupled. This scan creates pending
    delivery rows per contact channel; an email/SMS worker can later consume
    them without reopening the trigger decision.
    """
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    today = local_activity_date(current)
    rows = await db.query_raw(
        """
        SELECT
            lw.id,
            lw.user_id AS "userId",
            lw.contacts,
            lw.inactivity_days AS "inactivityDays",
            u.last_seen_at AS "userLastSeenAt",
            u.updated_at AS "userUpdatedAt",
            u.created_at AS "userCreatedAt",
            MAX(uda.local_date) AS "lastActivityDate"
        FROM last_wills lw
        JOIN users u ON u.id = lw.user_id
        LEFT JOIN user_daily_activity uda ON uda.user_id = lw.user_id
        WHERE lw.status = 'active'
          AND lw.triggered_at IS NULL
          AND lw.inactivity_days BETWEEN 5 AND 365
          AND btrim(lw.content) <> ''
          AND jsonb_typeof(lw.contacts) = 'array'
          AND jsonb_array_length(lw.contacts) > 0
        GROUP BY lw.id, u.id
        ORDER BY lw.started_at ASC NULLS LAST, lw.updated_at ASC
        LIMIT $1
        """,
        limit,
    )

    checked = 0
    triggered = 0
    deliveries = 0
    for row in rows:
        checked += 1
        inactivity_days = int(_field(row, "inactivityDays") or 0)
        last_active = _last_active_date(row, today)
        missed_days = (today - last_active).days
        if missed_days < inactivity_days:
            continue

        will_id = str(_field(row, "id"))
        contacts = _json_contacts(_field(row, "contacts"))
        if not contacts:
            logger.warning("[last_will] skip active will with no valid contacts id=%s", will_id)
            continue

        updated_rows = await db.query_raw(
            """
            UPDATE last_wills
            SET status = 'triggered',
                triggered_at = $2::timestamp,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $1
              AND status = 'active'
              AND triggered_at IS NULL
            RETURNING id
            """,
            will_id,
            current,
        )
        if not updated_rows:
            logger.info("[last_will] trigger skipped after concurrent update id=%s", will_id)
            continue

        created = await _create_pending_deliveries(will_id, contacts)
        deliveries += created
        triggered += 1
        logger.info(
            "[last_will] triggered id=%s missed_days=%s threshold=%s deliveries=%s",
            will_id,
            missed_days,
            inactivity_days,
            created,
        )

    return {"checked": checked, "triggered": triggered, "deliveries": deliveries}


async def _create_pending_deliveries(will_id: str, contacts: list[dict[str, Any]]) -> int:
    created = 0
    for contact in contacts[:3]:
        for channel in ("email", "phone"):
            value = contact.get(channel)
            if not value:
                continue
            inserted = await db.query_raw(
                """
                INSERT INTO last_will_deliveries (
                    id, last_will_id, channel, contact, dedupe_key, status,
                    created_at, updated_at
                )
                VALUES (
                    gen_random_uuid(), $1, $2, $3::jsonb, $4, 'pending',
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                )
                ON CONFLICT (last_will_id, channel, dedupe_key) DO NOTHING
                RETURNING id
                """,
                will_id,
                channel,
                json.dumps(protect_contact(contact), ensure_ascii=False),
                _delivery_dedupe_key(channel, str(value)),
            )
            if inserted:
                created += 1
    return created


def _delivery_dedupe_key(channel: str, value: str) -> str:
    raw = f"{channel}:{value.strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
