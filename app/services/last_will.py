from __future__ import annotations

from datetime import UTC, date, datetime
import hashlib
import json
import logging
from typing import Any

from app.config import settings
from app.db import db
from app.models.last_will import LastWillContact
from app.services.last_will_crypto import protect_contact, reveal_contact, reveal_text
from app.services.sms.service import normalize_cn_phone
from app.services.sms.tencent import SmsSendError, send_template_sms
from app.services.user_activity import local_activity_date

logger = logging.getLogger(__name__)

_WILL_PREVIEW_MAX = 20


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


def _as_local_date(raw: Any, today: date) -> date | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return local_activity_date(raw)
    if isinstance(raw, date):
        return raw
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return today


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


def _effective_last_active(row: Any, today: date) -> date:
    """Last login day, floored at will activation day (started_at).

    Without the floor, a user who had been idle before confirming the will could
    trigger immediately on activation — the countdown must start when they arm it.
    """
    last_active = _last_active_date(row, today)
    started_day = _as_local_date(_field(row, "startedAt"), today)
    if started_day is not None:
        last_active = max(last_active, started_day)
    return last_active


def _last_will_sms_configured() -> bool:
    if not settings.last_will_sms_enabled:
        return False
    if not settings.sms_enabled:
        return False
    if settings.sms_mock_enabled and not settings.is_production():
        return True
    return bool(
        settings.tencent_sms_secret_id.strip()
        and settings.tencent_sms_secret_key.strip()
        and settings.tencent_sms_sdk_app_id.strip()
        and settings.tencent_sms_sign_name.strip()
        and settings.tencent_sms_last_will_template_id.strip()
    )


def _will_preview(content: str) -> str:
    text = (content or "").replace("\n", " ").strip()
    if len(text) <= _WILL_PREVIEW_MAX:
        return text
    return text[: _WILL_PREVIEW_MAX - 1] + "…"


async def _send_last_will_sms_to_phone(
    phone: str,
    *,
    contact_name: str,
    inactivity_days: int,
    content: str,
) -> None:
    preview = _will_preview(content) or "（无正文）"
    template_params = [contact_name, str(inactivity_days), preview]
    if settings.sms_mock_enabled and not settings.is_production():
        logger.info(
            "[SMS-MOCK] last_will to %s****%s params=%s",
            phone[:3],
            phone[-4:],
            template_params,
            extra={"event": "last_will_sms_mock"},
        )
        return
    await send_template_sms(
        phone,
        template_id=settings.tencent_sms_last_will_template_id.strip(),
        template_params=template_params,
    )


async def send_test_last_will_sms(phone: str) -> dict[str, str | bool]:
    """Admin-only smoke test using the production last-will SMS template."""
    if not _last_will_sms_configured():
        raise RuntimeError("sms_not_configured")
    normalized = normalize_cn_phone(phone)
    if not normalized:
        raise ValueError("invalid_phone")
    await _send_last_will_sms_to_phone(
        normalized,
        contact_name="测试联系人",
        inactivity_days=30,
        content="这是一条测试留言，请忽略。",
    )
    return {
        "phone_tail": normalized[-4:],
        "mock": settings.sms_mock_enabled and not settings.is_production(),
    }


async def scan_due_last_wills(now: datetime | None = None, *, limit: int = 500) -> dict[str, int]:
    """Mark active last wills as triggered after enough missed login days.

    Creates pending phone delivery rows per contact; ``dispatch_pending_last_will_deliveries``
    sends SMS via the same Tencent Cloud path as login verification codes.
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
            lw.started_at AS "startedAt",
            u.last_seen_at AS "userLastSeenAt",
            u.updated_at AS "userUpdatedAt",
            u.created_at AS "userCreatedAt",
            MAX(uda.local_date) AS "lastActivityDate"
        FROM last_wills lw
        JOIN users u ON u.id = lw.user_id
        LEFT JOIN user_daily_activity uda ON uda.user_id = lw.user_id
        WHERE lw.status = 'active'
          AND lw.triggered_at IS NULL
          AND lw.inactivity_days BETWEEN 1 AND 365
          AND btrim(lw.content) <> ''
          AND jsonb_typeof(lw.contacts) = 'array'
          AND jsonb_array_length(lw.contacts) > 0
          AND lw.started_at IS NOT NULL
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
        last_active = _effective_last_active(row, today)
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

        created = 0
        if _last_will_sms_configured():
            created = await _create_pending_deliveries(will_id, contacts)
        else:
            logger.info(
                "[last_will] SMS disabled, triggered without delivery id=%s",
                will_id,
            )
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
        phone = contact.get("phone")
        if not phone:
            continue
        inserted = await db.query_raw(
            """
            INSERT INTO last_will_deliveries (
                id, last_will_id, channel, contact, dedupe_key, status,
                created_at, updated_at
            )
            VALUES (
                gen_random_uuid(), $1, 'phone', $2::jsonb, $3, 'pending',
                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
            )
            ON CONFLICT (last_will_id, channel, dedupe_key) DO NOTHING
            RETURNING id
            """,
            will_id,
            json.dumps(protect_contact(contact), ensure_ascii=False),
            _delivery_dedupe_key("phone", str(phone)),
        )
        if inserted:
            created += 1
    return created


def _delivery_dedupe_key(channel: str, value: str) -> str:
    raw = f"{channel}:{value.strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def dispatch_pending_last_will_deliveries(
    *,
    limit: int = 50,
) -> dict[str, int]:
    """Send SMS for pending phone deliveries and mark rows sent/failed."""
    if not _last_will_sms_configured():
        return {"claimed": 0, "sent": 0, "failed": 0, "skipped": 0}

    rows = await db.query_raw(
        """
        SELECT
            d.id,
            d.last_will_id AS "lastWillId",
            d.contact,
            lw.content,
            lw.inactivity_days AS "inactivityDays"
        FROM last_will_deliveries d
        JOIN last_wills lw ON lw.id = d.last_will_id
        WHERE d.status = 'pending'
          AND d.channel = 'phone'
        ORDER BY d.created_at ASC
        LIMIT $1
        """,
        limit,
    )

    sent = 0
    failed = 0
    skipped = 0
    wills_completed: set[str] = set()

    for row in rows:
        delivery_id = str(_field(row, "id"))
        will_id = str(_field(row, "lastWillId"))

        try:
            contact = LastWillContact.model_validate(
                reveal_contact(_field(row, "contact"))
            )
        except Exception as exc:
            if await _finalize_delivery(delivery_id, success=False, error=f"invalid_contact:{type(exc).__name__}"):
                failed += 1
            else:
                skipped += 1
            continue

        phone = normalize_cn_phone(contact.phone or "")
        if not phone:
            if await _finalize_delivery(delivery_id, success=False, error="invalid_phone"):
                failed += 1
            else:
                skipped += 1
            continue

        inactivity_days = int(_field(row, "inactivityDays") or 0)
        try:
            await _send_last_will_sms_to_phone(
                phone,
                contact_name=contact.name,
                inactivity_days=inactivity_days,
                content=reveal_text(_field(row, "content")),
            )
        except SmsSendError as exc:
            if await _finalize_delivery(delivery_id, success=False, error=str(exc)[:180]):
                failed += 1
            else:
                skipped += 1
            logger.warning(
                "[last_will] SMS failed delivery=%s will=%s: %s",
                delivery_id,
                will_id,
                exc,
            )
            continue

        if not await _finalize_delivery(delivery_id, success=True):
            skipped += 1
            continue
        sent += 1
        wills_completed.add(will_id)
        logger.info(
            "[last_will] SMS sent delivery=%s will=%s phone_tail=%s",
            delivery_id,
            will_id,
            phone[-4:],
            extra={"event": "last_will_sms_sent", "will_id": will_id},
        )

    for will_id in wills_completed:
        await _maybe_mark_will_delivered(will_id)

    return {
        "claimed": len(rows),
        "sent": sent,
        "failed": failed,
        "skipped": skipped,
    }


async def _finalize_delivery(delivery_id: str, *, success: bool, error: str = "") -> bool:
    """Atomically move one row out of pending; returns False if already handled."""
    if success:
        rows = await db.query_raw(
            """
            UPDATE last_will_deliveries
            SET status = 'sent', error = NULL, updated_at = CURRENT_TIMESTAMP
            WHERE id = $1 AND status = 'pending'
            RETURNING id
            """,
            delivery_id,
        )
    else:
        rows = await db.query_raw(
            """
            UPDATE last_will_deliveries
            SET status = 'failed', error = $2, updated_at = CURRENT_TIMESTAMP
            WHERE id = $1 AND status = 'pending'
            RETURNING id
            """,
            delivery_id,
            error,
        )
    return bool(rows)


async def _maybe_mark_will_delivered(will_id: str) -> None:
    """Set delivered_at when every phone delivery for the will has left pending."""
    rows = await db.query_raw(
        """
        SELECT COUNT(*) FILTER (WHERE status = 'pending') AS pending
        FROM last_will_deliveries
        WHERE last_will_id = $1 AND channel = 'phone'
        """,
        will_id,
    )
    pending = int((rows[0] or {}).get("pending", 0) if rows else 0)
    if pending:
        return
    await db.execute_raw(
        """
        UPDATE last_wills
        SET delivered_at = COALESCE(delivered_at, CURRENT_TIMESTAMP),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1 AND status = 'triggered'
        """,
        will_id,
    )
