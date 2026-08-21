"""User-to-agent offerings: red packets and backpack gifts.

Ticket amounts are shop 钞票. For red packets the companion perceives
1 钞票 as 1 RMB; that mapping is prompt-only and never shown on user UI.
For gifts, ticket_amount/agent_value_yuan store the catalog list price
(积分) as the same perceived-yuan weight — inventory is consumed, 钞票
are not. Structured rows here are source of truth. memories_user/ai
生活/馈赠 rows exist so the agent can recall the gesture in later turns.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from app.db import db
from app.observability.events import (
    EVT_OFFERING_RECEIVED,
    EVT_OFFERING_RECLAIMED,
    EVT_OFFERING_SENT,
)
from app.services import wallet
from app.services.memory.provenance import AI_AUTHORED, USER_STATED
from app.services.memory.storage.persistence import store_memory
from app.services.prompting.store import get_prompt_text_or_default
from app.services.prompting.utils import render_template
from app.services.runtime.tasks import fire_background
from app.services.store_catalog import EXCHANGE_PRODUCTS
from app.services.store_inventory import add_inventory, consume_inventory

logger = logging.getLogger(__name__)

KIND_RED_PACKET = "red_packet"
KIND_GIFT = "gift"
STATUS_SENT = "sent"
STATUS_RECEIVED = "received"
RED_PACKET_ACCENT = "#FF4D5F"
GIFT_ACCENT = "#FF8A3D"
MAX_TICKET_AMOUNT = 1_000_000
MEMORY_IMPORTANCE = 0.72
MEMORY_LEVEL = 2
# HTTP send consumes inventory/tickets before the chat card is bound. If WS
# never binds, reclaim after this grace so the backpack/wallet can send again.
UNBOUND_TTL_SECONDS = 120
_OFFERING_RETURNING = """
    id, user_id, agent_id, conversation_id, message_id, kind,
    ticket_amount, agent_value_yuan, status, blessing, metadata,
    created_at, received_at
"""


def _field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


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


def _iso(value: Any) -> str | None:
    dt = _as_aware_dt(value)
    if dt is None:
        return None
    return dt.isoformat()


def _json_meta(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _offering_from_row(row: Any) -> dict[str, Any]:
    meta = _json_meta(_field(row, "metadata"))
    return {
        "id": str(_field(row, "id", "")),
        "user_id": str(_field(row, "user_id", "")),
        "agent_id": str(_field(row, "agent_id", "")),
        "conversation_id": _field(row, "conversation_id"),
        "message_id": _field(row, "message_id"),
        "kind": str(_field(row, "kind", KIND_RED_PACKET)),
        "ticket_amount": int(_field(row, "ticket_amount", 0) or 0),
        "agent_value_yuan": int(_field(row, "agent_value_yuan", 0) or 0),
        "status": str(_field(row, "status", STATUS_SENT)),
        "blessing": _field(row, "blessing"),
        "created_at": _iso(_field(row, "created_at")) or "",
        "received_at": _iso(_field(row, "received_at")),
        "offering_count": int(meta.get("offering_count") or 1),
        "previous_summary": str(meta.get("previous_summary") or ""),
        "agent_name": str(meta.get("agent_name") or ""),
        "workspace_id": meta.get("workspace_id"),
        "product_kind": str(meta.get("product_kind") or ""),
        "product_title": str(meta.get("product_title") or ""),
        "product_subcategory": str(meta.get("product_subcategory") or ""),
        "product_asset_key": str(meta.get("product_asset_key") or ""),
    }


def public_offering(offering: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "id": offering["id"],
        "kind": offering["kind"],
        "ticket_amount": offering["ticket_amount"],
        "agent_value_yuan": offering["agent_value_yuan"],
        "status": offering["status"],
        "blessing": offering.get("blessing"),
        "conversation_id": offering.get("conversation_id"),
        "message_id": offering.get("message_id"),
        "agent_id": offering["agent_id"],
        "created_at": offering.get("created_at") or "",
        "received_at": offering.get("received_at"),
    }
    if offering.get("kind") == KIND_GIFT:
        payload.update({
            "product_kind": offering.get("product_kind") or "",
            "product_title": offering.get("product_title") or "",
            "product_subcategory": offering.get("product_subcategory") or "",
            "product_asset_key": offering.get("product_asset_key") or "",
        })
    return payload


def build_red_packet_card(offering: dict[str, Any]) -> dict[str, Any]:
    received = offering.get("status") == STATUS_RECEIVED
    return {
        "version": 1,
        "type": "red_packet",
        "title": "红包",
        "subtitle": "",
        "body": "给你的一点心意",
        "footer": "点击查看",
        "accent": RED_PACKET_ACCENT,
        "payload": {
            "offering_id": offering["id"],
            "kind": KIND_RED_PACKET,
            "ticket_amount": offering["ticket_amount"],
            "agent_value_yuan": offering["agent_value_yuan"],
            "status": offering["status"],
            "status_label": "已领取" if received else "待领取",
            "created_at": offering.get("created_at") or "",
            "received_at": offering.get("received_at") or "",
            "agent_id": offering["agent_id"],
        },
    }


def build_gift_card(offering: dict[str, Any]) -> dict[str, Any]:
    received = offering.get("status") == STATUS_RECEIVED
    title = str(offering.get("product_title") or "礼物")
    subcategory = str(offering.get("product_subcategory") or "心意")
    return {
        "version": 1,
        "type": "gift",
        "title": title,
        "subtitle": "",
        "body": subcategory,
        "footer": "点击查看",
        "accent": GIFT_ACCENT,
        "payload": {
            "offering_id": offering["id"],
            "kind": KIND_GIFT,
            "product_kind": str(offering.get("product_kind") or ""),
            "product_title": title,
            "product_subcategory": subcategory,
            "product_asset_key": str(offering.get("product_asset_key") or ""),
            "ticket_amount": offering["ticket_amount"],
            "agent_value_yuan": offering["agent_value_yuan"],
            "status": offering["status"],
            "status_label": "已接收" if received else "待接收",
            "created_at": offering.get("created_at") or "",
            "received_at": offering.get("received_at") or "",
            "agent_id": offering["agent_id"],
        },
    }


def build_received_notice(offering: dict[str, Any]) -> dict[str, Any]:
    """WeChat-style centered system notice after the companion accepts."""
    agent_name = str(offering.get("agent_name") or "对方").strip() or "对方"
    if offering.get("kind") == KIND_GIFT:
        text = f"{agent_name}收下了你的礼物"
    else:
        text = f"{agent_name}领取了你的红包"
    return {
        "text": text,
        "kind": offering.get("kind") or KIND_RED_PACKET,
        "offering_id": offering["id"],
        "agent_name": agent_name,
    }


def build_offering_card(offering: dict[str, Any]) -> dict[str, Any]:
    if offering.get("kind") == KIND_GIFT:
        return build_gift_card(offering)
    return build_red_packet_card(offering)


def reply_context_payload(offering: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "offering_id": offering["id"],
        "kind": offering.get("kind") or KIND_RED_PACKET,
        "ticket_amount": offering["ticket_amount"],
        "agent_value_yuan": offering["agent_value_yuan"],
        "offering_count": int(offering.get("offering_count") or 1),
        "previous_summary": str(offering.get("previous_summary") or ""),
        "blessing": str(offering.get("blessing") or ""),
        "agent_id": offering["agent_id"],
        "conversation_id": offering.get("conversation_id"),
        "message_id": offering.get("message_id"),
    }
    if offering.get("kind") == KIND_GIFT:
        payload.update({
            "product_kind": str(offering.get("product_kind") or ""),
            "product_title": str(offering.get("product_title") or ""),
            "product_subcategory": str(offering.get("product_subcategory") or ""),
        })
    return payload


async def _load_conversation(conversation_id: str, user_id: str) -> dict[str, Any]:
    rows = await db.query_raw(
        """
        SELECT c.id, c.user_id, c.agent_id, c.workspace_id, a.name AS agent_name
        FROM conversations c
        JOIN ai_agents a ON a.id = c.agent_id
        WHERE c.id = $1 AND c.user_id = $2 AND c.is_deleted = FALSE
        LIMIT 1
        """,
        conversation_id,
        user_id,
    )
    if not rows:
        raise ValueError("conversation_not_found")
    row = rows[0]
    return {
        "id": str(_field(row, "id", "")),
        "user_id": str(_field(row, "user_id", "")),
        "agent_id": str(_field(row, "agent_id", "")),
        "workspace_id": _field(row, "workspace_id"),
        "agent_name": str(_field(row, "agent_name", "") or ""),
    }


async def _previous_context(
    client: Any,
    *,
    user_id: str,
    agent_id: str,
    kind: str = KIND_RED_PACKET,
) -> tuple[int, str]:
    rows = await client.query_raw(
        """
        SELECT ticket_amount, created_at, metadata
        FROM user_offerings
        WHERE user_id = $1 AND agent_id = $2 AND kind = $3
        ORDER BY created_at DESC
        LIMIT 1
        """,
        user_id,
        agent_id,
        kind,
    )
    count_rows = await client.query_raw(
        """
        SELECT COUNT(*) AS n
        FROM user_offerings
        WHERE user_id = $1 AND agent_id = $2 AND kind = $3
        """,
        user_id,
        agent_id,
        kind,
    )
    previous_count = int(_field(count_rows[0], "n", 0) or 0) if count_rows else 0
    summary = ""
    if rows:
        when = _iso(_field(rows[0], "created_at")) or ""
        day = when[:10]
        if kind == KIND_GIFT:
            meta = _json_meta(_field(rows[0], "metadata"))
            title = str(meta.get("product_title") or "礼物")
            summary = f"{day} 送过 {title}".strip()
        else:
            amount = int(_field(rows[0], "ticket_amount", 0) or 0)
            summary = f"{day} 发过 {amount} 钞票".strip()
    return previous_count, summary


def _is_stale_unbound(offering: dict[str, Any], now: datetime) -> bool:
    created = _as_aware_dt(offering.get("created_at"))
    if created is None:
        return True
    return (now - created) >= timedelta(seconds=UNBOUND_TTL_SECONDS)


async def _lock_unbound_offerings(client: Any, user_id: str) -> list[dict[str, Any]]:
    rows = await client.query_raw(
        f"""
        SELECT {_OFFERING_RETURNING}
        FROM user_offerings
        WHERE user_id = $1 AND message_id IS NULL AND status = $2
        FOR UPDATE
        """,
        user_id,
        STATUS_SENT,
    )
    return [_offering_from_row(row) for row in rows]


async def _inventory_snapshot(
    client: Any,
    user_id: str,
    product_kind: str,
) -> dict[str, Any]:
    rows = await client.query_raw(
        """
        SELECT product_kind, quantity, acquired_at, updated_at
        FROM user_store_inventory
        WHERE user_id = $1 AND product_kind = $2
        LIMIT 1
        """,
        user_id,
        product_kind,
    )
    if not rows:
        return {
            "product_kind": product_kind,
            "quantity": 0,
            "acquired_at": None,
            "updated_at": None,
        }
    from app.services.store_inventory import _inventory_row

    return _inventory_row(rows[0])


async def _delete_and_restore(client: Any, offering: dict[str, Any]) -> bool:
    """Return inventory/tickets only after the unbound row is actually deleted.

    Delete first so a concurrent bind that lost the row lock cannot be
    refunded: UPDATE bind waits, then sees zero rows.
    """
    deleted = await client.query_raw(
        """
        DELETE FROM user_offerings
        WHERE id = $1 AND user_id = $2 AND message_id IS NULL AND status = $3
        RETURNING id
        """,
        offering["id"],
        offering["user_id"],
        STATUS_SENT,
    )
    if not deleted:
        return False
    if offering.get("kind") == KIND_GIFT:
        product_kind = str(offering.get("product_kind") or "")
        if product_kind:
            await add_inventory(
                offering["user_id"], product_kind, quantity=1, client=client,
            )
    else:
        amount = int(offering.get("ticket_amount") or 0)
        if amount > 0:
            await wallet.credit_tickets(
                offering["user_id"],
                amount,
                source="red_packet_unbound_refund",
                source_id=offering["id"],
                metadata={"kind": KIND_RED_PACKET, "reason": "unbound_reclaim"},
                client=client,
            )
    logger.info(
        "offering reclaimed kind=%s",
        offering.get("kind"),
        extra={
            "event": EVT_OFFERING_RECLAIMED,
            "offering_id": offering["id"],
            "kind": offering.get("kind"),
            "ticket_amount": offering.get("ticket_amount"),
            "product_kind": offering.get("product_kind") or "",
        },
    )
    return True


async def _retarget_unbound_offering(
    client: Any,
    offering: dict[str, Any],
    conv: dict[str, Any],
) -> dict[str, Any] | None:
    """Point a reused unbound row at the conversation the user is sending in."""
    if (
        offering.get("conversation_id") == conv["id"]
        and offering.get("agent_id") == conv["agent_id"]
    ):
        return offering
    meta: dict[str, Any] = {
        "offering_count": int(offering.get("offering_count") or 1),
        "previous_summary": str(offering.get("previous_summary") or ""),
        "agent_name": conv["agent_name"],
        "workspace_id": conv.get("workspace_id"),
    }
    if offering.get("kind") == KIND_GIFT:
        meta.update({
            "product_kind": str(offering.get("product_kind") or ""),
            "product_title": str(offering.get("product_title") or ""),
            "product_subcategory": str(offering.get("product_subcategory") or ""),
            "product_asset_key": str(offering.get("product_asset_key") or ""),
        })
    rows = await client.query_raw(
        f"""
        UPDATE user_offerings
        SET conversation_id = $2,
            agent_id = $3,
            metadata = $4::jsonb
        WHERE id = $1 AND message_id IS NULL AND status = $5
        RETURNING {_OFFERING_RETURNING}
        """,
        offering["id"],
        conv["id"],
        conv["agent_id"],
        json.dumps(meta, ensure_ascii=False),
        STATUS_SENT,
    )
    if not rows:
        return None
    return _offering_from_row(rows[0])


async def _prepare_unbound_for_send(
    client: Any,
    *,
    user_id: str,
    conv: dict[str, Any],
    kind: str,
    product_kind: str | None = None,
    ticket_amount: int | None = None,
) -> dict[str, Any] | None:
    """Reclaim stale unbound rows, then reuse a fresh match for this send."""
    now = datetime.now(timezone.utc)
    matching: dict[str, Any] | None = None
    leftovers: list[dict[str, Any]] = []
    for item in await _lock_unbound_offerings(client, user_id):
        if _is_stale_unbound(item, now):
            await _delete_and_restore(client, item)
            continue
        same_kind = item.get("kind") == kind
        if kind == KIND_GIFT:
            is_match = (
                same_kind
                and str(item.get("product_kind") or "") == str(product_kind or "")
            )
        else:
            is_match = (
                same_kind
                and int(item.get("ticket_amount") or 0) == int(ticket_amount or 0)
            )
        if is_match and matching is None:
            retargeted = await _retarget_unbound_offering(client, item, conv)
            if retargeted is not None:
                matching = retargeted
            continue
        leftovers.append(item)
    if kind == KIND_RED_PACKET:
        for item in leftovers:
            if item.get("kind") == KIND_RED_PACKET:
                await _delete_and_restore(client, item)
    return matching


async def reclaim_stale_unbound_offerings(
    *,
    max_age_seconds: int = UNBOUND_TTL_SECONDS,
) -> int:
    """Cron sweep: restore backpack/tickets for cards that never hit chat."""
    async with db.tx() as tx:
        rows = await tx.query_raw(
            f"""
            SELECT {_OFFERING_RETURNING}
            FROM user_offerings
            WHERE message_id IS NULL
              AND status = $1
              AND created_at < CURRENT_TIMESTAMP - ($2 * INTERVAL '1 second')
            FOR UPDATE SKIP LOCKED
            """,
            STATUS_SENT,
            max_age_seconds,
        )
        reclaimed = 0
        for row in rows:
            if await _delete_and_restore(tx, _offering_from_row(row)):
                reclaimed += 1
        return reclaimed


async def send_red_packet(
    *,
    user_id: str,
    conversation_id: str,
    ticket_amount: int,
    blessing: str | None = None,
) -> dict[str, Any]:
    if ticket_amount < 1 or ticket_amount > MAX_TICKET_AMOUNT:
        raise ValueError("invalid_amount")
    conv = await _load_conversation(conversation_id, user_id)
    balance = await wallet.ensure_wallet(user_id)
    offering_id = str(uuid.uuid4())
    cleaned_blessing = (blessing or "").strip()[:40] or None
    agent_value = ticket_amount
    reused: dict[str, Any] | None = None

    async with db.tx() as tx:
        reused = await _prepare_unbound_for_send(
            tx,
            user_id=user_id,
            conv=conv,
            kind=KIND_RED_PACKET,
            ticket_amount=ticket_amount,
        )
        if reused is None:
            balance = await wallet.debit_tickets(
                user_id,
                ticket_amount,
                source="red_packet",
                source_id=offering_id,
                metadata={"kind": KIND_RED_PACKET, "agent_id": conv["agent_id"]},
                client=tx,
            )
            previous_count, previous_summary = await _previous_context(
                tx, user_id=user_id, agent_id=conv["agent_id"], kind=KIND_RED_PACKET,
            )
            metadata = {
                "offering_count": previous_count + 1,
                "previous_summary": previous_summary,
                "agent_name": conv["agent_name"],
                "workspace_id": conv.get("workspace_id"),
            }
            rows = await tx.query_raw(
                """
                INSERT INTO user_offerings (
                    id, user_id, agent_id, conversation_id, kind,
                    ticket_amount, agent_value_yuan, status, blessing, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
                RETURNING id, user_id, agent_id, conversation_id, message_id, kind,
                          ticket_amount, agent_value_yuan, status, blessing,
                          metadata, created_at, received_at
                """,
                offering_id,
                user_id,
                conv["agent_id"],
                conversation_id,
                KIND_RED_PACKET,
                ticket_amount,
                agent_value,
                STATUS_SENT,
                cleaned_blessing,
                json.dumps(metadata, ensure_ascii=False),
            )
            offering = _offering_from_row(rows[0])
        else:
            offering = reused

    offering["agent_name"] = conv["agent_name"]
    offering["workspace_id"] = conv["workspace_id"]
    if reused is None:
        logger.info(
            "offering sent kind=%s amount=%s",
            KIND_RED_PACKET,
            ticket_amount,
            extra={
                "event": EVT_OFFERING_SENT,
                "offering_id": offering["id"],
                "ticket_amount": ticket_amount,
                "agent_id": conv["agent_id"],
            },
        )
    else:
        # Leftover unbound packets may have been refunded inside the tx.
        balance = await wallet.ensure_wallet(user_id)
    return {
        "offering": offering,
        "component_card": build_red_packet_card(offering),
        "wallet": balance,
    }


async def send_gift(
    *,
    user_id: str,
    conversation_id: str,
    product_kind: str,
) -> dict[str, Any]:
    kind = str(product_kind or "").strip()
    product = EXCHANGE_PRODUCTS.get(kind)
    if product is None or product.category != "gift":
        raise ValueError("not_giftable")
    perceived_value = int(product.list_price)
    if perceived_value < 1 or perceived_value > MAX_TICKET_AMOUNT:
        raise ValueError("invalid_amount")
    conv = await _load_conversation(conversation_id, user_id)
    balance = await wallet.ensure_wallet(user_id)
    offering_id = str(uuid.uuid4())
    reused: dict[str, Any] | None = None

    async with db.tx() as tx:
        reused = await _prepare_unbound_for_send(
            tx,
            user_id=user_id,
            conv=conv,
            kind=KIND_GIFT,
            product_kind=kind,
        )
        if reused is None:
            inventory_item = await consume_inventory(
                user_id, kind, quantity=1, client=tx,
            )
            previous_count, previous_summary = await _previous_context(
                tx, user_id=user_id, agent_id=conv["agent_id"], kind=KIND_GIFT,
            )
            metadata = {
                "offering_count": previous_count + 1,
                "previous_summary": previous_summary,
                "agent_name": conv["agent_name"],
                "workspace_id": conv.get("workspace_id"),
                "product_kind": product.product_kind,
                "product_title": product.title,
                "product_subcategory": product.subcategory or "",
                "product_asset_key": product.asset_key or "",
            }
            rows = await tx.query_raw(
                """
                INSERT INTO user_offerings (
                    id, user_id, agent_id, conversation_id, kind,
                    ticket_amount, agent_value_yuan, status, blessing, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
                RETURNING id, user_id, agent_id, conversation_id, message_id, kind,
                          ticket_amount, agent_value_yuan, status, blessing,
                          metadata, created_at, received_at
                """,
                offering_id,
                user_id,
                conv["agent_id"],
                conversation_id,
                KIND_GIFT,
                perceived_value,
                perceived_value,
                STATUS_SENT,
                None,
                json.dumps(metadata, ensure_ascii=False),
            )
            offering = _offering_from_row(rows[0])
        else:
            offering = reused
            inventory_item = await _inventory_snapshot(tx, user_id, kind)

    offering["agent_name"] = conv["agent_name"]
    offering["workspace_id"] = conv["workspace_id"]
    if reused is None:
        logger.info(
            "offering sent kind=%s product=%s",
            KIND_GIFT,
            kind,
            extra={
                "event": EVT_OFFERING_SENT,
                "offering_id": offering["id"],
                "ticket_amount": perceived_value,
                "product_kind": kind,
                "agent_id": conv["agent_id"],
            },
        )
    return {
        "offering": offering,
        "component_card": build_gift_card(offering),
        "wallet": balance,
        "inventory_item": inventory_item,
    }


async def get_red_packet(
    *,
    offering_id: str,
    user_id: str,
) -> dict[str, Any]:
    return await get_offering(
        offering_id=offering_id,
        user_id=user_id,
        expected_kind=KIND_RED_PACKET,
    )


async def get_gift(
    *,
    offering_id: str,
    user_id: str,
) -> dict[str, Any]:
    return await get_offering(
        offering_id=offering_id,
        user_id=user_id,
        expected_kind=KIND_GIFT,
    )


async def get_offering(
    *,
    offering_id: str,
    user_id: str,
    expected_kind: str | None = None,
) -> dict[str, Any]:
    rows = await db.query_raw(
        """
        SELECT id, user_id, agent_id, conversation_id, message_id, kind,
               ticket_amount, agent_value_yuan, status, blessing, metadata,
               created_at, received_at
        FROM user_offerings
        WHERE id = $1 AND user_id = $2
        LIMIT 1
        """,
        offering_id,
        user_id,
    )
    if not rows:
        raise ValueError("offering_not_found")
    offering = _offering_from_row(rows[0])
    if expected_kind and offering["kind"] != expected_kind:
        raise ValueError("offering_not_found")
    return {
        "offering": offering,
        "component_card": build_offering_card(offering),
    }


async def authorize_red_packet_card(
    component_card: dict | None,
    *,
    user_id: str,
    agent_id: str,
    conversation_id: str,
) -> dict | None:
    """Replace a client red-packet card with the authoritative server card."""
    return await authorize_offering_card(
        component_card,
        user_id=user_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
        expected_type="red_packet",
        expected_kind=KIND_RED_PACKET,
    )


async def authorize_gift_card(
    component_card: dict | None,
    *,
    user_id: str,
    agent_id: str,
    conversation_id: str,
) -> dict | None:
    """Replace a client gift card with the authoritative server card."""
    return await authorize_offering_card(
        component_card,
        user_id=user_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
        expected_type="gift",
        expected_kind=KIND_GIFT,
    )


async def authorize_offering_card(
    component_card: dict | None,
    *,
    user_id: str,
    agent_id: str,
    conversation_id: str,
    expected_type: str,
    expected_kind: str,
) -> dict | None:
    if not component_card or component_card.get("type") != expected_type:
        return component_card
    payload = component_card.get("payload") if isinstance(component_card.get("payload"), dict) else {}
    offering_id = str(payload.get("offering_id") or "").strip()
    if not offering_id:
        raise ValueError("offering_not_found")
    rows = await db.query_raw(
        """
        SELECT id, user_id, agent_id, conversation_id, message_id, kind,
               ticket_amount, agent_value_yuan, status, blessing, metadata,
               created_at, received_at
        FROM user_offerings
        WHERE id = $1
        LIMIT 1
        """,
        offering_id,
    )
    if not rows:
        raise ValueError("offering_not_found")
    offering = _offering_from_row(rows[0])
    if offering["user_id"] != user_id:
        raise ValueError("offering_forbidden")
    if offering["agent_id"] != agent_id:
        raise ValueError("offering_forbidden")
    if offering.get("conversation_id") and offering["conversation_id"] != conversation_id:
        raise ValueError("offering_forbidden")
    if offering.get("message_id"):
        raise ValueError("offering_already_bound")
    if offering["kind"] != expected_kind:
        raise ValueError("offering_forbidden")
    card = build_offering_card(offering)
    card["_offering"] = offering
    return card


async def bind_red_packet_message(
    *,
    offering_id: str,
    message_id: str,
    user_id: str,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    return await bind_offering_message(
        offering_id=offering_id,
        message_id=message_id,
        user_id=user_id,
        conversation_id=conversation_id,
    )


async def bind_offering_message(
    *,
    offering_id: str,
    message_id: str,
    user_id: str,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    rows = await db.query_raw(
        """
        UPDATE user_offerings
        SET message_id = $2
        WHERE id = $1 AND user_id = $3 AND message_id IS NULL
          AND ($4::text IS NULL OR conversation_id = $4)
        RETURNING id, user_id, agent_id, conversation_id, message_id, kind,
                  ticket_amount, agent_value_yuan, status, blessing, metadata,
                  created_at, received_at
        """,
        offering_id,
        message_id,
        user_id,
        conversation_id,
    )
    if not rows:
        raise ValueError("offering_already_bound")
    bound = _offering_from_row(rows[0])
    fire_background(_write_offering_memories(bound))
    return bound


async def build_red_packet_user_message(offering: dict[str, Any]) -> str:
    tpl = await get_prompt_text_or_default("chat.red_packet_user_message")
    previous = str(offering.get("previous_summary") or "")
    return render_template(
        tpl,
        {
            "ticket_amount": offering["ticket_amount"],
            "agent_value_yuan": offering["agent_value_yuan"],
            "offering_count": int(offering.get("offering_count") or 1),
            "previous_summary": previous,
            "blessing": str(offering.get("blessing") or ""),
        },
        optional_keys=["previous_summary", "blessing"],
    )


async def build_gift_user_message(offering: dict[str, Any]) -> str:
    tpl = await get_prompt_text_or_default("chat.gift_user_message")
    previous = str(offering.get("previous_summary") or "")
    previous_line = f"上次：{previous}" if previous else ""
    return render_template(
        tpl,
        {
            "product_title": str(offering.get("product_title") or "礼物"),
            "product_subcategory": str(offering.get("product_subcategory") or ""),
            "agent_value_yuan": offering["agent_value_yuan"],
            "offering_count": int(offering.get("offering_count") or 1),
            "previous_summary": previous_line,
        },
        optional_keys=["previous_summary", "product_subcategory"],
    )


async def build_offering_user_message(offering: dict[str, Any]) -> str:
    if offering.get("kind") == KIND_GIFT:
        return await build_gift_user_message(offering)
    return await build_red_packet_user_message(offering)


async def mark_red_packet_received(
    *,
    offering_id: str,
    user_id: str,
    conversation_id: str,
) -> dict[str, Any] | None:
    return await mark_offering_received(
        offering_id=offering_id,
        user_id=user_id,
        conversation_id=conversation_id,
    )


async def mark_gift_received(
    *,
    offering_id: str,
    user_id: str,
    conversation_id: str,
) -> dict[str, Any] | None:
    return await mark_offering_received(
        offering_id=offering_id,
        user_id=user_id,
        conversation_id=conversation_id,
    )


async def mark_offering_received(
    *,
    offering_id: str,
    user_id: str,
    conversation_id: str,
) -> dict[str, Any] | None:
    async with db.tx() as tx:
        rows = await tx.query_raw(
            """
            UPDATE user_offerings
            SET status = $3, received_at = CURRENT_TIMESTAMP
            WHERE id = $1 AND user_id = $2 AND status = $4
            RETURNING id, user_id, agent_id, conversation_id, message_id, kind,
                      ticket_amount, agent_value_yuan, status, blessing, metadata,
                      created_at, received_at
            """,
            offering_id,
            user_id,
            STATUS_RECEIVED,
            STATUS_SENT,
        )
        if not rows:
            existing = await tx.query_raw(
                """
                SELECT id, user_id, agent_id, conversation_id, message_id, kind,
                       ticket_amount, agent_value_yuan, status, blessing, metadata,
                       created_at, received_at
                FROM user_offerings
                WHERE id = $1 AND user_id = $2
                LIMIT 1
                """,
                offering_id,
                user_id,
            )
            if not existing:
                return None
            offering = _offering_from_row(existing[0])
            return {
                "offering": offering,
                "component_card": build_offering_card(offering),
            }

        offering = _offering_from_row(rows[0])
        if offering["kind"] == KIND_RED_PACKET:
            await tx.execute_raw(
                """
                INSERT INTO agent_wallets (id, agent_id, user_id, received_tickets)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (agent_id) DO UPDATE
                SET received_tickets = agent_wallets.received_tickets + EXCLUDED.received_tickets,
                    updated_at = CURRENT_TIMESTAMP
                """,
                str(uuid.uuid4()),
                offering["agent_id"],
                user_id,
                offering["ticket_amount"],
            )
        notice = await _persist_received_notice(
            tx, offering, conversation_id=conversation_id,
        )
    card = build_offering_card(offering)
    message_id = offering.get("message_id")
    if message_id:
        await db.execute_raw(
            """
            UPDATE messages
            SET metadata = jsonb_set(
                COALESCE(metadata, '{}'::jsonb),
                '{component_card}',
                $2::jsonb
            )
            WHERE id = $1
            """,
            message_id,
            json.dumps(card, ensure_ascii=False),
        )
    event_name = "gift" if offering["kind"] == KIND_GIFT else "red_packet"
    logger.info(
        "offering received kind=%s amount=%s",
        offering["kind"],
        offering["ticket_amount"],
        extra={
            "event": EVT_OFFERING_RECEIVED,
            "offering_id": offering_id,
            "ticket_amount": offering["ticket_amount"],
            "agent_id": offering["agent_id"],
        },
    )
    try:
        from app.services.runtime.ws_manager import manager

        await manager.send_event(
            conversation_id,
            event_name,
            {
                "offering_id": offering_id,
                "message_id": message_id,
                "status": STATUS_RECEIVED,
                "component_card": card,
                "notice": notice,
            },
        )
    except Exception:
        logger.warning("%s ws emit failed offering=%s", event_name, offering_id[:8])
    return {"offering": offering, "component_card": card}


async def _persist_received_notice(
    client: Any,
    offering: dict[str, Any],
    *,
    conversation_id: str,
) -> dict[str, Any]:
    payload = build_received_notice(offering)
    cid = str(offering.get("conversation_id") or conversation_id or "")
    if not cid:
        return {**payload, "message_id": "", "created_at": ""}
    notice_id = str(uuid.uuid4())
    metadata = {
        "offering_received": True,
        "offering_kind": payload["kind"],
        "offering_id": payload["offering_id"],
        "agent_name": payload["agent_name"],
    }
    rows = await client.query_raw(
        """
        INSERT INTO messages (id, conversation_id, role, content, metadata)
        SELECT $1, $2, 'assistant', $3, $4::jsonb
        WHERE NOT EXISTS (
            SELECT 1 FROM messages
            WHERE conversation_id = $2
              AND COALESCE(metadata->>'offering_received', '') = 'true'
              AND metadata->>'offering_id' = $5
        )
        RETURNING id, created_at
        """,
        notice_id,
        cid,
        payload["text"],
        json.dumps(metadata, ensure_ascii=False),
        payload["offering_id"],
    )
    row = rows[0] if rows else None
    if row is None:
        existing = await client.query_raw(
            """
            SELECT id, created_at
            FROM messages
            WHERE conversation_id = $1
              AND COALESCE(metadata->>'offering_received', '') = 'true'
              AND metadata->>'offering_id' = $2
            ORDER BY created_at ASC
            LIMIT 1
            """,
            cid,
            payload["offering_id"],
        )
        row = existing[0] if existing else None
    return {
        **payload,
        "message_id": str(_field(row, "id", "") or ""),
        "created_at": (_iso(_field(row, "created_at")) or "") if row else "",
    }


async def _write_offering_memories(offering: dict[str, Any]) -> None:
    agent_name = str(offering.get("agent_name") or "对方")
    workspace_id = offering.get("workspace_id")
    user_id = offering["user_id"]
    if offering.get("kind") == KIND_GIFT:
        title = str(offering.get("product_title") or "礼物")
        user_text = f"我给{agent_name}送了{title}"
        ai_text = f"用户送给我一份{title}"
    else:
        amount = offering["ticket_amount"]
        user_text = f"我给{agent_name}发了{amount}钞票的红包"
        ai_text = f"用户给我发了{amount}钞票的红包"
    try:
        await store_memory(
            user_id,
            user_text,
            level=MEMORY_LEVEL,
            importance=MEMORY_IMPORTANCE,
            main_category="生活",
            sub_category="馈赠",
            source="user",
            workspace_id=workspace_id,
            provenance=USER_STATED,
        )
        await store_memory(
            user_id,
            ai_text,
            level=MEMORY_LEVEL,
            importance=MEMORY_IMPORTANCE,
            main_category="生活",
            sub_category="馈赠",
            source="ai",
            workspace_id=workspace_id,
            provenance=AI_AUTHORED,
        )
    except Exception:
        logger.exception(
            "offering memory write failed offering=%s",
            str(offering.get("id") or "")[:8],
        )
