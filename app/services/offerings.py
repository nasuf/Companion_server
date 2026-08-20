"""User-to-agent offerings: red packets now, virtual gifts later.

Ticket amounts are shop 钞票. For red packets the companion perceives
1 钞票 as 1 RMB; that mapping is prompt-only and never shown on user UI.
Structured rows here are source of truth. memories_user/ai 生活/馈赠
rows exist so the agent can recall the gesture in later turns.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from app.db import db
from app.observability.events import EVT_OFFERING_RECEIVED, EVT_OFFERING_SENT
from app.services import wallet
from app.services.memory.provenance import AI_AUTHORED, USER_STATED
from app.services.memory.storage.persistence import store_memory
from app.services.prompting.store import get_prompt_text_or_default
from app.services.prompting.utils import render_template
from app.services.runtime.tasks import fire_background

logger = logging.getLogger(__name__)

KIND_RED_PACKET = "red_packet"
STATUS_SENT = "sent"
STATUS_RECEIVED = "received"
RED_PACKET_ACCENT = "#FF4D5F"
MAX_TICKET_AMOUNT = 1_000_000
MEMORY_IMPORTANCE = 0.72
MEMORY_LEVEL = 2


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
    }


def public_offering(offering: dict[str, Any]) -> dict[str, Any]:
    return {
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


def build_red_packet_card(offering: dict[str, Any]) -> dict[str, Any]:
    received = offering.get("status") == STATUS_RECEIVED
    return {
        "version": 1,
        "type": "red_packet",
        "title": "红包",
        "subtitle": "已领取" if received else "待领取",
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


def reply_context_payload(offering: dict[str, Any]) -> dict[str, Any]:
    return {
        "offering_id": offering["id"],
        "ticket_amount": offering["ticket_amount"],
        "agent_value_yuan": offering["agent_value_yuan"],
        "offering_count": int(offering.get("offering_count") or 1),
        "previous_summary": str(offering.get("previous_summary") or ""),
        "blessing": str(offering.get("blessing") or ""),
        "agent_id": offering["agent_id"],
        "conversation_id": offering.get("conversation_id"),
        "message_id": offering.get("message_id"),
    }


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
) -> tuple[int, str]:
    rows = await client.query_raw(
        """
        SELECT ticket_amount, created_at
        FROM user_offerings
        WHERE user_id = $1 AND agent_id = $2 AND kind = $3
        ORDER BY created_at DESC
        LIMIT 1
        """,
        user_id,
        agent_id,
        KIND_RED_PACKET,
    )
    count_rows = await client.query_raw(
        """
        SELECT COUNT(*) AS n
        FROM user_offerings
        WHERE user_id = $1 AND agent_id = $2 AND kind = $3
        """,
        user_id,
        agent_id,
        KIND_RED_PACKET,
    )
    previous_count = int(_field(count_rows[0], "n", 0) or 0) if count_rows else 0
    summary = ""
    if rows:
        amount = int(_field(rows[0], "ticket_amount", 0) or 0)
        when = _iso(_field(rows[0], "created_at")) or ""
        summary = f"{when[:10]} 发过 {amount} 钞票".strip()
    return previous_count, summary


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
    await wallet.ensure_wallet(user_id)
    offering_id = str(uuid.uuid4())
    cleaned_blessing = (blessing or "").strip()[:40] or None
    agent_value = ticket_amount

    async with db.tx() as tx:
        balance = await wallet.debit_tickets(
            user_id,
            ticket_amount,
            source="red_packet",
            source_id=offering_id,
            metadata={"kind": KIND_RED_PACKET, "agent_id": conv["agent_id"]},
            client=tx,
        )
        previous_count, previous_summary = await _previous_context(
            tx, user_id=user_id, agent_id=conv["agent_id"],
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
    offering["agent_name"] = conv["agent_name"]
    offering["workspace_id"] = conv["workspace_id"]
    logger.info(
        "offering sent kind=%s amount=%s",
        KIND_RED_PACKET,
        ticket_amount,
        extra={
            "event": EVT_OFFERING_SENT,
            "offering_id": offering_id,
            "ticket_amount": ticket_amount,
            "agent_id": conv["agent_id"],
        },
    )
    fire_background(_write_offering_memories(offering))
    return {
        "offering": offering,
        "component_card": build_red_packet_card(offering),
        "wallet": balance,
    }


async def get_red_packet(
    *,
    offering_id: str,
    user_id: str,
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
    return {
        "offering": offering,
        "component_card": build_red_packet_card(offering),
    }


async def authorize_red_packet_card(
    component_card: dict | None,
    *,
    user_id: str,
    agent_id: str,
    conversation_id: str,
) -> dict | None:
    """Replace a client red-packet card with the authoritative server card."""
    if not component_card or component_card.get("type") != "red_packet":
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
    if offering["kind"] != KIND_RED_PACKET:
        raise ValueError("offering_forbidden")
    card = build_red_packet_card(offering)
    card["_offering"] = offering
    return card


async def bind_red_packet_message(
    *,
    offering_id: str,
    message_id: str,
    user_id: str,
) -> dict[str, Any]:
    rows = await db.query_raw(
        """
        UPDATE user_offerings
        SET message_id = $2
        WHERE id = $1 AND user_id = $3 AND message_id IS NULL
        RETURNING id, user_id, agent_id, conversation_id, message_id, kind,
                  ticket_amount, agent_value_yuan, status, blessing, metadata,
                  created_at, received_at
        """,
        offering_id,
        message_id,
        user_id,
    )
    if not rows:
        raise ValueError("offering_already_bound")
    return _offering_from_row(rows[0])


async def build_red_packet_user_message(offering: dict[str, Any]) -> str:
    tpl = await get_prompt_text_or_default("chat.red_packet_user_message")
    return render_template(
        tpl,
        {
            "ticket_amount": offering["ticket_amount"],
            "agent_value_yuan": offering["agent_value_yuan"],
            "offering_count": int(offering.get("offering_count") or 1),
            "previous_summary": str(offering.get("previous_summary") or ""),
            "blessing": str(offering.get("blessing") or ""),
        },
        optional_keys=["previous_summary", "blessing"],
    )


async def mark_red_packet_received(
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
                "component_card": build_red_packet_card(offering),
            }

        offering = _offering_from_row(rows[0])
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
    card = build_red_packet_card(offering)
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
    logger.info(
        "offering received kind=%s amount=%s",
        KIND_RED_PACKET,
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
            "red_packet",
            {
                "offering_id": offering_id,
                "message_id": message_id,
                "status": STATUS_RECEIVED,
                "component_card": card,
            },
        )
    except Exception:
        logger.warning("red_packet ws emit failed offering=%s", offering_id[:8])
    return {"offering": offering, "component_card": card}


async def _write_offering_memories(offering: dict[str, Any]) -> None:
    amount = offering["ticket_amount"]
    agent_name = str(offering.get("agent_name") or "对方")
    workspace_id = offering.get("workspace_id")
    user_id = offering["user_id"]
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
