"""Atomic admission of a location card on the existing Redis chat transport.

This does not activate SQL chat consumers/checkpoints. Only message, quota and
location follow-up admission share a transaction; replies use the existing path.
"""

import uuid
from prisma import Json

from app.db import db
from app.services import wallet
from app.services.location_memory import location_place
from app.services.runtime.chat_entrypoint import parse_client_chat_input
from app.services.runtime.chat_ingress import ChatRequestConflict
from app.services.runtime.chat_ingress_effects import (
    ChatIngressQuotaBlocked,
    ChatIngressResourceInvalid,
)
from app.services.runtime.execution_scope import (
    bind_conversation_scope,
    scoped_transaction,
)
from app.services.vip.chat_quota import consume_one_in_transaction


def location_request(*, text, card, client_id, attachment_ids):
    location_place(card)
    return parse_client_chat_input(
        {
            "message": text,
            "component_card": card,
            "client_id": client_id or str(uuid.uuid4()),
            "attachments": attachment_ids or [],
        }
    ).request


async def previous_location_message(database, *, conversation_id, user_id, request):
    rows = await database.query_raw(
        """SELECT m.id,m.metadata FROM messages m JOIN conversations c ON c.id=m.conversation_id
        WHERE m.conversation_id=$1 AND c.user_id=$2 AND NOT c.is_deleted
        AND m.role='user' AND m.metadata->'component_card'->>'type'='location'
        AND m.metadata->>'client_id'=$3 LIMIT 1""",
        conversation_id,
        user_id,
        request.client_id,
    )
    if not rows:
        return None
    if rows[0]["metadata"].get("location_input_fingerprint") != request.fingerprint:
        raise ChatRequestConflict()
    return rows[0]["id"]


async def accept_location_message(
    *, conversation_id, user_id, request, content, metadata, paid_confirmed
):
    scope = await bind_conversation_scope(
        actor_user_id=user_id, conversation_id=conversation_id, database=db
    )
    async with scoped_transaction(scope, database=db) as tx:
        # Serialize simultaneous retries of this user's conversation; the
        # durable client ID/fingerprint is checked before charging or writing.
        await tx.execute_raw(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 78216015))",
            conversation_id,
        )
        previous = await previous_location_message(
            tx, conversation_id=conversation_id, user_id=user_id, request=request
        )
        if previous:
            return previous, False
        taken_over = await tx.query_raw(
            "SELECT status FROM human_takeovers WHERE user_id=$1 FOR SHARE", user_id
        )
        if taken_over and taken_over[0]["status"] == "active":
            raise ChatIngressResourceInvalid()
        metadata = {
            **metadata,
            "client_id": request.client_id,
            "location_input_fingerprint": request.fingerprint,
        }
        saved = await tx.message.create(
            data={
                "conversationId": conversation_id,
                "role": "user",
                "content": content,
                "metadata": Json(metadata),
            }
        )
        ids = [item["id"] for item in metadata.get("attachments", [])]
        if ids:
            count = await tx.execute_raw(
                """UPDATE chat_message_attachments SET message_id=$1,updated_at=NOW()
                WHERE id=ANY($2::text[]) AND user_id=$3 AND conversation_id=$4 AND message_id IS NULL""",
                saved.id,
                ids,
                user_id,
                conversation_id,
            )
            if count != len(ids):
                raise ChatIngressResourceInvalid()
        quota = await consume_one_in_transaction(
            user_id,
            is_vip=await wallet.is_vip(user_id, client=tx),
            paid_confirmed=paid_confirmed,
            client=tx,
            source_id=saved.id,
        )
        if not quota["allowed"]:
            raise ChatIngressQuotaBlocked(quota)
    return saved.id, True
