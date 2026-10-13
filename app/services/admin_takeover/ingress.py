"""Short, scoped SQL commit for human chat; no graph/model/background writes.

The active case row serializes retries and resource binding. Quota and the
message commit together. A lost ACK can safely recover the same client ID.
"""
from __future__ import annotations

import json
import uuid

from prisma import Json
from dataclasses import replace

from app.services.runtime.chat_ingress_contracts import ChatRequestInput, canonical_object

from app.services.chat_media.prompt import attachment_to_metadata
from app.services.chat_media.repo import _attachment_from_row
from app.services.offerings import _offering_from_row, build_offering_card
from app.services.runtime.chat_entrypoint import parse_client_chat_input
from app.services.runtime.chat_ingress import ChatRequestConflict
from app.services.runtime.chat_ingress_effects import ChatIngressEffects, ChatIngressResourceInvalid
from app.services.runtime.execution_scope import (
    ExecutionScopeUnavailable, bind_conversation_scope, scoped_transaction,
)

# ChatRequestInput rejects a turn that has no text, attachments, or card.
# Music invites and other share cards often have an empty body. The stored
# message keeps that body; this stand-in exists only on the quota request
# after the card has been removed.
_QUOTA_STANDIN_TEXT = "［分享］"


def quota_request_without_card(request: ChatRequestInput, original: dict) -> ChatRequestInput:
    """Quota adapter sees an ordinary text turn, not the shared card."""
    stripped = {**original, "component_card": None}
    text = stripped.get("text")
    if (type(text) is not str or not text.strip()) and not stripped.get("attachment_ids"):
        stripped["text"] = _QUOTA_STANDIN_TEXT
    return replace(request, input_json=canonical_object(stripped))


async def persist_user_message(*, conversation_id: str, user_id: str, text: str,
                               metadata: dict, paid_confirmed: bool, database) -> dict:
    # Old SSE callers may omit an ID. Keep them usable with a server identity;
    # retry deduplication requires the stable ID sent by Web/Flutter clients.
    incoming = parse_client_chat_input({
        "message": text, "client_id": metadata.get("client_id") or str(uuid.uuid4()),
        "attachments": metadata.get("attachments", []),
        "component_card": metadata.get("component_card"), "paid_confirmed": paid_confirmed,
    })
    request = incoming.request
    original = json.loads(request.input_json)
    scope = await bind_conversation_scope(actor_user_id=user_id, conversation_id=conversation_id,
                                          database=database)
    if scope.owner_user_id != user_id:
        raise ExecutionScopeUnavailable()
    async with scoped_transaction(scope, database=database) as tx:
        rows = await tx.query_raw(
            "SELECT t.id,t.owner_admin_id AS \"ownerAdminId\",t.user_id AS \"userId\",t.agent_id AS \"agentId\", "
            "t.user_message_seq,COALESCE(NULLIF(u.display_name,''),u.username) AS user_name "
            "FROM human_takeovers t JOIN users u ON u.id=t.user_id WHERE t.conversation_id=$1 AND t.user_id=$2 "
            "AND t.status='active' FOR UPDATE OF t", conversation_id, user_id,
        )
        if len(rows) != 1:
            raise ExecutionScopeUnavailable()
        row = rows[0]
        previous = await tx.query_raw(
            "SELECT id,metadata FROM messages WHERE conversation_id=$1 AND role='user' "
            "AND metadata->>'takeover_request_key'=$2 LIMIT 1", conversation_id, request.request_key,
        )
        if previous:
            if previous[0]['metadata'].get('takeover_input_fingerprint') != request.fingerprint:
                raise ChatRequestConflict()
            return {"created": False, "message_id": previous[0]['id']}
        prepared = {"human_takeover": True, "client_id": request.client_id,
                    "takeover_sequence": row['user_message_seq'] + 1,
                    "takeover_request_key": request.request_key,
                    "takeover_input_fingerprint": request.fingerprint}
        ids = original['attachment_ids']
        if ids:
            attachments = await tx.query_raw(
                "SELECT * FROM chat_message_attachments WHERE id=ANY($1::text[]) "
                "AND user_id=$2 AND conversation_id=$3 AND message_id IS NULL",
                ids, user_id, conversation_id,
            )
            by_id = {item['id']: item for item in attachments}
            if set(by_id) != set(ids):
                raise ChatIngressResourceInvalid()
            prepared['attachments'] = [attachment_to_metadata(_attachment_from_row(by_id[ident])) for ident in ids]
        card = original['component_card']
        effect_request = request
        if card:
            kind = card.get('type')
            if kind in {'gift', 'red_packet'}:
                payload = card.get('payload')
                ident = payload.get('offering_id') if type(payload) is dict else None
                if type(ident) is not str:
                    raise ChatIngressResourceInvalid()
                offerings = await tx.query_raw(
                    "SELECT * FROM user_offerings WHERE id=$1 AND user_id=$2 AND agent_id=$3 "
                    "AND conversation_id=$4 AND kind=$5 AND status='sent' AND message_id IS NULL",
                    ident, user_id, scope.agent_id, conversation_id, kind,
                )
                if not offerings:
                    raise ChatIngressResourceInvalid()
                prepared['component_card'] = build_offering_card(_offering_from_row(offerings[0]))
            else:
                from app.api.realtime.ws import _sanitize_component_card
                sanitized = _sanitize_component_card(card)
                if sanitized is None:
                    raise ChatIngressResourceInvalid()
                prepared['component_card'] = sanitized
                # Nonfinancial sharing is delivered to the human without AI
                # automation, and pays ordinary quota. Preserve original retry
                # identity; only the trusted effects adapter sees no offering.
                effect_request = quota_request_without_card(request, original)
        message = await tx.message.create(data={
            "conversation": {"connect": {"id": conversation_id}}, "role": "user",
            "content": text.strip(), "metadata": Json(prepared),
        })
        effect_metadata = dict(prepared)
        if effect_request is not request:
            effect_metadata.pop("component_card")
        effects = await ChatIngressEffects(incoming.paid_confirmed).commit(tx, scope, effect_request, message.id, effect_metadata)
        prepared['ingress_effects'] = effects
        await tx.message.update(where={"id": message.id}, data={"metadata": Json(prepared)})
        result = await tx.query_raw(
            "UPDATE human_takeovers SET unread_count=unread_count+1,user_message_seq=user_message_seq+1,last_user_message_at=CURRENT_TIMESTAMP, "
            "last_message_preview=$2,updated_at=CURRENT_TIMESTAMP WHERE id=$1 "
            "RETURNING unread_count AS \"unreadCount\"", row['id'], ' '.join(text.split())[:80] or '［新消息］',
        )
        row.update(result[0])
        # The alert and message commit together. A process death after COMMIT
        # cannot lose the push; the existing dispatcher retries the outbox.
        from app.services.admin_takeover.service import _push_owner_message
        await _push_owner_message(row['ownerAdminId'], row['id'], conversation_id,
            message.id, ' '.join(text.split())[:80] or '［新消息］',
            user_name=row['user_name'], database=tx, dispatch_now=False)
    return {"created": True, "message_id": message.id, "row": row,
            "metadata": prepared, "created_at": message.createdAt.isoformat()}


async def persist_admin_message(*, takeover_id: str, admin_id: str, text: str,
                                component_card: dict | None, attachments: list[dict] | None,
                                client_id: str | None, database) -> dict:
    from app.services.admin_takeover.service import TakeoverClaimError

    if component_card or attachments or not text.strip() or len(text) > 4000:
        raise ValueError('Takeover replies support text only')

    request = parse_client_chat_input({
        'message': text, 'client_id': client_id or str(uuid.uuid4()),
        'attachments': attachments or [], 'component_card': component_card,
    }).request
    rows = await database.query_raw(
        "SELECT conversation_id FROM human_takeovers WHERE id=$1 AND owner_admin_id=$2 "
        "AND status='active'", takeover_id, admin_id,
    )
    if not rows:
        raise TakeoverClaimError('not_owner')
    scope = await bind_conversation_scope(actor_user_id=admin_id, conversation_id=rows[0]['conversation_id'],
                                          database=database)
    async with scoped_transaction(scope, database=database) as tx:
        rows = await tx.query_raw(
            "SELECT id FROM human_takeovers WHERE id=$1 AND owner_admin_id=$2 AND status='active' "
            "AND conversation_id=$3 FOR UPDATE", takeover_id, admin_id, scope.conversation_id,
        )
        if len(rows) != 1:
            raise TakeoverClaimError('not_owner')
        # Recheck current role inside the fenced transaction, not JWT history.
        admin = await tx.query_raw("SELECT 1 FROM users WHERE id=$1 AND role='admin'", admin_id)
        if not admin:
            raise TakeoverClaimError('not_owner')
        previous = await tx.query_raw(
            "SELECT id,metadata FROM messages WHERE conversation_id=$1 AND role='assistant' "
            "AND metadata->>'takeover_request_key'=$2 LIMIT 1", scope.conversation_id, request.request_key,
        )
        if previous:
            if previous[0]['metadata'].get('takeover_input_fingerprint') != request.fingerprint:
                raise ChatRequestConflict()
            return {'message_id': previous[0]['id'], 'created': False, 'scope': scope}
        metadata = {'human_takeover': True, 'admin_id': admin_id, 'client_id': request.client_id,
                    'takeover_request_key': request.request_key, 'takeover_input_fingerprint': request.fingerprint}
        message = await tx.message.create(data={'conversation': {'connect': {'id': scope.conversation_id}},
            'role': 'assistant', 'content': text, 'metadata': Json(metadata)})
    return {'message_id': message.id, 'created': True, 'scope': scope, 'metadata': metadata}
