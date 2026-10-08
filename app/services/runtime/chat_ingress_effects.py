"""SQL-only business effects for staged chat acceptance.

Called after receipt deduplication, inside the same scoped transaction as the
message and Run/Job. Never call the legacy offering binder: it starts background
memory work and rewrites messages outside this transaction. Endpoint/worker
activation and those durable follow-up jobs are separate roadmap gates.
"""

from dataclasses import dataclass
import json
from typing import Any

from app.services import wallet
from app.services.offerings import _offering_from_row, build_offering_card
from app.services.runtime.chat_ingress_contracts import ChatRequestInput
from app.services.runtime.execution_scope import ExecutionScope
from app.services.vip.chat_quota import consume_one_in_transaction


class ChatIngressResourceInvalid(ValueError):
    def __init__(self) -> None:
        super().__init__("Chat resource is invalid or already bound")


class ChatIngressQuotaBlocked(ValueError):
    def __init__(self, result: dict) -> None:
        super().__init__("Chat quota requires confirmation or funds")
        self.result = dict(result)


@dataclass(frozen=True, slots=True)
class ChatIngressEffects:
    """Server adapter options; no client-provided ownership or VIP authority."""
    paid_confirmed: bool = False

    def __post_init__(self) -> None:
        if type(self.paid_confirmed) is not bool:
            raise ValueError("Payment confirmation must be a boolean")

    async def commit(self, tx: Any, scope: ExecutionScope, request: ChatRequestInput,
                     message_id: str, metadata: dict) -> dict:
        original = json.loads(request.input_json)
        ids = original["attachment_ids"]
        card = original["component_card"]
        card_type = card.get("type") if card else None
        # Other cards have additional state mutations, providers or follow-ups.
        # Fail closed until their adapters are implemented; never pretend they
        # are ordinary chat or silently exempt their quota.
        if (card is not None and card_type not in {"gift", "red_packet"}) or metadata.get("link_card"):
            raise ChatIngressResourceInvalid()
        prepared_card = metadata.get("component_card")
        if ((prepared_card is not None and type(prepared_card) is not dict)
                or (prepared_card or {}).get("type") != card_type):
            raise ChatIngressResourceInvalid()

        attachments = metadata.get("attachments", [])
        if not isinstance(attachments, list) or any(type(a) is not dict for a in attachments):
            raise ChatIngressResourceInvalid()
        if [a.get("id") for a in attachments] != ids:
            raise ChatIngressResourceInvalid()
        if ids:
            rows = await tx.query_raw(
                "SELECT id FROM chat_message_attachments WHERE id=ANY($1::text[]) "
                "AND user_id=$2 AND conversation_id=$3 AND message_id IS NULL "
                "ORDER BY id FOR UPDATE", ids, scope.owner_user_id, scope.conversation_id,
            )
            if {r["id"] for r in rows} != set(ids):
                raise ChatIngressResourceInvalid()
            count = await tx.execute_raw(
                "UPDATE chat_message_attachments SET message_id=$1,updated_at=clock_timestamp() "
                "WHERE id=ANY($2::text[]) AND user_id=$3 AND conversation_id=$4 "
                "AND message_id IS NULL", message_id, ids, scope.owner_user_id, scope.conversation_id,
            )
            if count != len(ids):
                raise ChatIngressResourceInvalid()

        offering_id = None
        if card_type:
            payload = card.get("payload")
            offering_id = payload.get("offering_id") if isinstance(payload, dict) else None
            if type(offering_id) is not str or not offering_id:
                raise ChatIngressResourceInvalid()
            rows = await tx.query_raw(
                "SELECT * FROM user_offerings WHERE id=$1 AND user_id=$2 "
                "AND agent_id=$3 AND conversation_id=$4 AND kind=$5 "
                "AND status='sent' AND message_id IS NULL FOR UPDATE", offering_id,
                scope.owner_user_id, scope.agent_id, scope.conversation_id, card_type,
            )
            if len(rows) != 1:
                raise ChatIngressResourceInvalid()
            authoritative_card = build_offering_card(_offering_from_row(rows[0]))
            if metadata.get("component_card") != authoritative_card:
                raise ChatIngressResourceInvalid()
            count = await tx.execute_raw(
                "UPDATE user_offerings SET message_id=$2 WHERE id=$1 AND message_id IS NULL "
                "AND status='sent'", offering_id, message_id,
            )
            if count != 1:
                raise ChatIngressResourceInvalid()
            quota = {"allowed": True, "mode": "exempt", "charged": 0}
        else:
            quota = await consume_one_in_transaction(
                scope.owner_user_id, is_vip=await wallet.is_vip(scope.owner_user_id, client=tx),
                paid_confirmed=self.paid_confirmed, client=tx, source_id=message_id,
            )
            if not quota["allowed"]:
                raise ChatIngressQuotaBlocked(quota)
        return {"version": 1, "quota": quota, "attachment_ids": ids, "offering_id": offering_id}
