"""Authenticated transport -> trusted preparation -> atomic SQL acceptance.

No Redis queue submission or legacy background hook is allowed on this path.
The shared activation gate keeps it staged until consumers/delivery qualify.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import logging
from typing import Any

from app.services.runtime.chat_ingress import (
    AcceptedChatMessage, ChatRequestConflict, accept_chat_message, lookup_chat_request,
)
from app.services.runtime.chat_ingress_contracts import (
    ChatAggregationPolicy, ChatExecutionSnapshot, ChatRequestInput, PreparedChatMessage,
)
from app.services.runtime.chat_ingress_effects import (
    ChatIngressEffects, ChatIngressQuotaBlocked, ChatIngressResourceInvalid,
)
from app.services.runtime.execution_scope import (
    ExecutionScope, ExecutionScopeUnavailable, bind_conversation_scope,
)

logger = logging.getLogger(__name__)


class ChatIngressInputInvalid(ValueError):
    """Only transport validation failures are attributed to client input."""


@dataclass(frozen=True, slots=True)
class ClientChatInput:
    request: ChatRequestInput
    paid_confirmed: bool


def payment_confirmation(payload: dict) -> bool:
    value = payload.get("paid_confirmed", False)
    if type(value) is not bool:
        raise ValueError("Payment confirmation must be a boolean")
    return value


def parse_client_chat_input(payload: Any) -> ClientChatInput:
    """Preserve original text/card and ordered resource IDs as retry identity."""
    try:
        if type(payload) is not dict:
            raise ValueError("Invalid message envelope")
        paid = payment_confirmation(payload)
        raw_ids = payload.get("attachments", [])
        if type(raw_ids) is not list or len(raw_ids) > 3:
            raise ValueError("Invalid attachment list")
        ids = [item.get("id") if type(item) is dict else item for item in raw_ids]
        request = ChatRequestInput.from_client(
            client_id=payload.get("client_id"), text=payload.get("message", ""),
            attachment_ids=ids, component_card=payload.get("component_card"),
        )
    except ValueError as error:
        raise ChatIngressInputInvalid("Invalid chat input") from error
    return ClientChatInput(request, paid)


@dataclass(frozen=True, slots=True)
class PreparedChatAcceptance:
    message: PreparedChatMessage
    snapshot: ChatExecutionSnapshot
    policy: ChatAggregationPolicy


def _bound_execution_snapshot(conversation_id: str) -> ChatExecutionSnapshot:
    from app.config import settings
    from app.services.chat.graph_executor import select_executor
    from app.services.chat.main_graph import GRAPH_VERSION
    from app.services.prompting.registry import PROMPT_DEFINITIONS
    from app.services.prompting.store import get_prompt_snapshot
    from app.services.runtime_config import export_bound_chat_configuration

    bound = get_prompt_snapshot()
    if bound is None:
        raise RuntimeError("A bound prompt snapshot is required")
    templates = {key: {"content": content, "enabled": enabled,
                       "sha256": sha256(content.encode()).hexdigest()}
                 for key, (content, enabled) in bound.items()}
    # Match runtime fallback for registry keys that have never been bootstrapped.
    for definition in PROMPT_DEFINITIONS:
        templates.setdefault(definition.key, {
            "content": definition.default_text, "enabled": True,
            "sha256": sha256(definition.default_text.encode()).hexdigest(),
        })
    executor = select_executor(conversation_id)
    return ChatExecutionSnapshot.capture(
        executor=executor, graph_version=GRAPH_VERSION if executor == "langgraph" else "legacy-chat-v1",
        state_version=1, config=export_bound_chat_configuration(),
        prompts={"schema_version": 1, "templates": templates},
        budget={"schema_version": 1,
                "stream_timeout_s": settings.llm_chat_stream_timeout_s,
                "first_chunk_timeout_s": settings.llm_chat_stream_first_chunk_timeout_s,
                "utility_timeout_s": settings.llm_utility_timeout_s},
    )


async def prepare_sql_chat_message(
    scope: ExecutionScope, request: ChatRequestInput, *, client_supports_voice: bool = False,
) -> PreparedChatAcceptance:
    from app.db import db
    from app.services import offerings
    from app.services.chat_links import extract_first_url
    from app.services.chat_media import repo
    from app.services.chat_media.prompt import render_user_message_with_attachments
    from app.services.chat_media.vision import ensure_vision_summaries
    from app.services.interaction.reply_context import build_reply_timing_context
    from app.services.interaction.user_turn_aggregation import plan_sql_user_message_aggregation
    from app.services.llm.usage_tracker import usage_session
    from app.services.mbti import get_mbti
    from app.services.relationship.emotion import quick_emotion_estimate
    from app.services.runtime_config import bind_agent_context, reset_current_agent
    from app.services.schedule_domain.schedule import (
        generate_daily_schedule, get_cached_schedule, get_current_status, get_life_overview,
    )
    from app.services.schedule_domain.time_service import _now_corrected

    original = json.loads(request.input_json)
    text = original["text"].strip()
    card = original["component_card"]
    card_type = card.get("type") if card else None
    # R01.03.8.3 must provide their own binding/follow-up adapters. Reject before
    # link provider fetches, mutations or ordinary-message quota exemptions.
    if (card is not None and card_type not in {"gift", "red_packet"}) or extract_first_url(text):
        raise ChatIngressResourceInvalid()
    agent = await db.aiagent.find_unique(where={"id": scope.agent_id})
    if not agent or agent.userId != scope.owner_user_id or agent.status != "active":
        raise ExecutionScopeUnavailable()
    received_at = _now_corrected()
    binding = await bind_agent_context(scope.agent_id, require_loaded=True)
    try:
        snapshot = _bound_execution_snapshot(scope.conversation_id)
        attachments = await repo.get_message_attachments(
            attachment_ids=original["attachment_ids"], user_id=scope.owner_user_id,
            conversation_id=scope.conversation_id,
        )
        by_id = {item.id: item for item in attachments}
        if set(by_id) != set(original["attachment_ids"]):
            raise ChatIngressResourceInvalid()
        attachments = [by_id[ident] for ident in original["attachment_ids"]]
        offering = None
        if card_type:
            authorize = offerings.authorize_gift_card if card_type == "gift" else offerings.authorize_red_packet_card
            try:
                card = await authorize(card, user_id=scope.owner_user_id, agent_id=scope.agent_id,
                                       conversation_id=scope.conversation_id)
            except ValueError as error:
                raise ChatIngressResourceInvalid() from error
            card = dict(card)
            offering = card.pop("_offering")
        metadata = {}
        if attachments:
            async with usage_session(scope="chat_media", conversation_id=scope.conversation_id,
                                     agent_id=scope.agent_id, user_id=scope.owner_user_id):
                metadata["attachments"] = await ensure_vision_summaries(attachments, user_text=text)
        prompt_text = render_user_message_with_attachments(text, metadata.get("attachments", []))
        schedule = await get_cached_schedule(scope.agent_id)
        if not schedule:
            schedule = await generate_daily_schedule(
                scope.agent_id, agent.name, get_mbti(agent), user_id=scope.owner_user_id,
                life_overview=await get_life_overview(scope.agent_id),
            )
        status = get_current_status(schedule) if schedule else {
            "activity": "自由时间", "type": "leisure", "status": "idle",
        }
        context = await build_reply_timing_context(
            agent_id=scope.agent_id, user_id=scope.owner_user_id, received_status=status,
            user_emotion=quick_emotion_estimate(prompt_text),
        )
        context.update(received_at=received_at.isoformat(), client_supports_voice=client_supports_voice)
        if offering:
            metadata.update(component_card=card, queued=True, **{card_type: True})
            context.update(delay_seconds=0.0, component_card_reply=True, skip_time_memory_lookup=True)
            context[card_type] = offerings.reply_context_payload(offering)
            prompt_text = await offerings.build_offering_user_message(offering)
        policy = await plan_sql_user_message_aggregation(
            conversation_id=scope.conversation_id, text=prompt_text,
            reply_context=context, offering=offering is not None,
        )
        if policy.mode == "fragment_window":
            metadata["fragment"] = True
        else:
            metadata["queued"] = True
        message = PreparedChatMessage.capture(persisted_text=text, prompt_text=prompt_text,
                                             metadata=metadata, reply_context=context, received_at=received_at)
        return PreparedChatAcceptance(message, snapshot, policy)
    finally:
        reset_current_agent(binding)


async def receive_client_chat(
    *, actor_user_id: str, conversation_id: str, payload: Any,
    client_supports_voice: bool = False, database=None, prepare=None,
) -> AcceptedChatMessage:
    """No accepted response escapes a failed commit; never submit a Redis job."""
    incoming = parse_client_chat_input(payload)
    scope = await bind_conversation_scope(actor_user_id=actor_user_id,
                                          conversation_id=conversation_id, database=database)
    existing = await lookup_chat_request(scope, incoming.request, database=database)
    if existing is not None:
        return existing
    try:
        prepared = await (prepare or prepare_sql_chat_message)(
            scope, incoming.request, client_supports_voice=client_supports_voice,
        )
    except ChatIngressResourceInvalid:
        # A concurrent identical request may have bound the resource after our
        # initial lookup. Recover that receipt; absence is never authorization.
        existing = await lookup_chat_request(scope, incoming.request, database=database)
        if existing is not None:
            return existing
        raise
    if type(prepared) is not PreparedChatAcceptance:
        raise TypeError("Trusted prepared acceptance is required")
    return await accept_chat_message(
        scope, incoming.request, prepared.message, prepared.snapshot, prepared.policy,
        database=database, effects=ChatIngressEffects(incoming.paid_confirmed),
    )


def accepted_chat_events(receipt: AcceptedChatMessage) -> tuple[dict, ...]:
    """ACK describes persistence, never successful execution or a replayed reply."""
    now = datetime.now(timezone.utc)
    delay = max(0.0, (receipt.available_at - now).total_seconds())
    collecting = receipt.phase == "collecting" and receipt.run_status == "queued"
    ack = {"type": "ack", "data": {
        "message_id": receipt.message_id, "client_id": receipt.client_id,
        "run_id": receipt.run_id, "received_at": now.isoformat(),
        "ui_delay_seconds": 0.0 if collecting else delay, "defer_ui": collecting,
        "run_status": receipt.run_status, "job_status": receipt.job_status,
        "duplicate": not receipt.created,
    }}
    if receipt.run_status in {"queued", "running", "waiting"}:
        return ack, {"type": "pending", "data": {
            "status": "aggregating" if receipt.phase == "collecting" else "queued",
            "delay": delay, "run_id": receipt.run_id,
        }}
    return (ack,)


async def handle_sql_chat_frame(websocket, actor_user_id: str, conversation_id: str,
                                payload: Any, *, client_supports_voice: bool = False) -> None:
    client_id = payload.get("client_id") if type(payload) is dict else None
    # Keep only a bounded valid string in diagnostics; never echo a raw envelope.
    client_id = client_id if type(client_id) is str and len(client_id) <= 249 else None
    try:
        receipt = await receive_client_chat(actor_user_id=actor_user_id,
            conversation_id=conversation_id, payload=payload, client_supports_voice=client_supports_voice)
    except ChatIngressQuotaBlocked as error:
        await websocket.send_json({"type": "quota_blocked", "data": {
            "reason": error.result["reason"], "per_msg_cost": error.result["per_msg_cost"],
            "spendable_tickets": error.result["spendable_tickets"], "client_id": client_id,
        }})
        return
    except ChatRequestConflict:
        code, message = "request_conflict", "同一消息ID的内容已改变，请检查后重新发送"
    except ChatIngressResourceInvalid:
        code, message = "invalid_resource", "附件或卡片无效或已发送"
    except ExecutionScopeUnavailable:
        await websocket.close(code=4403, reason="conversation_access_denied")
        return
    except ChatIngressInputInvalid:
        code, message = "invalid_request", "消息格式无效，请重新发送"
    except Exception:
        # A lost commit ACK is indistinguishable from failure. Retrying the same
        # identity discovers SQL state; do not charge or execute through legacy.
        logger.warning("SQL chat acceptance unavailable")
        code, message = "storage_unavailable", "消息接收暂不可用，请使用原消息ID重试"
    else:
        for event in accepted_chat_events(receipt):
            await websocket.send_json(event)
        return
    await websocket.send_json({"type": "error", "data": {
        "code": code, "message": message, "client_id": client_id,
    }})
