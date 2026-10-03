"""Invocation-owned handles and the single parent turn lifecycle.

These objects are not graph state and must never be persisted as checkpoints.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any


class ParentTraceView:
    """Handlers may close a segment; only the driver closes the trace root."""

    def __init__(self, tracer):
        self._tracer = tracer

    def __getattr__(self, name):
        return getattr(self._tracer, name)

    def close(self):
        pass


@dataclass
class ChatFrame:
    services: Any
    owner: Any
    conversation_id: str
    user_message: str
    agent: Any
    user_id: str
    reply_context: dict | None
    user_message_id: str | None
    delivered_from_queue: bool
    sub_intent_mode: bool
    forced_intent: Any
    reply_index_offset: int
    parent_patience: int | None
    achievement_turn_final: bool
    agent_id: str | None
    workspace_id: str | None
    tracer: Any
    owned_tasks: list[asyncio.Task] = field(default_factory=list)
    completion_reason: str | None = None
    completion_reason_set: bool = False
    _build_main_chat_messages: Any = None
    _cancel_fetch_task: asyncio.Task | None = None
    _cancel_l3_task: asyncio.Task | None = None
    aggregated_turn_text: str = None
    ai_status: Any = None
    allow_count_variation: Any = None
    boundary_ctx: Any = None
    cached_patience: int = 100
    classified_memories: list = None
    continuation_lines: list[str] = None
    continuation_task: asyncio.Task | None = None
    contradiction_inquiry: str | None = None
    covered_until_user_ts: Any = None
    crisis_care_turn: bool = False
    crisis_decision: Any = None
    crisis_followup_active: bool = None
    crisis_followup_check_mode: Any = None
    crisis_force_intent: bool = None
    crisis_memory_task: asyncio.Task | None = None
    crisis_portrait_task: asyncio.Task | None = None
    current_achievement_turn_id: str | None = None
    current_state_fast_path: bool = None
    current_turn_ids: set[str] = field(default_factory=set)
    delay_context: str | None = None
    detected_intent: Any = None
    emitted_replies: list[dict] = field(default_factory=list)
    fetch_task: asyncio.Task | None = None
    fetched: Any = None
    first_assistant_message_id: str | None = None
    full_response: str = ""
    get_latest_portrait: Any = None
    gift_context: dict | None = None
    intimacy_stage: str = None
    l3_memories: list[str] = None
    l3_task: asyncio.Task | None = None
    last_reply_count: Any = None
    location_share_context: bool = None
    max_reply_count: Any = None
    max_total: Any = None
    mbti: Any = None
    memory_relevance: str = None
    messages_dicts: list[dict] = field(default_factory=list)
    needs_web_search: bool = None
    offering_context: dict | None = None
    pending_sub_fragments: dict[str, str] = field(default_factory=dict)
    portrait: Any = None
    preflight_ctx: Any = None
    prepared_voice: Any = None
    previous_assistant: Any = None
    prompt_user_emotion: dict | None = None
    recent_context_text: str = None
    recent_crisis_context: Any = None
    recent_messages: list = None
    red_packet_context: dict | None = None
    reengagement_gap_seconds: float | None = None
    relational_context: Any = None
    replies: list[str] = None
    reply_count: int = None
    reply_emotion_pre: Any = None
    reply_is_fallback: bool = None
    response_diagnostics: dict = field(default_factory=dict)
    retrieve_crisis_followup_memories: Any = None
    retrieve_crisis_memories: Any = None
    sc_ctx: Any = None
    schedule: Any = None
    schedule_context: Any = None
    session_recap_task: asyncio.Task | None = None
    skip_time_memory_lookup: bool = None
    time_context: Any = None
    time_memories: list[str] = None
    topic_context: dict | None = None
    topic_intimacy: float = None
    voice_data: Any = None

    def __post_init__(self):
        context = self.reply_context or {}
        self.skip_time_memory_lookup = bool(context.get("skip_time_memory_lookup"))
        self.red_packet_context = (
            context.get("red_packet")
            if isinstance(context.get("red_packet"), dict)
            and context["red_packet"].get("offering_id")
            else None
        )
        self.gift_context = (
            context.get("gift")
            if isinstance(context.get("gift"), dict)
            and context["gift"].get("offering_id")
            else None
        )
        self.offering_context = self.red_packet_context or self.gift_context
        self.location_share_context = bool(context.get("location_share"))

    def completed(self, phase):
        from app.services.chat.main_graph import PhaseOutcome

        return PhaseOutcome(phase)

    def spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.owned_tasks.append(task)
        return task

    async def cleanup(self):
        for task in self.owned_tasks:
            if not task.done():
                task.cancel()
        if self.owned_tasks:
            await asyncio.gather(*self.owned_tasks, return_exceptions=True)
        if self.prepared_voice is not None and not self.first_assistant_message_id:
            from app.services.speech_output.delivery import (
                discard_prepared_voice_output,
            )

            await discard_prepared_voice_output(self.prepared_voice)
            self.prepared_voice = None

    def intent_snapshot(self):
        if self.detected_intent is None:
            return {}
        return {
            "type": self.detected_intent.intent.value,
            "origin": "forced"
            if self.forced_intent is not None
            else "crisis_guard"
            if self.crisis_care_turn
            else "current_state_phrase"
            if self.current_state_fast_path
            else "unified_classifier",
            "confidence": self.detected_intent.confidence,
            "pending_labels": list(self.pending_sub_fragments),
        }

    async def save_replies(self, conversation_id, replies, **kwargs):
        if conversation_id != self.conversation_id:
            raise ValueError("Reply persistence escaped its conversation")
        # Per-message achievements still run in the existing persistence layer.
        # The parent submits the turn achievement once with all fragment replies.
        from app.services.chat.main_graph import GRAPH_VERSION

        # Persist diagnostics using the existing metadata/API contract. Runtime
        # handles and entire graph state never enter message metadata.
        payloads = []
        for reply in replies:
            payload = dict(reply) if isinstance(reply, dict) else {"text": reply}
            diagnostics = dict(
                payload.get("response_diagnostics") or self.response_diagnostics or {}
            )
            diagnostics.update(
                executor="langgraph",
                graph_version=GRAPH_VERSION,
                fragment_index=self.owner.fragment_index,
                checkpoint_enabled=False,
            )
            payload["response_diagnostics"] = diagnostics
            payloads.append(payload)
        replies = payloads
        kwargs["achievement_turn_final"] = False
        message_id = await self.services._save_replies(
            conversation_id, replies, **kwargs
        )
        if not message_id:
            raise RuntimeError("Chat reply persistence failed")
        self.owner.persisted.append(
            {
                "message_id": message_id,
                "texts": [
                    str(r.get("text", "")) if isinstance(r, dict) else str(r)
                    for r in replies
                ],
                "metadata": dict(replies[0])
                if replies and isinstance(replies[0], dict)
                else {},
            }
        )
        return message_id

    async def short_reply(self, reply, conversation_id, agent_id, user_id, **kwargs):
        from app.services.chat.multi_intent import short_circuit_reply

        self.completion_reason = kwargs.get("proactive_reason", "short_circuit")
        self.completion_reason_set = True
        kwargs.update(defer_turn_finalization=True, achievement_turn_final=False)
        return await short_circuit_reply(
            reply, conversation_id, agent_id, user_id, self.save_replies, **kwargs
        )

    async def boundary_reply(self, *args, **kwargs):
        kwargs["proactive_reason"] = None
        return await self.short_reply(*args, **kwargs)

    async def system_reply(self, *args, **kwargs):
        kwargs["proactive_reason"] = self.services.ARM_REASON_SYSTEM
        return await self.short_reply(*args, **kwargs)


@dataclass
class ChatGraphContext:
    frame: ChatFrame
    original_message: str
    achievement_turn_final: bool
    primary: ChatFrame | None = None
    fragment_index: int = 0
    next_reply_index: int = 0
    pending: list[tuple[str, str]] = field(default_factory=list)
    persisted: list[dict] = field(default_factory=list)
    finished: bool = False
    background_submitted: bool = False
    allow_ai_memory: bool = False

    def note_event(self, event):
        if event["event"] == "reply":
            payload = json.loads(event["data"])
            if payload.get("index") != self.next_reply_index:
                raise ValueError("Chat fragment reply indices are discontinuous")
            self.next_reply_index += 1

    async def advance_fragment(self):
        current = self.frame
        await current.cleanup()
        if not (current.boundary_ctx and current.boundary_ctx.stopped):
            self.allow_ai_memory |= not (
                current.sc_ctx
                and current.sc_ctx.last_short_circuit_kind
                in {"schedule_query", "current_state"}
            )
        if self.primary is None:
            self.primary = current
            from app.services.chat.intent_dispatcher import (
                INTENT_PRIORITY,
                LABEL_TO_INTENT,
                IntentType,
            )

            fragments = (
                {}
                if current.sc_ctx and current.sc_ctx.consumed_full_message
                else current.pending_sub_fragments
            )
            self.pending = sorted(
                fragments.items(),
                key=lambda item: (
                    INTENT_PRIORITY.index(item[0])
                    if item[0] in INTENT_PRIORITY
                    else float("inf")
                ),
            )
            if len(self.pending) > len(LABEL_TO_INTENT):
                raise ValueError("Too many chat intent fragments")
        if not self.pending:
            return False
        label, message = self.pending.pop(0)
        from app.services.chat.intent_dispatcher import LABEL_TO_INTENT, IntentType

        parent = self.primary
        self.fragment_index += 1
        self.frame = ChatFrame(
            services=parent.services,
            owner=self,
            conversation_id=parent.conversation_id,
            user_message=message,
            agent=parent.agent,
            user_id=parent.user_id,
            reply_context=parent.reply_context,
            user_message_id=None,
            delivered_from_queue=True,
            sub_intent_mode=True,
            forced_intent=LABEL_TO_INTENT.get(label, IntentType.NONE),
            reply_index_offset=self.next_reply_index,
            parent_patience=parent.cached_patience,
            achievement_turn_final=False,
            agent_id=parent.agent_id,
            workspace_id=parent.workspace_id,
            tracer=parent.tracer,
        )
        return True

    async def finish_turn(self):
        if self.finished or self.primary is None or not self.persisted:
            raise RuntimeError("Chat turn cannot complete without persisted replies")
        parent = self.primary
        services = parent.services
        boundary_handled = (
            parent.boundary_ctx is not None and parent.boundary_ctx.stopped
        )
        if boundary_handled:
            reason = None
        elif parent.sc_ctx is not None and parent.sc_ctx.last_short_circuit_kind:
            from app.services.proactive.state import short_circuit_arm_reason

            reason = short_circuit_arm_reason(parent.sc_ctx.last_short_circuit_kind)
        elif parent.completion_reason_set:
            reason = parent.completion_reason
        elif (
            parent.preflight_ctx is not None
            and parent.preflight_ctx.last_short_circuit_reply is not None
        ):
            reason = services.ARM_REASON_SYSTEM
        else:
            reason = (
                services.ARM_REASON_CRISIS
                if parent.crisis_care_turn
                else services.ARM_REASON_SYSTEM
                if parent.contradiction_inquiry
                else services.ARM_REASON_REPLY
            )
        await services.finish_assistant_turn(
            conversation_id=parent.conversation_id,
            agent_id=parent.agent_id,
            user_id=parent.user_id,
            workspace_id=parent.workspace_id,
            turn_message_ids=sorted(parent.current_turn_ids),
            proactive_reason=reason,
        )
        if self.achievement_turn_final:
            from app.services.achievements.service import handle_assistant_turn_event

            first = self.persisted[0]
            services._fire_background(
                handle_assistant_turn_event(
                    conversation_id=parent.conversation_id,
                    message_id=first["message_id"],
                    assistant_texts=[
                        text for row in self.persisted for text in row["texts"]
                    ],
                    user_message_ids=sorted(parent.current_turn_ids),
                    turn_id=parent.current_achievement_turn_id,
                    metadata=first["metadata"],
                )
            )
        if not boundary_handled:
            # Submit once after all fragments and persistence have succeeded.
            services._fire_background(
                services._background_post_process(
                    user_id=parent.user_id,
                    agent_id=parent.agent_id,
                    conversation_id=parent.conversation_id,
                    user_message=self.original_message,
                    user_message_id=parent.user_message_id,
                    full_response=" ".join(
                        text for row in self.persisted for text in row["texts"]
                    ),
                    messages_dicts=parent.messages_dicts,
                    user_emotion=parent.prompt_user_emotion,
                    skip_ai_memory=not self.allow_ai_memory,
                    skip_memory=bool(parent.offering_context)
                    or parent.location_share_context,
                    workspace_id=parent.workspace_id,
                )
            )
            self.background_submitted = True
        if parent.full_response and len(parent.recent_messages) <= 1:
            title = (
                "红包"
                if parent.red_packet_context
                else "礼物"
                if parent.gift_context
                else parent.user_message[:50]
                + ("..." if len(parent.user_message) > 50 else "")
            )
            services._fire_background(
                services.db.conversation.update(
                    where={"id": parent.conversation_id}, data={"title": title}
                )
            )
        self.finished = True
        done = {"message_id": "complete"}
        if parent.tracer.trace_id and parent.tracer.is_active:
            done["assistant_message_id"] = self.persisted[0]["message_id"]
        return {"event": "done", "data": json.dumps(done)}
