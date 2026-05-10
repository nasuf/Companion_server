"""Unified user-turn aggregation orchestration.

This module is the public boundary for chat entrypoints.  It hides the two
internal aggregation strategies:

- fragment window: incomplete 1-2 character fragments, longer wait, direct join
- turn window: rapid complete user messages, short quiet window, newline join
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from app.services.chat.crisis_guard_phase import is_crisis_message
from app.services.chat.intent_dispatcher import (
    IntentType,
    detect_intent,
    is_explicit_current_state_query,
    is_explicit_l3_recall_query,
)
from app.services.interaction.aggregation import (
    flush_pending,
    has_turn_pending,
    is_short_message,
    push_pending,
    push_turn_pending,
    scan_expired,
    scan_turn_expired,
)
from app.services.interaction.boundary import check_banned_keywords
from app.services.interaction.reply_context import merge_reply_contexts
from app.services.memory.interaction.contradiction import load_pending_contradiction
from app.services.memory.interaction.deletion import load_pending_action
from app.services.rules.chat_keywords import (
    HIGH_CONFIDENCE_CANCEL_KEYWORDS,
    RECORD_MEMORY_CUES,
    REMINDER_ACTION_CUES,
    REMINDER_CONTENT_CUES,
    UNDO_CANCEL_KEYWORDS,
)

AggregationRoute = Literal["fragment_window", "turn_window", "immediate"]

_TURN_BYPASS_RECORD_CUES = (
    *HIGH_CONFIDENCE_CANCEL_KEYWORDS,
    *RECORD_MEMORY_CUES,
    *REMINDER_ACTION_CUES,
    *REMINDER_CONTENT_CUES,
    *UNDO_CANCEL_KEYWORDS,
)

_TURN_AGGREGATED_INTENTS = {
    IntentType.CURRENT_STATE,
    IntentType.SCHEDULE_QUERY,
}

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class UserMessageAggregationPlan:
    """Decision returned to API entrypoints before user message persistence."""

    route: AggregationRoute
    agent_id: str
    user_id: str
    conversation_id: str
    text: str
    metadata: dict[str, Any]
    final_message: str
    final_context: dict | None
    fallback_message: str
    fallback_context: dict | None

    @property
    def should_wait(self) -> bool:
        return self.route in {"fragment_window", "turn_window"}


async def should_bypass_user_turn_aggregation(
    conversation_id: str,
    text: str,
) -> bool:
    """Return True for messages that should not wait for a normal turn window."""
    if is_crisis_message(text):
        return True
    if check_banned_keywords(text):
        return True
    if any(cue in text for cue in _TURN_BYPASS_RECORD_CUES):
        return True
    if is_explicit_current_state_query(text):
        return False
    if is_explicit_l3_recall_query(text):
        return True

    try:
        if await load_pending_contradiction(conversation_id):
            return True
        if await load_pending_action(conversation_id):
            return True
    except Exception as e:
        # Pending state lookup is a convenience gate; Redis trouble should not
        # block normal chat delivery.
        logger.warning(f"[TURN-BYPASS] pending-state lookup failed: {e}")

    detected = detect_intent(text)
    if detected.intent in _TURN_AGGREGATED_INTENTS:
        return False
    return detected.intent is not IntentType.NONE


async def plan_user_message_aggregation(
    *,
    agent_id: str,
    user_id: str,
    conversation_id: str,
    text: str,
    reply_context: dict | None,
) -> UserMessageAggregationPlan:
    """Build the aggregation plan for one persisted-or-about-to-persist user message."""
    if is_short_message(text):
        if await has_turn_pending(agent_id=agent_id, user_id=user_id):
            return UserMessageAggregationPlan(
                route="turn_window",
                agent_id=agent_id,
                user_id=user_id,
                conversation_id=conversation_id,
                text=text,
                metadata={"queued": True},
                final_message=text,
                final_context=reply_context,
                fallback_message=text,
                fallback_context=reply_context,
            )
        return UserMessageAggregationPlan(
            route="fragment_window",
            agent_id=agent_id,
            user_id=user_id,
            conversation_id=conversation_id,
            text=text,
            metadata={"fragment": True},
            final_message=text,
            final_context=reply_context,
            fallback_message=text,
            fallback_context=reply_context,
        )

    pending_text, _, pending_context, _ = await flush_pending(
        agent_id=agent_id,
        user_id=user_id,
    )
    if pending_text:
        return UserMessageAggregationPlan(
            route="immediate",
            agent_id=agent_id,
            user_id=user_id,
            conversation_id=conversation_id,
            text=text,
            metadata={"queued": True},
            final_message="".join(part for part in [pending_text, text] if part),
            final_context=merge_reply_contexts(pending_context, reply_context),
            fallback_message=text,
            fallback_context=reply_context,
        )

    if await should_bypass_user_turn_aggregation(conversation_id, text):
        return UserMessageAggregationPlan(
            route="immediate",
            agent_id=agent_id,
            user_id=user_id,
            conversation_id=conversation_id,
            text=text,
            metadata={"queued": True},
            final_message=text,
            final_context=reply_context,
            fallback_message=text,
            fallback_context=reply_context,
        )

    return UserMessageAggregationPlan(
        route="turn_window",
        agent_id=agent_id,
        user_id=user_id,
        conversation_id=conversation_id,
        text=text,
        metadata={"queued": True},
        final_message=text,
        final_context=reply_context,
        fallback_message=text,
        fallback_context=reply_context,
    )


async def enqueue_planned_user_message(
    plan: UserMessageAggregationPlan,
    *,
    message_id: str | None,
) -> bool:
    """Enqueue a planned aggregation window. Returns False for immediate plans or failures."""
    if plan.route == "fragment_window":
        return await push_pending(
            agent_id=plan.agent_id,
            user_id=plan.user_id,
            conversation_id=plan.conversation_id,
            text=plan.text,
            reply_context=plan.final_context,
            message_id=message_id,
        )
    if plan.route == "turn_window":
        return await push_turn_pending(
            agent_id=plan.agent_id,
            user_id=plan.user_id,
            conversation_id=plan.conversation_id,
            text=plan.text,
            reply_context=plan.final_context,
            message_id=message_id,
        )
    return False


async def scan_due_user_turns() -> list[tuple[str, str, str, str, dict | None, str | None]]:
    """Return all due fragment and normal-turn aggregation windows."""
    fragment_turns = await scan_expired()
    normal_turns = await scan_turn_expired()
    return [*fragment_turns, *normal_turns]
