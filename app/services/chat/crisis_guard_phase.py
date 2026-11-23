"""Crisis guard phase.

This phase is intentionally ahead of boundary, preflight, and general intent
detection.  It owns the combined state machine for:

- direct crisis signals
- semantic/ambiguous crisis signals
- time-bounded crisis-care follow-up (aftercare)
- release turns
- boundary/patience interaction while crisis-care is active

Aftercare policy (2026-11-23, after conversation f2cb492c kept asking
"你现在还好吗" for 24 days): an episode lasts ``CRISIS_CARE_WINDOW`` from the
latest crisis signal and follow-up turns never extend it.  One explicit release
ends it.  The release classifier sees the real dialogue since the signal,
including the AI's own safety questions, so "对呀" answering "你现在安全吗" is
read as an answer.  Safety checks are spaced out and capped per episode, and
stop once the user says the questions are unwelcome.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from app.observability.events import EVT_CHAT_CRISIS_DETECTED
from app.services.chat.crisis_state import (
    CRISIS_CARE_WINDOW,
    load_crisis_care_state,
    mark_crisis_care_active,
    mark_crisis_care_released,
    parse_state_time,
)
from app.services.chat.data_fetch_phase import format_recent_context
from app.services.chat.intent_replies import (
    crisis_followup_classify,
    crisis_message_classify,
)
from app.services.interaction.boundary import (
    check_banned_keywords,
    restore_patience_for_crisis_care,
)
from app.services.rules.chat_keywords import (
    CRISIS_CHECK_ANNOYED_TERMS,
    CRISIS_KEYWORDS,
    CRISIS_RELEASE_KEYWORDS,
    CRISIS_SEMANTIC_DIRECT_PHRASES,
    CRISIS_SEMANTIC_HINTS,
)

logger = logging.getLogger(__name__)

CrisisGuardStatus = Literal[
    "none",
    "direct_crisis",
    "semantic_crisis",
    "crisis_followup",
    "released",
]


@dataclass
class CrisisGuardDecision:
    status: CrisisGuardStatus = "none"
    crisis_force_intent: bool = False
    crisis_followup_active: bool = False
    crisis_care_turn: bool = False
    skip_boundary: bool = False
    should_restore_patience: bool = False
    cached_patience: int | None = None
    recent_crisis_context: str | None = None
    crisis_followup_check_mode: str = "none"
    semantic_checked: bool = False
    semantic_detected: bool = False
    boundary_attack_present: bool = False
    reason: str | None = None

    @property
    def use_crisis_intent(self) -> bool:
        return self.crisis_force_intent or self.crisis_followup_active

    @property
    def intent_metadata(self) -> dict[str, Any]:
        if not self.crisis_followup_active:
            return {}
        return {
            "followup": True,
            "safety_check_mode": self.crisis_followup_check_mode,
        }


# The crisis reply itself already asks; later checks come at most every few
# user turns and at most twice per episode, like scheduled clinical follow-up
# rather than a per-turn questionnaire.
_SAFETY_CHECK_TURN_INTERVAL = 4
_MAX_SAFETY_CHECKS_PER_EPISODE = 2
_CONTEXT_MAX_LINES = 12
_CONTEXT_LINE_CHARS = 220


def _strip_release_phrases(text: str) -> str:
    candidate = text or ""
    for release in CRISIS_RELEASE_KEYWORDS:
        candidate = candidate.replace(release, "")
    return candidate


def is_crisis_released(text: str) -> bool:
    """Fast conservative release signal."""
    if not text:
        return False
    return any(keyword in text for keyword in CRISIS_RELEASE_KEYWORDS)


def is_crisis_message(text: str) -> bool:
    """Hard crisis trigger. High-confidence only; ambiguous cases use LLM."""
    if not text:
        return False
    candidate = _strip_release_phrases(text)
    return any(keyword in candidate for keyword in CRISIS_KEYWORDS)


def should_semantic_crisis_check(text: str) -> bool:
    """Recall gate for semantic crisis classification."""
    if not text:
        return False
    candidate = _strip_release_phrases(text)
    hit_count = sum(1 for hint in CRISIS_SEMANTIC_HINTS if hint in candidate)
    if hit_count >= 2:
        return True
    return any(phrase in candidate for phrase in CRISIS_SEMANTIC_DIRECT_PHRASES)


def is_safety_check_unwelcome(text: str) -> bool:
    return any(term in (text or "") for term in CRISIS_CHECK_ANNOYED_TERMS)


def crisis_followup_safety_check_mode(
    *,
    turns_since_safety_check: int,
    safety_check_count: int,
    checks_suppressed: bool,
    user_message: str,
) -> str:
    """Return none/soft/annoyed for the aftercare safety-check cadence.

    ``annoyed`` means: acknowledge the over-asking and stop; it is not another
    check.  After it, the episode suppresses further scheduled checks.
    """
    if checks_suppressed:
        return "none"
    if is_safety_check_unwelcome(user_message):
        return "annoyed"
    if safety_check_count >= _MAX_SAFETY_CHECKS_PER_EPISODE:
        return "none"
    if turns_since_safety_check >= _SAFETY_CHECK_TURN_INTERVAL:
        return "soft"
    return "none"


def _message_time(message: dict) -> datetime | None:
    return parse_state_time(message.get("createdAt"))


def _excluded(message: dict, excluded: set[str]) -> bool:
    return bool(message.get("id")) and message.get("id") in excluded


def latest_crisis_signal_at(
    messages: list[dict],
    *,
    now: datetime,
    exclude_ids: set[str] | None = None,
) -> datetime | None:
    """Fallback when Redis state is missing: latest unreleased crisis signal
    inside the aftercare window.  Rows without timestamps count as recent so a
    missing clock never drops an active episode."""
    excluded = exclude_ids or set()
    for msg in reversed(messages):
        if _excluded(msg, excluded) or msg.get("role") != "user":
            continue
        content = str(msg.get("content") or "")
        said_at = _message_time(msg)
        if said_at is not None and now - said_at >= CRISIS_CARE_WINDOW:
            return None
        if is_crisis_released(content) and not is_crisis_message(content):
            return None
        if is_crisis_message(content):
            return said_at or now
    return None


def crisis_episode_context(
    messages: list[dict],
    *,
    started_at: datetime,
    trigger: str = "",
    exclude_ids: set[str] | None = None,
) -> str:
    """Dialogue since the crisis signal, both sides, anchored on the trigger.

    The AI's lines matter: without them a release classifier cannot tell that
    "对呀" answered "你现在安全吗"."""
    excluded = exclude_ids or set()
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role")
        if role not in {"user", "assistant"} or _excluded(msg, excluded):
            continue
        said_at = _message_time(msg)
        if said_at is not None and said_at < started_at:
            continue
        content = " ".join(str(msg.get("content") or "").split())
        if not content:
            continue
        speaker = "用户" if role == "user" else "AI"
        lines.append(f"{speaker}: {content[:_CONTEXT_LINE_CHARS]}")
    head = f"用户: {trigger[:_CONTEXT_LINE_CHARS]}" if trigger else ""
    if head and lines and lines[0] == head:
        lines = lines[1:]
    tail = lines[-_CONTEXT_MAX_LINES:]
    if head:
        marker = ["(中间省略)"] if len(lines) > len(tail) else []
        return "\n".join([head, *marker, *tail])
    return "\n".join(tail)


def _latest_user_crisis_text(messages: list[dict], exclude_ids: set[str]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user" and not _excluded(msg, exclude_ids):
            content = str(msg.get("content") or "")
            if is_crisis_message(content):
                return content
    return ""


async def run_crisis_guard(
    *,
    conversation_id: str,
    user_id: str,
    workspace_id: str | None,
    agent_id: str | None,
    user_message: str,
    sub_intent_mode: bool,
    messages_dicts: list[dict],
    user_message_id: str | None,
    exclude_message_ids: set[str] | None = None,
    now: datetime | None = None,
    semantic_classify_fn: Callable[..., Awaitable[bool]] | None = None,
    followup_classify_fn: Callable[..., Awaitable[str]] | None = None,
) -> CrisisGuardDecision:
    """Run the crisis state machine before boundary/preflight/intent."""
    decision = CrisisGuardDecision()
    if sub_intent_mode:
        return decision

    semantic_classify_fn = semantic_classify_fn or crisis_message_classify
    followup_classify_fn = followup_classify_fn or crisis_followup_classify
    now = now or datetime.now(UTC)
    excluded = set(exclude_message_ids or ())
    if user_message_id:
        excluded.add(user_message_id)
    scope = {"workspace_id": workspace_id, "agent_id": agent_id}

    recent_context_text = format_recent_context(
        messages_dicts, exclude_message_id=user_message_id,
        exclude_message_ids=exclude_message_ids, purpose="crisis",
    )

    if is_crisis_message(user_message):
        decision.status = "direct_crisis"
        decision.crisis_force_intent = True
        decision.reason = "keyword"
    elif should_semantic_crisis_check(user_message):
        decision.semantic_checked = True
        try:
            semantic_detected = await semantic_classify_fn(
                message=user_message,
                context=recent_context_text,
            )
        except Exception as e:
            logger.warning(f"Crisis semantic classifier failed: {e}")
            semantic_detected = False
        if semantic_detected:
            decision.status = "semantic_crisis"
            decision.crisis_force_intent = True
            decision.semantic_detected = True
            decision.reason = "semantic"

    if decision.crisis_force_intent:
        # A new signal always starts a fresh episode and window.
        await mark_crisis_care_active(
            conversation_id, user_id, **scope,
            context=f"用户: {user_message}",
            trigger=user_message,
            source=decision.status,
            started_at=now,
            now=now,
        )
    else:
        await _run_aftercare(
            decision,
            conversation_id=conversation_id,
            user_id=user_id,
            scope=scope,
            user_message=user_message,
            messages_dicts=messages_dicts,
            excluded=excluded,
            now=now,
            followup_classify_fn=followup_classify_fn,
        )

    decision.crisis_care_turn = (
        decision.crisis_force_intent or decision.recent_crisis_context is not None
    )
    decision.skip_boundary = decision.crisis_care_turn
    decision.boundary_attack_present = bool(check_banned_keywords(user_message))
    decision.should_restore_patience = (
        decision.crisis_care_turn
        and bool(agent_id)
        and not decision.boundary_attack_present
    )
    if decision.should_restore_patience and agent_id:
        try:
            decision.cached_patience = await restore_patience_for_crisis_care(
                agent_id, user_id,
            )
        except Exception as e:
            logger.warning(f"Crisis-care patience restore failed: {e}")

    if decision.crisis_force_intent:
        logger.warning(
            f"[CRISIS] {decision.status} detected (len={len(user_message)})",
            extra={
                "event": EVT_CHAT_CRISIS_DETECTED,
                "user_message_len": len(user_message),
                "crisis_status": decision.status,
                "semantic_checked": decision.semantic_checked,
            },
        )
    elif decision.crisis_followup_active:
        logger.warning(
            "[CRISIS-FOLLOWUP] crisis aftercare window active, guarding reply",
            extra={
                "event": EVT_CHAT_CRISIS_DETECTED,
                "user_message_len": len(user_message),
                "crisis_followup": True,
                "crisis_followup_check_mode": decision.crisis_followup_check_mode,
            },
        )

    return decision


async def _run_aftercare(
    decision: CrisisGuardDecision,
    *,
    conversation_id: str,
    user_id: str,
    scope: dict[str, str | None],
    user_message: str,
    messages_dicts: list[dict],
    excluded: set[str],
    now: datetime,
    followup_classify_fn: Callable[..., Awaitable[str]],
) -> None:
    state = await load_crisis_care_state(conversation_id, user_id, **scope, now=now)
    if state is not None and state["status"] == "released":
        return
    if state is not None and state["started_at"] is not None:
        started_at = state["started_at"]
        trigger = state["trigger"]
    else:
        # Redis missing/legacy: recover from history, bounded by time.
        signal_at = latest_crisis_signal_at(messages_dicts, now=now, exclude_ids=excluded)
        if signal_at is None:
            return
        state = state or {}
        started_at = signal_at
        trigger = _latest_user_crisis_text(messages_dicts, excluded)

    context = crisis_episode_context(
        messages_dicts, started_at=started_at, trigger=trigger, exclude_ids=excluded,
    )
    decision.recent_crisis_context = context or f"用户: {trigger}"
    try:
        followup_status = await followup_classify_fn(
            message=user_message,
            context=decision.recent_crisis_context,
        )
    except Exception as e:
        logger.warning(f"Crisis followup classifier failed, guarding: {e}")
        followup_status = "guard"

    if followup_status == "release":
        decision.status = "released"
        decision.reason = "classifier_release"
        await mark_crisis_care_released(
            conversation_id, user_id, **scope, started_at=started_at, now=now,
        )
        return

    turns_since_check = int(state.get("turns_since_safety_check") or 0) + 1
    check_count = int(state.get("safety_check_count") or 0)
    suppressed = bool(state.get("checks_suppressed"))
    mode = crisis_followup_safety_check_mode(
        turns_since_safety_check=turns_since_check,
        safety_check_count=check_count,
        checks_suppressed=suppressed,
        user_message=user_message,
    )
    decision.crisis_followup_check_mode = mode
    decision.crisis_followup_active = True
    decision.status = "crisis_followup"
    await mark_crisis_care_active(
        conversation_id, user_id, **scope,
        context=decision.recent_crisis_context,
        trigger=trigger,
        source="followup_guard",
        started_at=started_at,
        aftercare_turn_count=int(state.get("aftercare_turn_count") or 0) + 1,
        turns_since_safety_check=0 if mode == "soft" else turns_since_check,
        safety_check_count=check_count + (1 if mode == "soft" else 0),
        checks_suppressed=suppressed or mode == "annoyed",
        now=now,
    )


__all__ = [
    "CrisisGuardDecision",
    "crisis_episode_context",
    "crisis_followup_safety_check_mode",
    "is_crisis_message",
    "is_crisis_released",
    "is_safety_check_unwelcome",
    "latest_crisis_signal_at",
    "run_crisis_guard",
    "should_semantic_crisis_check",
]
