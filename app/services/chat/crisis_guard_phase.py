"""Crisis guard phase.

This phase is intentionally ahead of boundary, preflight, and general intent
detection.  It owns the combined state machine for:

- direct crisis signals
- semantic/ambiguous crisis signals
- active crisis-care follow-up state
- release turns
- boundary/patience interaction while crisis-care is active
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from app.observability.events import EVT_CHAT_CRISIS_DETECTED
from app.services.chat.crisis_state import (
    clear_crisis_care_state,
    load_crisis_care_state,
    mark_crisis_care_active,
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

logger = logging.getLogger(__name__)

CrisisGuardStatus = Literal[
    "none",
    "direct_crisis",
    "semantic_crisis",
    "crisis_followup",
    "release_pending",
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


# 危机安全网触发关键字 — 自伤 / 自杀 / 想结束生命的明确表达.
# 关键词层是硬兜底, 不是完整检测系统。含蓄表达由语义分类器负责。
_CRISIS_KEYWORDS = (
    "跳楼", "跳河", "跳桥", "跳轨", "跳海",
    "自杀", "自残", "自伤", "轻生",
    "想死", "我去死", "去死算了", "不想活", "活不下去",
    "活着没意思", "活着没意义", "活够了",
    "结束生命", "结束自己", "了结自己", "了结我自己",
    "上吊", "割腕", "吃药自尽",
    "不想存在", "消失算了", "消失就好",
    "跟这个世界说再见", "和这个世界说再见", "向这个世界说再见",
    "跟世界说再见", "和世界说再见", "向世界说再见",
    "告别这个世界", "告别世界", "离开这个世界", "离开世界",
    "对这个世界的最后一次", "在这个世界的最后一次",
)

_CRISIS_RELEASE_KEYWORDS = (
    "我安全", "安全了", "现在安全",
    "不想死了", "不会自杀", "不会自残",
    "刚才是气话", "只是气话", "刚才是开玩笑",
)

_CRISIS_SEMANTIC_HINTS = (
    "再见", "永别", "告别", "最后一次", "世界", "离开", "消失",
    "撑不住", "受不了", "没意义", "没意思", "解脱", "放弃",
    "不想继续", "别管我", "算了", "下辈子",
)

_CRISIS_CARE_ASSISTANT_MARKERS = (
    "你现在安全吗",
    "有没有伤害自己",
    "伤害自己的冲动",
    "我还在看着你刚才",
    "没翻过去",
    "我不会跳过",
)

_CRISIS_SOFT_CHECK_TURN_INTERVAL = 2
_CRISIS_CHECK_ANNOYED_TERMS = (
    "无聊的问题", "问这么多", "别问", "不要问", "烦不烦",
    "烦死", "审问", "查户口",
)


def _strip_release_phrases(text: str) -> str:
    candidate = text or ""
    for release in _CRISIS_RELEASE_KEYWORDS:
        candidate = candidate.replace(release, "")
    return candidate


def is_crisis_released(text: str) -> bool:
    """Fast conservative release signal."""
    if not text:
        return False
    return any(keyword in text for keyword in _CRISIS_RELEASE_KEYWORDS)


def is_crisis_message(text: str) -> bool:
    """Hard crisis trigger. High-confidence only; ambiguous cases use LLM."""
    if not text:
        return False
    candidate = _strip_release_phrases(text)
    return any(keyword in candidate for keyword in _CRISIS_KEYWORDS)


def should_semantic_crisis_check(text: str) -> bool:
    """Recall gate for semantic crisis classification."""
    if not text:
        return False
    candidate = _strip_release_phrases(text)
    hit_count = sum(1 for hint in _CRISIS_SEMANTIC_HINTS if hint in candidate)
    if hit_count >= 2:
        return True
    return any(
        phrase in candidate
        for phrase in (
            "说再见", "最后一次发泄", "最后一次告别", "让我走",
            "不想继续了", "不想撑了", "撑不下去了",
        )
    )


def is_crisis_care_assistant_message(text: str) -> bool:
    if not text:
        return False
    return any(marker in text for marker in _CRISIS_CARE_ASSISTANT_MARKERS)


def crisis_followup_safety_check_mode(
    *,
    followup_status: str,
    prior_release_count: int,
    turns_since_safety_check: int,
    user_message: str,
) -> str:
    """Return none/soft/annoyed for deterministic aftercare safety check cadence."""
    if followup_status != "guard":
        return "none"
    due = (
        prior_release_count > 0
        or turns_since_safety_check >= _CRISIS_SOFT_CHECK_TURN_INTERVAL
    )
    if not due:
        return "none"
    if any(term in user_message for term in _CRISIS_CHECK_ANNOYED_TERMS):
        return "annoyed"
    return "soft"


def _format_crisis_context(items: list[tuple[str, str]], max_items: int = 10) -> str:
    recent = items[-max_items:]
    lines = []
    for role, content in recent:
        speaker = "用户" if role == "user" else "AI"
        lines.append(f"{speaker}: {content[:220]}")
    return "\n".join(lines)


def recent_unresolved_crisis_message(
    messages: list[dict],
    *,
    exclude_id: str | None = None,
    window: int = 8,
) -> str | None:
    """Return the latest recent crisis user message unless later user text released it."""
    checked = 0
    for msg in reversed(messages):
        if exclude_id and msg.get("id") == exclude_id:
            continue
        if msg.get("role") != "user":
            continue
        content = str(msg.get("content") or "")
        if not content:
            continue
        checked += 1
        if checked > window:
            return None
        if is_crisis_released(content):
            return None
        if is_crisis_message(content):
            return content
    return None


def recent_unresolved_crisis_context(
    messages: list[dict],
    *,
    exclude_id: str | None = None,
    window: int = 24,
) -> str | None:
    """Infer unresolved crisis-care context from recent conversation history."""
    checked = 0
    seen_care_signal = False
    collected_reversed: list[tuple[str, str]] = []
    for msg in reversed(messages):
        if exclude_id and msg.get("id") == exclude_id:
            continue
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        checked += 1
        if checked > window:
            break
        if role == "user" and is_crisis_message(content):
            seen_care_signal = True
        if role == "assistant" and is_crisis_care_assistant_message(content):
            seen_care_signal = True
        collected_reversed.append((role, content))

    if not seen_care_signal:
        return None
    return _format_crisis_context(list(reversed(collected_reversed)))


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
    semantic_classify_fn: Callable[..., Awaitable[bool]] | None = None,
    followup_classify_fn: Callable[..., Awaitable[str]] | None = None,
) -> CrisisGuardDecision:
    """Run the crisis state machine before boundary/preflight/intent."""
    decision = CrisisGuardDecision()
    if sub_intent_mode:
        return decision

    semantic_classify_fn = semantic_classify_fn or crisis_message_classify
    followup_classify_fn = followup_classify_fn or crisis_followup_classify

    recent_context_text = format_recent_context(
        messages_dicts, exclude_message_id=user_message_id,
    )

    direct_keyword_hit = is_crisis_message(user_message)
    if direct_keyword_hit:
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

    crisis_state_context: str | None = None
    release_count = 0
    aftercare_turn_count = 0
    turns_since_safety_check = 0
    if not decision.crisis_force_intent:
        crisis_state = await load_crisis_care_state(
            conversation_id,
            user_id,
            workspace_id=workspace_id,
            agent_id=agent_id,
        )
        if crisis_state is not None:
            crisis_state_context = str(crisis_state.get("context") or "").strip()
            release_count = int(crisis_state.get("release_count") or 0)
            aftercare_turn_count = int(crisis_state.get("aftercare_turn_count") or 0)
            turns_since_safety_check = int(
                crisis_state.get("turns_since_safety_check") or 0
            )
        else:
            crisis_state_context = recent_unresolved_crisis_context(
                messages_dicts, exclude_id=user_message_id,
            )

    decision.recent_crisis_context = crisis_state_context
    decision.crisis_care_turn = decision.crisis_force_intent or crisis_state_context is not None

    if crisis_state_context is not None:
        prior_release_count = release_count
        next_aftercare_turn_count = aftercare_turn_count + 1
        try:
            followup_status = await followup_classify_fn(
                message=user_message,
                context=crisis_state_context,
            )
        except Exception as e:
            logger.warning(f"Crisis followup classifier failed, guarding: {e}")
            followup_status = "guard"

        if followup_status == "release":
            release_count += 1
            decision.crisis_followup_active = release_count < 2
            if decision.crisis_followup_active:
                decision.status = "release_pending"
                await mark_crisis_care_active(
                    conversation_id,
                    user_id,
                    workspace_id=workspace_id,
                    agent_id=agent_id,
                    context=f"{crisis_state_context}\n用户: {user_message}",
                    source="followup_release_pending",
                    release_count=release_count,
                    aftercare_turn_count=next_aftercare_turn_count,
                    turns_since_safety_check=0,
                )
            else:
                decision.status = "released"
                await clear_crisis_care_state(
                    conversation_id,
                    user_id,
                    workspace_id=workspace_id,
                    agent_id=agent_id,
                )
        else:
            next_turns_since_safety_check = turns_since_safety_check + 1
            decision.crisis_followup_check_mode = crisis_followup_safety_check_mode(
                followup_status=followup_status,
                prior_release_count=prior_release_count,
                turns_since_safety_check=next_turns_since_safety_check,
                user_message=user_message,
            )
            decision.crisis_followup_active = True
            decision.status = "crisis_followup"
            await mark_crisis_care_active(
                conversation_id,
                user_id,
                workspace_id=workspace_id,
                agent_id=agent_id,
                context=f"{crisis_state_context}\n用户: {user_message}",
                source="followup_guard",
                release_count=0,
                aftercare_turn_count=next_aftercare_turn_count,
                turns_since_safety_check=(
                    0 if decision.crisis_followup_check_mode != "none"
                    else next_turns_since_safety_check
                ),
            )

    if decision.crisis_force_intent:
        await mark_crisis_care_active(
            conversation_id,
            user_id,
            workspace_id=workspace_id,
            agent_id=agent_id,
            context=f"用户: {user_message}",
            source=decision.status,
            release_count=0,
            aftercare_turn_count=0,
            turns_since_safety_check=0,
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
            "[CRISIS-FOLLOWUP] recent unresolved crisis detected, guarding reply",
            extra={
                "event": EVT_CHAT_CRISIS_DETECTED,
                "user_message_len": len(user_message),
                "crisis_followup": True,
                "crisis_followup_check_mode": decision.crisis_followup_check_mode,
            },
        )

    return decision


__all__ = [
    "CrisisGuardDecision",
    "crisis_followup_safety_check_mode",
    "is_crisis_message",
    "is_crisis_released",
    "recent_unresolved_crisis_context",
    "recent_unresolved_crisis_message",
    "run_crisis_guard",
    "should_semantic_crisis_check",
]
