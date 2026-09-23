"""Conversation-first companionship for an active offline activity."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from app.config import settings
from app.db import db
from app.observability.events import EVT_OFFLINE_COMPANION_DECISION
from app.services.llm.models import get_chat_model, invoke_text
from app.services.offline import chat_emit
from app.services.offline import repository as repo
from app.services.offline.guidance import safe_guidance
from app.services.offline.module_settings import is_activity_enabled
from app.services.prompting.store import get_prompt_text

logger = logging.getLogger(__name__)

_ACTIONS = {"silent", "social", "ambient", "gentle_hint", "care"}
_TRIGGER_BY_ACTION = {
    "social": "offline_activity_companion_social",
    "ambient": "offline_activity_companion_ambient",
    "gentle_hint": "offline_activity_companion_hint",
    "care": "offline_activity_companion_care",
}
_REASON_CODES = {
    "paused_unanswered",
    "active_conversation",
    "user_replied_during_generation",
    "task_hint_ratio_guard",
    "no_remaining_focus",
    "model_silent",
    "model_social",
    "model_ambient",
    "model_gentle_hint",
    "model_care",
    "claim_fenced",
    "generation_failed",
}


class CompanionDecisionGenerationError(RuntimeError):
    pass


def _now() -> datetime:
    return datetime.now(UTC)


def _aware(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        return None


def _random_due(now: datetime | None = None) -> datetime:
    base = now or _now()
    low = max(1, int(settings.offline_activity_companion_min_interval_minutes))
    high = max(low, int(settings.offline_activity_companion_max_interval_minutes))
    return base + timedelta(minutes=random.randint(low, high))


async def can_take_over_proactive_channel(
    user_id: str,
    workspace_id: str | None,
) -> bool:
    if (
        not settings.offline_activity_companion_enabled
        or not await is_activity_enabled()
    ):
        return False
    activity = await repo.get_current_reached_activity(user_id, workspace_id)
    return bool(activity and activity.get("conditions_ready_at"))


async def note_user_interaction(
    user_id: str,
    workspace_id: str | None,
) -> None:
    """Resume a paused activity companion after any user message."""
    if not settings.offline_activity_companion_enabled:
        return
    activity = await repo.get_current_reached_activity(user_id, workspace_id)
    if activity:
        await repo.touch_activity_interaction(
            activity["id"],
            next_companion_at=_random_due(),
        )


async def scan_activity_companions() -> dict[str, int]:
    if not await is_activity_enabled():
        return {
            "recovered": 0,
            "pregen_repaired": 0,
            "claimed": 0,
            "sent": 0,
            "silent": 0,
            "failed": 0,
        }
    from app.services.offline.shooting_conditions import (
        recover_missing_prewritten_fragments,
        recover_unready_reached_activities,
    )

    recovery = await recover_unready_reached_activities()
    pregen_recovery = await recover_missing_prewritten_fragments()
    if not settings.offline_activity_companion_enabled:
        return {
            "recovered": recovery["recovered"],
            "pregen_repaired": pregen_recovery["repaired"],
            "claimed": 0,
            "sent": 0,
            "silent": 0,
            "failed": recovery["failed"] + pregen_recovery["failed"],
        }
    activities = await repo.claim_due_companion_activities()
    stats = {
        "recovered": recovery["recovered"],
        "pregen_repaired": pregen_recovery["repaired"],
        "claimed": len(activities),
        "sent": 0,
        "silent": 0,
        "failed": recovery["failed"] + pregen_recovery["failed"],
    }
    semaphore = asyncio.Semaphore(4)

    async def _run_one(activity: dict[str, Any]) -> tuple[str, bool]:
        async with semaphore:
            try:
                return "ok", await _process_activity(activity)
            except Exception as exc:  # noqa: BLE001 - isolate per-activity failures
                logger.warning(
                    "[offline-companion] activity=%s failed: %s",
                    activity.get("id"),
                    exc,
                )
                state = dict(activity.get("companion_state") or {})
                claim_token = str(activity.get("companion_claim_token") or "")
                if claim_token:
                    await repo.save_companion_decision(
                        activity["id"],
                        claim_token=claim_token,
                        state=state,
                        next_companion_at=_now() + timedelta(minutes=5),
                        sent=False,
                    )
                return "failed", False

    results = await asyncio.gather(*(_run_one(activity) for activity in activities))
    for status, sent in results:
        if status == "failed":
            stats["failed"] += 1
        else:
            stats["sent" if sent else "silent"] += 1
    return stats


async def _process_activity(activity: dict[str, Any]) -> bool:
    state = dict(activity.get("companion_state") or {})
    claim_token = str(activity.get("companion_claim_token") or "")
    if not claim_token:
        return False
    ctx = await repo.resolve_user_context(
        activity["user_id"],
        activity.get("workspace_id"),
    )
    conversation_id = (ctx or {}).get("conversation_id") or activity.get(
        "conversation_id"
    )
    if not ctx or not conversation_id:
        await repo.save_companion_decision(
            activity["id"],
            claim_token=claim_token,
            state=state,
            next_companion_at=_random_due(),
            sent=False,
        )
        return False

    messages = await _recent_messages(str(conversation_id))
    latest_at = _aware(messages[-1]["created_at"]) if messages else None
    last_user_at = next(
        (
            _aware(message["created_at"])
            for message in reversed(messages)
            if message["role"] == "user"
        ),
        None,
    )
    last_companion_at = _aware(activity.get("last_companion_at"))
    if last_user_at and (not last_companion_at or last_user_at > last_companion_at):
        state["unanswered_count"] = 0

    unanswered = int(state.get("unanswered_count") or 0)
    pause_after = max(
        1,
        int(settings.offline_activity_companion_pause_after_ignored),
    )
    if unanswered >= pause_after:
        state["mode"] = "paused_unanswered"
        await repo.save_companion_decision(
            activity["id"],
            claim_token=claim_token,
            state=state,
            next_companion_at=None,
            sent=False,
        )
        _log_decision(activity, "silent", "paused_unanswered")
        return False

    active_window = timedelta(
        minutes=max(
            1,
            int(settings.offline_activity_companion_active_chat_minutes),
        )
    )
    if latest_at and _now() - latest_at < active_window:
        await repo.save_companion_decision(
            activity["id"],
            claim_token=claim_token,
            state=state,
            next_companion_at=_random_due(),
            sent=False,
        )
        _log_decision(activity, "silent", "active_conversation")
        return False

    conditions = await repo.list_untriggered_conditions(activity["id"])
    focus_id = str(activity.get("focus_condition_id") or "")
    focus = next(
        (item for item in conditions if str(item.get("id")) == focus_id),
        None,
    )
    if not focus and conditions:
        focus = random.choice(conditions)
        await repo.set_activity_focus(
            activity["id"],
            str(focus["id"]),
            reason="companion_recovered_focus",
        )
    safe_hint = safe_guidance(focus, "weak") if focus else ""
    recent_modes = [
        str(mode)
        for mode in (state.get("recent_modes") or [])
        if str(mode) in _ACTIONS
    ][-3:]
    decision = await _generate_decision(
        activity=activity,
        messages=messages,
        safe_hint=safe_hint,
        activity_phase="free_roam" if not conditions else "guided",
        recent_modes=recent_modes,
        unanswered=unanswered,
    )
    if await _new_user_message_arrived(str(conversation_id), last_user_at):
        state["unanswered_count"] = 0
        await repo.save_companion_decision(
            activity["id"],
            claim_token=claim_token,
            state=state,
            next_companion_at=_random_due(),
            sent=False,
        )
        _log_decision(activity, "silent", "user_replied_during_generation")
        return False
    action = str(decision.get("action") or "silent")
    if action not in _ACTIONS:
        action = "silent"
    if action == "gentle_hint" and not focus:
        action = "silent"
        decision["reason"] = "no_remaining_focus"
    if action == "gentle_hint" and "gentle_hint" in recent_modes[-3:]:
        action = "silent"
        decision["reason"] = "task_hint_ratio_guard"

    delay = _clamp_delay(decision.get("next_delay_minutes"))
    next_due = _now() + timedelta(minutes=delay)
    if action == "silent":
        await repo.save_companion_decision(
            activity["id"],
            claim_token=claim_token,
            state=state,
            next_companion_at=next_due,
            sent=False,
        )
        _log_decision(activity, "silent", "model_silent")
        return False
    text = await _generate_companion_message(
        action=action,
        activity=activity,
        ctx=ctx,
        messages=messages,
        safe_hint=safe_hint,
    )

    from app.services.offline.recognition import _guard_visible_message

    message = await _guard_visible_message(
        text,
        conditions,
        safe_hint=safe_hint,
        fallback="逛得怎么样啦？不用赶，累了就找个地方歇一会儿。",
    )
    recent_modes = (recent_modes + [action])[-3:]
    state.update(
        {
            "mode": "free_roam" if not conditions else "guided",
            "last_action": action,
            "recent_modes": recent_modes,
            "unanswered_count": unanswered + 1,
        }
    )
    delivery_key = uuid4().hex
    reserved = await repo.reserve_companion_send(
        activity["id"],
        claim_token=claim_token,
        delivery_key=delivery_key,
        state=state,
        next_companion_at=next_due,
    )
    if not reserved:
        _log_decision(activity, "silent", "claim_fenced")
        return False
    try:
        emitted = await chat_emit.emit_assistant(
            conversation_id=str(conversation_id),
            user_id=activity["user_id"],
            agent_id=str(ctx["agent_id"]),
            workspace_id=activity.get("workspace_id"),
            message=message,
            real_world_type="activity",
            source_id=activity["id"],
            trigger_type=_TRIGGER_BY_ACTION.get(
                action,
                "offline_activity_companion_social",
            ),
            extra_metadata={
                "offline_companion_mode": action,
                "offline_companion_delivery_key": delivery_key,
            },
            guard_delivery_key=delivery_key,
        )
    except Exception:
        if not await repo.companion_message_exists(delivery_key):
            await repo.reschedule_failed_companion_delivery(
                activity["id"],
                delivery_key=delivery_key,
                next_companion_at=_now() + timedelta(minutes=5),
                previous_last_companion_at=activity.get("last_companion_at"),
            )
        raise
    if not emitted:
        await repo.cancel_guarded_companion_delivery(
            activity["id"],
            delivery_key=delivery_key,
            previous_last_companion_at=activity.get("last_companion_at"),
        )
        return False

    _log_decision(activity, action, f"model_{action}")
    return True


async def _generate_decision(
    *,
    activity: dict[str, Any],
    messages: list[dict[str, Any]],
    safe_hint: str,
    activity_phase: str,
    recent_modes: list[str],
    unanswered: int,
) -> dict[str, Any]:
    arrived = _aware(activity.get("arrival_confirmed_at")) or _now()
    elapsed = max(0, int((_now() - arrived).total_seconds() // 60))
    prompt = (await get_prompt_text("offline.activity_companion_decision")).format(
        activity_title=activity.get("title") or "这次外出",
        location_name=activity.get("location_name") or activity.get("address") or "现场",
        elapsed_minutes=elapsed,
        activity_phase=activity_phase,
        recent_modes="、".join(recent_modes) or "（无）",
        unanswered_count=unanswered,
        safe_hint=safe_hint or "（无，当前只陪伴闲聊）",
        recent_dialogue=_format_dialogue(messages) or "（无）",
        min_delay_minutes=max(
            1,
            int(settings.offline_activity_companion_min_interval_minutes),
        ),
        max_delay_minutes=max(
            int(settings.offline_activity_companion_min_interval_minutes),
            int(settings.offline_activity_companion_max_interval_minutes),
        ),
    )
    try:
        raw = await invoke_text(get_chat_model(), prompt)
        parsed = _parse_json_object(raw)
        if str(parsed.get("action") or "") not in _ACTIONS:
            raise CompanionDecisionGenerationError("invalid companion decision")
        return parsed
    except CompanionDecisionGenerationError:
        raise
    except Exception as exc:
        logger.warning("[offline-companion] decision generation failed: %s", exc)
        raise CompanionDecisionGenerationError(
            "companion decision generation failed"
        ) from exc


async def _generate_companion_message(
    *,
    action: str,
    activity: dict[str, Any],
    ctx: dict[str, Any],
    messages: list[dict[str, Any]],
    safe_hint: str,
) -> str:
    mbti = ctx.get("agent_mbti")
    mbti_type = str(mbti.get("type") or "").strip() if isinstance(mbti, dict) else ""
    agent_style = "，".join(
        part
        for part in (
            str(ctx.get("agent_name") or "").strip(),
            str(ctx.get("agent_occupation") or "").strip(),
            mbti_type,
        )
        if part
    ) or "像熟悉用户的朋友一样自然说话"
    prompt = (await get_prompt_text("offline.activity_companion_message")).format(
        agent_style=agent_style,
        user_name=ctx.get("user_name") or "你",
        activity_title=activity.get("title") or "这次外出",
        location_name=activity.get("location_name")
        or activity.get("address")
        or "现场",
        action=action,
        safe_hint=safe_hint if action == "gentle_hint" else "（本轮不使用）",
        recent_dialogue=_format_dialogue(messages) or "（无）",
    )
    try:
        parsed = _parse_json_object(await invoke_text(get_chat_model(), prompt))
        text = str(parsed.get("text") or "").strip()
        if not text:
            raise CompanionDecisionGenerationError("empty companion message")
        return text
    except CompanionDecisionGenerationError:
        raise
    except Exception as exc:
        logger.warning("[offline-companion] message generation failed: %s", exc)
        raise CompanionDecisionGenerationError(
            "companion message generation failed"
        ) from exc


async def _recent_messages(
    conversation_id: str,
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    rows = await db.message.find_many(
        where={"conversationId": conversation_id},
        order={"createdAt": "desc"},
        take=limit,
    )
    return [
        {
            "role": str(getattr(row, "role", "") or ""),
            "content": str(getattr(row, "content", "") or "").strip()[:240],
            "created_at": getattr(row, "createdAt", None),
        }
        for row in reversed(rows or [])
    ]


async def _new_user_message_arrived(
    conversation_id: str,
    previous_last_user_at: datetime | None,
) -> bool:
    row = await db.message.find_first(
        where={"conversationId": conversation_id, "role": "user"},
        order={"createdAt": "desc"},
    )
    current = _aware(getattr(row, "createdAt", None)) if row else None
    if current is None:
        return False
    return previous_last_user_at is None or current > previous_last_user_at


def _format_dialogue(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        text = str(message.get("content") or "").replace("\n", " ").strip()
        if not text:
            continue
        role = "用户" if message.get("role") == "user" else "AI"
        lines.append(f"{role}：{text[:160]}")
    return "\n".join(lines)[-1800:]


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text).rstrip("`").strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _clamp_delay(value: Any) -> int:
    low = max(1, int(settings.offline_activity_companion_min_interval_minutes))
    high = max(low, int(settings.offline_activity_companion_max_interval_minutes))
    try:
        delay = int(value)
    except (TypeError, ValueError):
        delay = random.randint(low, high)
    return max(low, min(high, delay))


def _log_decision(
    activity: dict[str, Any],
    action: str,
    reason: str,
) -> None:
    reason_code = reason if reason in _REASON_CODES else f"model_{action}"
    logger.info(
        "[offline-companion] action=%s reason=%s",
        action,
        reason_code,
        extra={
            "event": EVT_OFFLINE_COMPANION_DECISION,
            "activity_id": activity.get("id"),
            "decision": action,
            "reason": reason_code,
        },
    )
