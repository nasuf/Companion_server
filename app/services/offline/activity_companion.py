"""Conversation-first companionship for an active offline activity.

Send timing follows the activity-period probability windows. A hit still
uses the friend-like message prompt and the hidden-target rewrite. The
model does not decide whether to speak.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass
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
from app.services.schedule_domain.schedule import (
    get_cached_schedule,
    get_current_status,
)

logger = logging.getLogger(__name__)

# Minutes are half-open: [start, end). Attempt 0 is the first opening line.
# The third opening attempt uses the document's section 3.3 column.
OPENING_WINDOWS: tuple[tuple[tuple[int, int, float], ...], ...] = (
    (
        (0, 1, 0.80),
        (1, 2, 0.65),
        (2, 3, 0.50),
        (3, 7, 0.35),
        (7, 12, 0.18),
    ),
    (
        (0, 1, 0.55),
        (1, 2, 0.40),
        (2, 3, 0.28),
        (3, 7, 0.18),
        (7, 12, 0.10),
    ),
    (
        (0, 1, 0.40),
        (1, 2, 0.28),
        (2, 3, 0.18),
        (3, 7, 0.12),
        (7, 12, 0.06),
    ),
)
FOLLOWUP_WINDOWS: tuple[tuple[tuple[int, int, float], ...], ...] = (
    ((0, 2, 0.55), (2, 5, 0.35), (5, 10, 0.20)),
    ((0, 3, 0.40), (3, 8, 0.25), (8, 15, 0.12)),
)
ACTIVITY_CAP = 5
OPENING_CAP = 3
FOLLOWUP_CAP = 2
AI_MESSAGE_GAP = timedelta(minutes=2)
_PHASES = {"opening", "followup", "stopped"}
_SPEAK_ACTIONS = ("social", "ambient", "care")
_TRIGGER_BY_ACTION = {
    "social": "offline_activity_companion_social",
    "ambient": "offline_activity_companion_ambient",
    "gentle_hint": "offline_activity_companion_hint",
    "care": "offline_activity_companion_care",
}
_REASON_CODES = {
    "opening_migrated",
    "awaiting_reply",
    "window_expired",
    "ai_busy",
    "ai_gap",
    "probability_miss",
    "not_due",
    "segment_done",
    "activity_cap",
    "stopped",
    "user_replied_during_generation",
    "claim_fenced",
    "generation_failed",
    "missing_context",
    "sent_social",
    "sent_ambient",
    "sent_gentle_hint",
    "sent_care",
}
_DELIVERY_KEYS = (
    "delivery_key",
    "delivery_reserved_at",
    "delivery_canceled",
    "delivery_failed",
)


class CompanionDecisionGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class WindowSlot:
    index: int
    anchor: datetime
    start: datetime
    end: datetime
    due: datetime
    probability: float


@dataclass(frozen=True)
class TimingDecision:
    outcome: str
    state: dict[str, Any]
    next_at: datetime | None
    reason: str


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


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()


def _unit(rng: random.Random | None) -> float:
    if rng is None:
        return random.random()
    return rng.random()


def _wait_minutes(rng: random.Random | None) -> int:
    if rng is None:
        return random.randint(1, 3)
    return rng.randint(1, 3)


def _windows(phase: str, segment_sent: int) -> tuple[tuple[int, int, float], ...]:
    tables = OPENING_WINDOWS if phase == "opening" else FOLLOWUP_WINDOWS
    if segment_sent < 0 or segment_sent >= len(tables):
        return ()
    return tables[segment_sent]


def _segment_cap(phase: str) -> int:
    return OPENING_CAP if phase == "opening" else FOLLOWUP_CAP


def _strip_delivery(state: dict[str, Any]) -> None:
    for key in _DELIVERY_KEYS:
        state.pop(key, None)


def plan_slot(
    anchor: datetime,
    phase: str,
    segment_sent: int,
    window_index: int,
    rng: random.Random | None = None,
) -> WindowSlot | None:
    """Pick one random instant inside a single probability window."""
    windows = _windows(phase, segment_sent)
    if window_index < 0 or window_index >= len(windows):
        return None
    start_m, end_m, probability = windows[window_index]
    start = anchor + timedelta(minutes=start_m)
    end = anchor + timedelta(minutes=end_m)
    span = max((end - start).total_seconds(), 0.0)
    due = start + timedelta(seconds=_unit(rng) * span)
    return WindowSlot(
        index=window_index,
        anchor=anchor,
        start=start,
        end=end,
        due=due,
        probability=probability,
    )


def _apply_slot(state: dict[str, Any], slot: WindowSlot) -> dict[str, Any]:
    updated = dict(state)
    updated.update(
        {
            "window_index": slot.index,
            "anchor_at": _iso(slot.anchor),
            "window_start": _iso(slot.start),
            "window_end": _iso(slot.end),
            "due_at": _iso(slot.due),
            "probability": slot.probability,
        }
    )
    return updated


def begin_opening(
    anchor: datetime,
    rng: random.Random | None = None,
) -> tuple[dict[str, Any], datetime]:
    slot = plan_slot(anchor, "opening", 0, 0, rng)
    if slot is None:
        raise RuntimeError("opening window table is empty")
    state = _apply_slot(
        {
            "phase": "opening",
            "segment_sent": 0,
            "activity_sent": 0,
            "recent_modes": [],
            "awaiting_passive_reply": False,
        },
        slot,
    )
    return state, slot.due


def begin_followup(
    reply_at: datetime,
    *,
    activity_sent: int,
    recent_modes: list[str],
    rng: random.Random | None = None,
) -> tuple[dict[str, Any], datetime]:
    """Start the follow-up table after a 1–3 minute wait past the reply."""
    anchor = reply_at + timedelta(minutes=_wait_minutes(rng))
    slot = plan_slot(anchor, "followup", 0, 0, rng)
    if slot is None:
        raise RuntimeError("follow-up window table is empty")
    state = _apply_slot(
        {
            "phase": "followup",
            "segment_sent": 0,
            "activity_sent": int(activity_sent),
            "recent_modes": [str(mode) for mode in recent_modes][-3:],
            "awaiting_passive_reply": False,
        },
        slot,
    )
    return state, slot.due


def advance_window(
    state: dict[str, Any],
    rng: random.Random | None = None,
    *,
    reason: str,
) -> TimingDecision:
    """Move to the next window without consuming a send."""
    phase = str(state.get("phase") or "opening")
    segment_sent = int(state.get("segment_sent") or 0)
    window_index = int(state.get("window_index") or 0) + 1
    anchor = _aware(state.get("anchor_at")) or _now()
    slot = plan_slot(anchor, phase, segment_sent, window_index, rng)
    if slot is None:
        stopped = dict(state)
        stopped["phase"] = "stopped"
        stopped["window_index"] = window_index
        _strip_delivery(stopped)
        return TimingDecision("stop", stopped, None, "segment_done")
    updated = _apply_slot(state, slot)
    _strip_delivery(updated)
    return TimingDecision("defer", updated, slot.due, reason)


def evaluate_tick(
    state: dict[str, Any],
    now: datetime,
    *,
    ai_idle: bool,
    last_ai_at: datetime | None,
    rng: random.Random | None = None,
) -> TimingDecision:
    """Decide whether this scan sends, waits, or skips the current window.

    A skipped window does not increment the activity count. A 1-minute
    window that is already over is a miss and is not rolled late.
    """
    state = dict(state)
    phase = str(state.get("phase") or "stopped")
    activity_sent = int(state.get("activity_sent") or 0)
    segment_sent = int(state.get("segment_sent") or 0)
    if phase == "stopped" or activity_sent >= ACTIVITY_CAP:
        state["phase"] = "stopped"
        reason = "activity_cap" if activity_sent >= ACTIVITY_CAP else "stopped"
        return TimingDecision("stop", state, None, reason)
    if phase not in _PHASES or segment_sent >= _segment_cap(phase):
        state["phase"] = "stopped"
        return TimingDecision("stop", state, None, "segment_done")

    window_end = _aware(state.get("window_end"))
    due = _aware(state.get("due_at"))
    if window_end is None or due is None:
        return advance_window(state, rng, reason="window_expired")
    if now >= window_end:
        return advance_window(state, rng, reason="window_expired")
    if not ai_idle:
        retry = min(now + timedelta(minutes=1), window_end - timedelta(seconds=1))
        if retry <= now:
            return advance_window(state, rng, reason="ai_busy")
        return TimingDecision("defer", state, retry, "ai_busy")
    if now < due:
        return TimingDecision("defer", state, due, "not_due")
    if last_ai_at is not None and due < last_ai_at + AI_MESSAGE_GAP:
        return advance_window(state, rng, reason="ai_gap")
    probability = float(state.get("probability") or 0.0)
    if _unit(rng) < probability:
        return TimingDecision("send", state, due, "send")
    return advance_window(state, rng, reason="probability_miss")


def retry_current_window(
    state: dict[str, Any],
    now: datetime,
    rng: random.Random | None = None,
) -> TimingDecision:
    """Generation failed: retry inside the open window, otherwise skip it."""
    window_end = _aware(state.get("window_end"))
    retry = now + timedelta(seconds=30)
    if window_end and retry < window_end:
        return TimingDecision("defer", dict(state), retry, "generation_failed")
    return advance_window(state, rng, reason="generation_failed")


def state_after_send(
    state: dict[str, Any],
    sent_at: datetime,
    action: str,
    rng: random.Random | None = None,
) -> tuple[dict[str, Any], datetime | None]:
    activity_sent = int(state.get("activity_sent") or 0) + 1
    segment_sent = int(state.get("segment_sent") or 0) + 1
    recent = [str(mode) for mode in (state.get("recent_modes") or [])]
    recent = (recent + [action])[-3:]
    phase = str(state.get("phase") or "opening")
    updated = dict(state)
    updated.update(
        {
            "activity_sent": activity_sent,
            "segment_sent": segment_sent,
            "recent_modes": recent,
            "last_action": action,
            "awaiting_passive_reply": False,
        }
    )
    _strip_delivery(updated)
    if activity_sent >= ACTIVITY_CAP or segment_sent >= _segment_cap(phase):
        updated["phase"] = "stopped"
        return updated, None
    slot = plan_slot(sent_at, phase, segment_sent, 0, rng)
    if slot is None:
        updated["phase"] = "stopped"
        return updated, None
    return _apply_slot(updated, slot), slot.due


def choose_companion_action(
    recent_modes: list[str],
    *,
    has_focus: bool,
) -> str:
    """Cycle casual lines. A focus hint is allowed only once per three sends."""
    recent = [
        mode
        for mode in recent_modes
        if mode in _TRIGGER_BY_ACTION
    ][-3:]
    last = next((mode for mode in reversed(recent) if mode in _SPEAK_ACTIONS), None)
    if has_focus and "gentle_hint" not in recent and last == "care":
        return "gentle_hint"
    if last is None:
        return "social"
    index = _SPEAK_ACTIONS.index(last)
    return _SPEAK_ACTIONS[(index + 1) % len(_SPEAK_ACTIONS)]


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


async def start_opening_segment(recommendation_id: str, user_id: str) -> None:
    """Start the opening clock after the arrival guide has been stored."""
    if not settings.offline_activity_companion_enabled:
        return
    current = await repo.get_activity(
        recommendation_id,
        user_id,
        reveal_task=True,
    )
    if not current or not current.get("reached"):
        return
    state = dict(current.get("companion_state") or {})
    if state.get("awaiting_passive_reply"):
        return
    if state.get("phase") in _PHASES:
        return
    new_state, due = begin_opening(_now())
    await repo.schedule_opening_unless_replied(recommendation_id, new_state, due)


async def note_user_interaction(
    user_id: str,
    workspace_id: str | None,
) -> None:
    """Cancel a not-yet-sent companion line. Do not start the follow-up clock."""
    if not settings.offline_activity_companion_enabled:
        return
    activity = await repo.get_current_reached_activity(user_id, workspace_id)
    if not activity:
        return
    await repo.cancel_pending_companion(activity["id"], uuid4().hex)


async def note_passive_reply(conversation_id: str) -> None:
    """Start follow-up timing after the reply to a cancelling user turn is stored."""
    try:
        await _note_passive_reply(conversation_id)
    except Exception as exc:  # noqa: BLE001 - reply persistence must not depend on this
        logger.debug("[offline-companion] passive reply hook skipped: %s", exc)


async def _note_passive_reply(conversation_id: str) -> None:
    if not settings.offline_activity_companion_enabled or not conversation_id:
        return
    conv = await db.conversation.find_unique(where={"id": conversation_id})
    if conv is None:
        return
    user_id = getattr(conv, "userId", None)
    workspace_id = getattr(conv, "workspaceId", None)
    if not user_id:
        return
    activity = await repo.get_current_reached_activity(user_id, workspace_id)
    if not activity:
        return
    state = dict(activity.get("companion_state") or {})
    reply_token = str(state.get("reply_token") or "")
    if not state.get("awaiting_passive_reply") or not reply_token:
        return
    new_state, due = begin_followup(
        _now(),
        activity_sent=int(state.get("activity_sent") or 0),
        recent_modes=list(state.get("recent_modes") or []),
    )
    await repo.schedule_followup_if_token(
        activity["id"],
        new_state,
        due,
        reply_token,
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
                        next_companion_at=_now() + timedelta(minutes=1),
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
    if state.get("phase") not in _PHASES:
        new_state, due = begin_opening(_now())
        await repo.save_companion_decision(
            activity["id"],
            claim_token=claim_token,
            state=new_state,
            next_companion_at=due,
            sent=False,
        )
        _log_decision(activity, "silent", "opening_migrated")
        return False
    if state.get("awaiting_passive_reply"):
        await repo.save_companion_decision(
            activity["id"],
            claim_token=claim_token,
            state=state,
            next_companion_at=None,
            sent=False,
        )
        _log_decision(activity, "silent", "awaiting_reply")
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
            next_companion_at=_now() + timedelta(minutes=1),
            sent=False,
        )
        _log_decision(activity, "silent", "missing_context")
        return False

    messages = await _recent_messages(str(conversation_id))
    last_user_at = next(
        (
            _aware(message["created_at"])
            for message in reversed(messages)
            if message["role"] == "user"
        ),
        None,
    )
    last_ai_at = await _latest_assistant_at(str(conversation_id))
    ai_idle = await _ai_is_idle(str(ctx["agent_id"]))
    decision = evaluate_tick(
        state,
        _now(),
        ai_idle=ai_idle,
        last_ai_at=last_ai_at,
    )
    if decision.outcome != "send":
        await repo.save_companion_decision(
            activity["id"],
            claim_token=claim_token,
            state=decision.state,
            next_companion_at=decision.next_at,
            sent=False,
        )
        _log_decision(activity, "silent", decision.reason)
        return False

    if await _new_user_message_arrived(str(conversation_id), last_user_at):
        await _hold_for_user_reply(activity["id"], claim_token, state)
        _log_decision(activity, "silent", "user_replied_during_generation")
        return False

    conditions = await repo.list_untriggered_conditions(activity["id"])
    focus = await _resolve_focus(activity, conditions)
    safe_hint = safe_guidance(focus, "weak") if focus else ""
    recent_modes = [
        str(mode)
        for mode in (state.get("recent_modes") or [])
        if str(mode) in _TRIGGER_BY_ACTION
    ][-3:]
    action = choose_companion_action(recent_modes, has_focus=focus is not None)
    try:
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
    except Exception as exc:  # noqa: BLE001 - a failed line must not consume the window
        logger.warning("[offline-companion] message generation failed: %s", exc)
        retry = retry_current_window(state, _now())
        await repo.save_companion_decision(
            activity["id"],
            claim_token=claim_token,
            state=retry.state,
            next_companion_at=retry.next_at,
            sent=False,
        )
        _log_decision(activity, "silent", "generation_failed")
        return False

    if await _new_user_message_arrived(str(conversation_id), last_user_at):
        await _hold_for_user_reply(activity["id"], claim_token, state)
        _log_decision(activity, "silent", "user_replied_during_generation")
        return False

    sent_at = _now()
    sent_state, next_due = state_after_send(state, sent_at, action)
    delivery_key = uuid4().hex
    # Lease past this emit. Counts stay on the pre-send state until the insert lands.
    reserved = await repo.reserve_companion_send(
        activity["id"],
        claim_token=claim_token,
        delivery_key=delivery_key,
        state=state,
        next_companion_at=sent_at + timedelta(minutes=15),
    )
    if not reserved:
        _log_decision(activity, "silent", "claim_fenced")
        return False
    if await _new_user_message_arrived(str(conversation_id), last_user_at):
        # The insert guard only sees messages newer than delivery_reserved_at.
        # A message that landed during reserve is older than that timestamp.
        await _restore_held_reply(
            activity,
            delivery_key=delivery_key,
            state=state,
        )
        _log_decision(activity, "silent", "user_replied_during_generation")
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
            retry = retry_current_window(state, _now())
            await repo.restore_companion_schedule(
                activity["id"],
                delivery_key=delivery_key,
                state=retry.state,
                next_companion_at=retry.next_at,
                previous_last_companion_at=activity.get("last_companion_at"),
            )
        raise
    if not emitted:
        retry = retry_current_window(state, _now())
        await repo.restore_companion_schedule(
            activity["id"],
            delivery_key=delivery_key,
            state=retry.state,
            next_companion_at=retry.next_at,
            previous_last_companion_at=activity.get("last_companion_at"),
        )
        return False
    await repo.mark_companion_sent(
        activity["id"],
        delivery_key,
        sent_state,
        next_due,
    )
    _log_decision(activity, action, f"sent_{action}")
    return True


async def _restore_held_reply(
    activity: dict[str, Any],
    *,
    delivery_key: str,
    state: dict[str, Any],
) -> None:
    held = dict(state)
    held["awaiting_passive_reply"] = True
    held["reply_token"] = uuid4().hex
    _strip_delivery(held)
    await repo.restore_companion_schedule(
        activity["id"],
        delivery_key=delivery_key,
        state=held,
        next_companion_at=None,
        previous_last_companion_at=activity.get("last_companion_at"),
    )


async def _hold_for_user_reply(
    recommendation_id: str,
    claim_token: str,
    state: dict[str, Any],
) -> None:
    """Stop the current window until this turn's reply persists.

    If the user hook already cleared the claim, this save does not land and
    that hook's reply token remains the one the follow-up clock must match.
    """
    held = dict(state)
    held["awaiting_passive_reply"] = True
    held["reply_token"] = uuid4().hex
    _strip_delivery(held)
    await repo.save_companion_decision(
        recommendation_id,
        claim_token=claim_token,
        state=held,
        next_companion_at=None,
        sent=False,
    )


async def _resolve_focus(
    activity: dict[str, Any],
    conditions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    focus_id = str(activity.get("focus_condition_id") or "")
    focus = next(
        (item for item in conditions if str(item.get("id")) == focus_id),
        None,
    )
    if focus or not conditions:
        return focus
    focus = random.choice(conditions)
    await repo.set_activity_focus(
        activity["id"],
        str(focus["id"]),
        reason="companion_recovered_focus",
    )
    return focus


async def _ai_is_idle(agent_id: str) -> bool:
    """Missing schedule counts as idle so companionship is not silently blocked."""
    schedule = await get_cached_schedule(agent_id)
    if not schedule:
        return True
    return get_current_status(schedule).get("status") == "idle"


async def _latest_assistant_at(conversation_id: str) -> datetime | None:
    row = await db.message.find_first(
        where={"conversationId": conversation_id, "role": "assistant"},
        order={"createdAt": "desc"},
    )
    if row is None:
        return None
    return _aware(getattr(row, "createdAt", None))


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


def _log_decision(
    activity: dict[str, Any],
    action: str,
    reason: str,
) -> None:
    reason_code = reason if reason in _REASON_CODES else "stopped"
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
