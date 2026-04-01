"""主动聊天状态服务。

第一阶段目标：
- 为每个 active workspace 持久化一条主动状态记录
- 在对话结束后记录新的 t0
- 在用户再次回复时重置等待态
- 为后续区间概率命中和衰减状态机提供稳定底座
"""

from __future__ import annotations

import json
import logging
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.db import db

logger = logging.getLogger(__name__)

UTC = timezone.utc

PROACTIVE_WINDOWS: tuple[dict[str, float | int | str], ...] = (
    {"index": 0, "name": "0-30m", "start_sec": 0, "end_sec": 1800, "hit_rate": 0.00},
    {"index": 1, "name": "30m-1h", "start_sec": 1800, "end_sec": 3600, "hit_rate": 0.05},
    {"index": 2, "name": "1h-2h", "start_sec": 3600, "end_sec": 7200, "hit_rate": 0.12},
    {"index": 3, "name": "2h-4h", "start_sec": 7200, "end_sec": 14400, "hit_rate": 0.25},
    {"index": 4, "name": "4h-6h", "start_sec": 14400, "end_sec": 21600, "hit_rate": 0.35},
)


@dataclass
class ProactiveStateRecord:
    id: str
    workspace_id: str
    user_id: str
    agent_id: str
    conversation_id: str | None
    status: str
    stage: str
    silence_level_n: int
    followup_plan_type: str
    remaining_forced_triggers: int | None
    current_window_index: int | None
    window_due_at: datetime | None
    response_deadline_at: datetime | None
    t0_at: datetime | None
    last_proactive_at: datetime | None
    last_user_reply_at: datetime | None
    last_assistant_reply_at: datetime | None
    last_attempt_at: datetime | None
    daily_scene_triggered_at: datetime | None
    stop_reason: str | None
    metadata: dict[str, Any] | None


def _now(now: datetime | None = None) -> datetime:
    ts = now or datetime.now(UTC)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(value))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        return None


def _parse_json(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _row_to_state(row: dict[str, Any]) -> ProactiveStateRecord:
    return ProactiveStateRecord(
        id=str(row["id"]),
        workspace_id=str(row["workspace_id"]),
        user_id=str(row["user_id"]),
        agent_id=str(row["agent_id"]),
        conversation_id=row.get("conversation_id"),
        status=str(row.get("status") or "idle"),
        stage=str(row.get("stage") or "cold_start"),
        silence_level_n=int(row.get("silence_level_n") or 0),
        followup_plan_type=str(row.get("followup_plan_type") or "normal"),
        remaining_forced_triggers=(int(row["remaining_forced_triggers"]) if row.get("remaining_forced_triggers") is not None else None),
        current_window_index=(int(row["current_window_index"]) if row.get("current_window_index") is not None else None),
        window_due_at=_parse_dt(row.get("window_due_at")),
        response_deadline_at=_parse_dt(row.get("response_deadline_at")),
        t0_at=_parse_dt(row.get("t0_at")),
        last_proactive_at=_parse_dt(row.get("last_proactive_at")),
        last_user_reply_at=_parse_dt(row.get("last_user_reply_at")),
        last_assistant_reply_at=_parse_dt(row.get("last_assistant_reply_at")),
        last_attempt_at=_parse_dt(row.get("last_attempt_at")),
        daily_scene_triggered_at=_parse_dt(row.get("daily_scene_triggered_at")),
        stop_reason=row.get("stop_reason"),
        metadata=_parse_json(row.get("metadata")),
    )


def _window_name(index: int | None) -> str | None:
    if index is None:
        return None
    for window in PROACTIVE_WINDOWS:
        if int(window["index"]) == index:
            return str(window["name"])
    return None


def _log_if_unavailable(action: str, error: Exception) -> None:
    logger.info(f"Proactive state {action} skipped: {error}")


def _pick_random_due_at(t0_at: datetime, window_index: int, now: datetime) -> datetime:
    for window in PROACTIVE_WINDOWS:
        if int(window["index"]) != window_index:
            continue
        start_at = t0_at + timedelta(seconds=float(window["start_sec"]))
        end_at = t0_at + timedelta(seconds=float(window["end_sec"]))
        floor = max(start_at, now)
        if floor >= end_at:
            return end_at
        span = max(1.0, (end_at - floor).total_seconds())
        return floor + timedelta(seconds=random.uniform(0, span))
    return now


def _pick_random_future_due_at(
    now: datetime,
    *,
    end_at: datetime,
) -> datetime:
    floor = now
    if floor >= end_at:
        return end_at
    span = max(1.0, (end_at - floor).total_seconds())
    return floor + timedelta(seconds=random.uniform(0, span))


async def _fetch_workspace_context(workspace_id: str) -> dict[str, Any] | None:
    try:
        rows = await db.query_raw(
            """
            SELECT
                w.id AS workspace_id,
                w.user_id,
                w.agent_id,
                c.id AS conversation_id
            FROM chat_workspaces w
            LEFT JOIN conversations c
              ON c.workspace_id = w.id
             AND c.is_deleted = FALSE
            WHERE w.id = $1
              AND w.status = 'active'
            ORDER BY c.updated_at DESC NULLS LAST, c.created_at DESC NULLS LAST
            LIMIT 1
            """,
            workspace_id,
        )
    except Exception as e:
        _log_if_unavailable("fetch workspace context", e)
        return None
    return rows[0] if rows else None


async def _count_workspace_memories(workspace_id: str) -> tuple[int, int]:
    try:
        rows = await db.query_raw(
            """
            SELECT
                COALESCE(SUM(CASE WHEN level = 1 THEN 1 ELSE 0 END), 0) AS l1_count,
                COALESCE(SUM(CASE WHEN level = 2 THEN 1 ELSE 0 END), 0) AS l2_count
            FROM memories_user
            WHERE workspace_id = $1
              AND is_archived = FALSE
            """,
            workspace_id,
        )
    except Exception as e:
        _log_if_unavailable("count memories", e)
        return 0, 0
    row = rows[0] if rows else {}
    return int(row.get("l1_count") or 0), int(row.get("l2_count") or 0)


async def _load_topic_intimacy(agent_id: str, user_id: str) -> int:
    try:
        rows = await db.query_raw(
            """
            SELECT topic_intimacy
            FROM intimacies
            WHERE agent_id = $1 AND user_id = $2
            LIMIT 1
            """,
            agent_id,
            user_id,
        )
    except Exception as e:
        _log_if_unavailable("load topic intimacy", e)
        return 50
    if not rows:
        return 50
    return int(rows[0].get("topic_intimacy") or 50)


async def determine_proactive_stage(workspace_id: str, agent_id: str, user_id: str) -> str:
    l1_count, l2_count = await _count_workspace_memories(workspace_id)
    topic_intimacy = await _load_topic_intimacy(agent_id, user_id)

    if l1_count >= 15 and l2_count >= 35 and topic_intimacy >= 65:
        return "intimate"
    if l1_count >= 8 and l2_count >= 15:
        return "warming"
    return "cold_start"


async def get_active_workspace_context(workspace_id: str) -> dict[str, Any] | None:
    return await _fetch_workspace_context(workspace_id)


async def ensure_proactive_state_for_workspace(
    workspace_id: str,
    *,
    now: datetime | None = None,
    reason: str = "ensure",
) -> ProactiveStateRecord | None:
    existing = await get_proactive_state_by_workspace(workspace_id)
    if existing:
        return existing

    ctx = await get_active_workspace_context(workspace_id)
    if not ctx:
        return None
    state_id = await start_or_restart_proactive_session(
        workspace_id=workspace_id,
        conversation_id=ctx.get("conversation_id"),
        user_id=str(ctx["user_id"]),
        agent_id=str(ctx["agent_id"]),
        now=now,
        reason=reason,
    )
    if not state_id:
        return None
    return await get_proactive_state_by_workspace(workspace_id)


async def get_proactive_state_by_workspace(workspace_id: str) -> ProactiveStateRecord | None:
    try:
        rows = await db.query_raw(
            """
            SELECT *
            FROM proactive_states
            WHERE workspace_id = $1
            LIMIT 1
            """,
            workspace_id,
        )
    except Exception as e:
        _log_if_unavailable("get", e)
        return None
    if not rows:
        return None
    return _row_to_state(rows[0])


async def log_proactive_event(
    *,
    state_id: str,
    workspace_id: str,
    user_id: str,
    agent_id: str,
    conversation_id: str | None,
    event_type: str,
    window_index: int | None = None,
    trigger_type: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    try:
        await db.execute_raw(
            """
            INSERT INTO proactive_event_logs (
                id, state_id, workspace_id, user_id, agent_id, conversation_id,
                event_type, window_index, window_name, trigger_type, payload
            )
            VALUES (
                $1, $2, $3, $4, $5, $6,
                $7, $8, $9, $10, $11::jsonb
            )
            """,
            str(uuid.uuid4()),
            state_id,
            workspace_id,
            user_id,
            agent_id,
            conversation_id,
            event_type,
            window_index,
            _window_name(window_index),
            trigger_type,
            json.dumps(payload or {}, ensure_ascii=False),
        )
    except Exception as e:
        _log_if_unavailable(f"log {event_type}", e)


async def claim_due_proactive_state(
    state_id: str,
    *,
    now: datetime | None = None,
) -> ProactiveStateRecord | None:
    now_ts = _now(now)
    try:
        rows = await db.query_raw(
            """
            UPDATE proactive_states
            SET
                status = 'processing',
                last_attempt_at = $2,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $1
              AND status = 'running'
              AND window_due_at IS NOT NULL
              AND window_due_at <= $2
            RETURNING *
            """,
            state_id,
            now_ts,
        )
    except Exception as e:
        _log_if_unavailable("claim due", e)
        return None
    if not rows:
        return None
    return _row_to_state(rows[0])


async def claim_waiting_timeout_state(
    state_id: str,
    *,
    now: datetime | None = None,
) -> ProactiveStateRecord | None:
    now_ts = _now(now)
    try:
        rows = await db.query_raw(
            """
            UPDATE proactive_states
            SET
                status = 'processing_timeout',
                last_attempt_at = $2,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $1
              AND status = 'waiting_user'
              AND response_deadline_at IS NOT NULL
              AND response_deadline_at <= $2
            RETURNING *
            """,
            state_id,
            now_ts,
        )
    except Exception as e:
        _log_if_unavailable("claim waiting timeout", e)
        return None
    if not rows:
        return None
    return _row_to_state(rows[0])


async def start_or_restart_proactive_session(
    *,
    workspace_id: str,
    conversation_id: str | None,
    user_id: str,
    agent_id: str,
    now: datetime | None = None,
    reason: str = "conversation_end",
) -> str | None:
    now_ts = _now(now)
    stage = await determine_proactive_stage(workspace_id, agent_id, user_id)
    first_window_index = 1
    due_at = _pick_random_due_at(now_ts, first_window_index, now_ts)

    try:
        rows = await db.query_raw(
            """
            INSERT INTO proactive_states (
                id, workspace_id, user_id, agent_id, conversation_id,
                status, stage, silence_level_n, followup_plan_type, remaining_forced_triggers,
                current_window_index, window_due_at, response_deadline_at,
                t0_at, last_assistant_reply_at, stop_reason, metadata
            )
            VALUES (
                $1, $2, $3, $4, $5,
                'running', $6, 0, 'normal', NULL,
                $7, $8, NULL,
                $9, $9, NULL, $10::jsonb
            )
            ON CONFLICT (workspace_id)
            DO UPDATE SET
                conversation_id = EXCLUDED.conversation_id,
                user_id = EXCLUDED.user_id,
                agent_id = EXCLUDED.agent_id,
                status = 'running',
                stage = EXCLUDED.stage,
                followup_plan_type = 'normal',
                remaining_forced_triggers = NULL,
                current_window_index = EXCLUDED.current_window_index,
                window_due_at = EXCLUDED.window_due_at,
                response_deadline_at = NULL,
                t0_at = EXCLUDED.t0_at,
                last_assistant_reply_at = EXCLUDED.last_assistant_reply_at,
                updated_at = CURRENT_TIMESTAMP,
                stop_reason = NULL,
                metadata = EXCLUDED.metadata
            RETURNING id
            """,
            str(uuid.uuid4()),
            workspace_id,
            user_id,
            agent_id,
            conversation_id,
            stage,
            first_window_index,
            due_at,
            now_ts,
            json.dumps({"reason": reason}, ensure_ascii=False),
        )
    except Exception as e:
        _log_if_unavailable("session start", e)
        return None
    state_id = str(rows[0]["id"]) if rows else None
    if state_id:
        await log_proactive_event(
            state_id=state_id,
            workspace_id=workspace_id,
            user_id=user_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            event_type="session_started",
            window_index=first_window_index,
            payload={"reason": reason, "stage": stage, "due_at": due_at.isoformat()},
        )
    return state_id


async def mark_user_replied_for_conversation(
    conversation_id: str,
    *,
    replied_at: datetime | None = None,
) -> None:
    replied_ts = _now(replied_at)
    try:
        rows = await db.query_raw(
            """
            UPDATE proactive_states ps
            SET
                status = 'idle',
                silence_level_n = 0,
                followup_plan_type = 'normal',
                remaining_forced_triggers = NULL,
                current_window_index = NULL,
                window_due_at = NULL,
                response_deadline_at = NULL,
                last_user_reply_at = $2,
                updated_at = CURRENT_TIMESTAMP
            FROM conversations c
            WHERE c.id = $1
              AND ps.workspace_id = c.workspace_id
            RETURNING ps.id, ps.workspace_id, ps.user_id, ps.agent_id, ps.conversation_id
            """,
            conversation_id,
            replied_ts,
        )
    except Exception as e:
        _log_if_unavailable("mark user replied", e)
        return
    if not rows:
        return
    row = rows[0]
    await log_proactive_event(
        state_id=str(row["id"]),
        workspace_id=str(row["workspace_id"]),
        user_id=str(row["user_id"]),
        agent_id=str(row["agent_id"]),
        conversation_id=row.get("conversation_id"),
        event_type="user_replied",
        payload={"replied_at": replied_ts.isoformat()},
    )


async def list_due_proactive_states(now: datetime | None = None) -> list[ProactiveStateRecord]:
    now_ts = _now(now)
    try:
        rows = await db.query_raw(
            """
            SELECT *
            FROM proactive_states
            WHERE status = 'running'
              AND window_due_at IS NOT NULL
              AND window_due_at <= $1
            ORDER BY window_due_at ASC
            LIMIT 100
            """,
            now_ts,
        )
    except Exception as e:
        _log_if_unavailable("list due", e)
        return []
    return [_row_to_state(row) for row in rows]


async def list_waiting_timeout_states(now: datetime | None = None) -> list[ProactiveStateRecord]:
    now_ts = _now(now)
    try:
        rows = await db.query_raw(
            """
            SELECT *
            FROM proactive_states
            WHERE status = 'waiting_user'
              AND response_deadline_at IS NOT NULL
              AND response_deadline_at <= $1
            ORDER BY response_deadline_at ASC
            LIMIT 100
            """,
            now_ts,
        )
    except Exception as e:
        _log_if_unavailable("list waiting timeouts", e)
        return []
    return [_row_to_state(row) for row in rows]


async def has_recent_user_activity(workspace_id: str, *, now: datetime | None = None, window_minutes: int = 30) -> bool:
    now_ts = _now(now)
    since = now_ts - timedelta(minutes=window_minutes)
    try:
        rows = await db.query_raw(
            """
            SELECT 1
            FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            WHERE c.workspace_id = $1
              AND c.is_deleted = FALSE
              AND m.role = 'user'
              AND m.created_at >= $2
            LIMIT 1
            """,
            workspace_id,
            since,
        )
    except Exception as e:
        _log_if_unavailable("recent activity check", e)
        return False
    return bool(rows)


async def stop_proactive_state(
    state: ProactiveStateRecord,
    *,
    reason: str,
    now: datetime | None = None,
) -> None:
    now_ts = _now(now)
    try:
        await db.execute_raw(
            """
            UPDATE proactive_states
            SET
                status = 'idle',
                followup_plan_type = 'normal',
                remaining_forced_triggers = NULL,
                current_window_index = NULL,
                window_due_at = NULL,
                response_deadline_at = NULL,
                last_attempt_at = $2,
                stop_reason = $3,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $1
            """,
            state.id,
            now_ts,
            reason,
        )
    except Exception as e:
        _log_if_unavailable("stop", e)
        return
    await log_proactive_event(
        state_id=state.id,
        workspace_id=state.workspace_id,
        user_id=state.user_id,
        agent_id=state.agent_id,
        conversation_id=state.conversation_id,
        event_type="mutex_blocked",
        window_index=state.current_window_index,
        payload={"reason": reason},
    )


def is_same_local_day(left: datetime | None, right: datetime | None) -> bool:
    if not left or not right:
        return False
    left_dt = left if left.tzinfo else left.replace(tzinfo=UTC)
    right_dt = right if right.tzinfo else right.replace(tzinfo=UTC)
    return left_dt.astimezone(UTC).date() == right_dt.astimezone(UTC).date()


async def mark_proactive_sent(
    state: ProactiveStateRecord,
    *,
    trigger_type: str,
    message: str,
    assistant_message_id: str | None,
    now: datetime | None = None,
    mark_daily_scene: bool = False,
    response_timeout_hours: int = 24,
) -> None:
    now_ts = _now(now)
    remaining_forced = state.remaining_forced_triggers
    if state.followup_plan_type in {"seven_day_sparse", "thirty_day_final"} and remaining_forced is not None:
        remaining_forced = max(0, remaining_forced - 1)
    try:
        await db.execute_raw(
            """
            UPDATE proactive_states
            SET
                status = 'waiting_user',
                current_window_index = NULL,
                window_due_at = NULL,
                response_deadline_at = $3,
                last_proactive_at = $2,
                last_attempt_at = $2,
                t0_at = $2,
                remaining_forced_triggers = $4,
                daily_scene_triggered_at = CASE WHEN $5 THEN $2 ELSE daily_scene_triggered_at END,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $1
            """,
            state.id,
            now_ts,
            now_ts + timedelta(hours=response_timeout_hours),
            remaining_forced,
            mark_daily_scene,
        )
    except Exception as e:
        _log_if_unavailable("mark proactive sent", e)
        return

    await log_proactive_event(
        state_id=state.id,
        workspace_id=state.workspace_id,
        user_id=state.user_id,
        agent_id=state.agent_id,
        conversation_id=state.conversation_id,
        event_type="message_sent",
        window_index=state.current_window_index,
        trigger_type=trigger_type,
        payload={
            "assistant_message_id": assistant_message_id,
            "message": message,
            "response_deadline_at": (now_ts + timedelta(hours=response_timeout_hours)).isoformat(),
            "remaining_forced_triggers": remaining_forced,
        },
    )


def _metadata_with_window_end(state: ProactiveStateRecord, plan_window_end_at: datetime) -> str:
    current = dict(state.metadata or {})
    current["plan_window_end_at"] = plan_window_end_at.isoformat()
    return json.dumps(current, ensure_ascii=False)


async def escalate_waiting_state(
    state: ProactiveStateRecord,
    *,
    now: datetime | None = None,
) -> None:
    now_ts = _now(now)
    next_n = state.silence_level_n + 1

    if state.followup_plan_type == "seven_day_sparse":
        remaining = int(state.remaining_forced_triggers or 0)
        if remaining > 0:
            window_end = _parse_dt((state.metadata or {}).get("plan_window_end_at")) or (now_ts + timedelta(days=7))
            next_due = _pick_random_future_due_at(now_ts, end_at=window_end)
            await _resume_forced_plan(
                state,
                now_ts=now_ts,
                due_at=next_due,
                followup_plan_type="seven_day_sparse",
                remaining_forced_triggers=remaining,
                metadata_json=_metadata_with_window_end(state, window_end),
                event_type="silence_plan_resumed",
                payload={"plan": "seven_day_sparse", "remaining": remaining, "due_at": next_due.isoformat()},
            )
            return
        next_n = max(next_n, 6)

    if state.followup_plan_type == "thirty_day_final":
        remaining = int(state.remaining_forced_triggers or 0)
        if remaining > 0:
            window_end = _parse_dt((state.metadata or {}).get("plan_window_end_at")) or (now_ts + timedelta(days=30))
            next_due = _pick_random_future_due_at(now_ts, end_at=window_end)
            await _resume_forced_plan(
                state,
                now_ts=now_ts,
                due_at=next_due,
                followup_plan_type="thirty_day_final",
                remaining_forced_triggers=remaining,
                metadata_json=_metadata_with_window_end(state, window_end),
                event_type="silence_plan_resumed",
                payload={"plan": "thirty_day_final", "remaining": remaining, "due_at": next_due.isoformat()},
            )
            return
        await stop_proactive_state(state, reason="final_no_reply", now=now_ts)
        await log_proactive_event(
            state_id=state.id,
            workspace_id=state.workspace_id,
            user_id=state.user_id,
            agent_id=state.agent_id,
            conversation_id=state.conversation_id,
            event_type="permanently_stopped",
            payload={"reason": "final_no_reply"},
        )
        return

    if next_n <= 4:
        due_at = _pick_random_due_at(now_ts, 1, now_ts)
        try:
            await db.execute_raw(
                """
                UPDATE proactive_states
                SET
                    status = 'running',
                    silence_level_n = $2,
                    followup_plan_type = 'normal',
                    remaining_forced_triggers = NULL,
                    current_window_index = 1,
                    window_due_at = $3,
                    response_deadline_at = NULL,
                    t0_at = $4,
                    last_attempt_at = $4,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = $1
                """,
                state.id,
                next_n,
                due_at,
                now_ts,
            )
        except Exception as e:
            _log_if_unavailable("resume normal followup", e)
            return
        await log_proactive_event(
            state_id=state.id,
            workspace_id=state.workspace_id,
            user_id=state.user_id,
            agent_id=state.agent_id,
            conversation_id=state.conversation_id,
            event_type="silence_escalated",
            window_index=1,
            payload={"n": next_n, "plan": "normal", "due_at": due_at.isoformat()},
        )
        return

    if next_n == 5:
        window_end = now_ts + timedelta(days=7)
        next_due = _pick_random_future_due_at(now_ts, end_at=window_end)
        await _resume_forced_plan(
            state,
            now_ts=now_ts,
            due_at=next_due,
            followup_plan_type="seven_day_sparse",
            remaining_forced_triggers=2,
            silence_level_n=5,
            metadata_json=_metadata_with_window_end(state, window_end),
            event_type="silence_escalated",
            payload={"n": 5, "plan": "seven_day_sparse", "remaining": 2, "due_at": next_due.isoformat()},
        )
        return

    if next_n == 6:
        window_end = now_ts + timedelta(days=30)
        next_due = _pick_random_future_due_at(now_ts, end_at=window_end)
        await _resume_forced_plan(
            state,
            now_ts=now_ts,
            due_at=next_due,
            followup_plan_type="thirty_day_final",
            remaining_forced_triggers=1,
            silence_level_n=6,
            metadata_json=_metadata_with_window_end(state, window_end),
            event_type="silence_escalated",
            payload={"n": 6, "plan": "thirty_day_final", "remaining": 1, "due_at": next_due.isoformat()},
        )
        return

    await stop_proactive_state(state, reason="silence_exhausted", now=now_ts)
    await log_proactive_event(
        state_id=state.id,
        workspace_id=state.workspace_id,
        user_id=state.user_id,
        agent_id=state.agent_id,
        conversation_id=state.conversation_id,
        event_type="permanently_stopped",
        payload={"reason": "silence_exhausted", "n": next_n},
    )


async def _resume_forced_plan(
    state: ProactiveStateRecord,
    *,
    now_ts: datetime,
    due_at: datetime,
    followup_plan_type: str,
    remaining_forced_triggers: int,
    metadata_json: str,
    event_type: str,
    payload: dict[str, Any],
    silence_level_n: int | None = None,
) -> None:
    try:
        await db.execute_raw(
            """
            UPDATE proactive_states
            SET
                status = 'running',
                silence_level_n = COALESCE($2, silence_level_n),
                followup_plan_type = $3,
                remaining_forced_triggers = $4,
                current_window_index = NULL,
                window_due_at = $5,
                response_deadline_at = NULL,
                last_attempt_at = $6,
                updated_at = CURRENT_TIMESTAMP,
                metadata = $7::jsonb
            WHERE id = $1
            """,
            state.id,
            silence_level_n,
            followup_plan_type,
            remaining_forced_triggers,
            due_at,
            now_ts,
            metadata_json,
        )
    except Exception as e:
        _log_if_unavailable("resume forced plan", e)
        return
    await log_proactive_event(
        state_id=state.id,
        workspace_id=state.workspace_id,
        user_id=state.user_id,
        agent_id=state.agent_id,
        conversation_id=state.conversation_id,
        event_type=event_type,
        payload=payload,
    )


async def advance_to_next_window(
    state: ProactiveStateRecord,
    *,
    now: datetime | None = None,
    event_type: str = "window_advanced",
    payload: dict[str, Any] | None = None,
) -> None:
    now_ts = _now(now)
    current_index = state.current_window_index if state.current_window_index is not None else 0
    next_index = current_index + 1
    if next_index >= len(PROACTIVE_WINDOWS):
        await stop_proactive_state(state, reason="windows_exhausted", now=now_ts)
        return

    due_at = _pick_random_due_at(state.t0_at or now_ts, next_index, now_ts)
    try:
        await db.execute_raw(
            """
            UPDATE proactive_states
            SET
                status = 'running',
                current_window_index = $2,
                window_due_at = $3,
                last_attempt_at = $4,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $1
            """,
            state.id,
            next_index,
            due_at,
            now_ts,
        )
    except Exception as e:
        _log_if_unavailable("advance window", e)
        return
    await log_proactive_event(
        state_id=state.id,
        workspace_id=state.workspace_id,
        user_id=state.user_id,
        agent_id=state.agent_id,
        conversation_id=state.conversation_id,
        event_type=event_type,
        window_index=next_index,
        payload={"due_at": due_at.isoformat(), **(payload or {})},
    )
