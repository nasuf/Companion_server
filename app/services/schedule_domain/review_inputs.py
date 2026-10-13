"""Bounded daily-review inputs, pinned before generation and rechecked at commit.

Plans and generated utterances are context, not receipts of real-world events.
No conversation text or user-location memories enter the Agent's self review.
"""

from datetime import datetime
import json
from types import SimpleNamespace

from app.services.memory.evidence import content_version
from app.services.memory.manual_write import MemoryWriteError, memory_scope

FORMAT = "daily-review-v1"
LIMITS = {"schedule": 96, "adjustments": 100, "proactive": 5, "games": 200}


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda v: v.isoformat() if isinstance(v, datetime) else str(v),
        allow_nan=False,
    )


async def collect_inputs(
    database, *, user_id, workspace_id, agent_id, start, end, lock=False
):
    if (
        not getattr(start.tzinfo, "key", None)
        or getattr(start.tzinfo, "key", None) != getattr(end.tzinfo, "key", None)
        or any(
            (
                start.hour,
                start.minute,
                start.second,
                start.microsecond,
                end.hour,
                end.minute,
                end.second,
                end.microsecond,
            )
        )
        or (end - start).total_seconds() != 86400
    ):
        raise ValueError("daily_review_invalid_window")
    scope = await memory_scope(
        database, SimpleNamespace(userId=user_id, workspaceId=workspace_id), lock=lock
    )
    if scope["agent_id"] != agent_id:
        raise MemoryWriteError(409, "daily_review_scope_changed")
    suffix = " FOR SHARE" if lock else ""
    takeover = await database.query_raw(
        "SELECT status FROM human_takeovers WHERE user_id=$1" + suffix, user_id
    )
    if any(row["status"] == "active" for row in takeover):
        raise MemoryWriteError(409, "daily_review_human_takeover")
    agents = await database.query_raw(
        "SELECT name,city,occupation,age FROM ai_agents WHERE id=$1 AND user_id=$2"
        + suffix,
        agent_id,
        user_id,
    )
    schedules = await database.query_raw(
        "SELECT id,schedule_data FROM ai_daily_schedules WHERE agent_id=$1 AND date=$2::date"
        + suffix,
        agent_id,
        start.date().isoformat(),
    )
    schedule = schedules[0]["schedule_data"] if schedules else []
    if isinstance(schedule, str):
        schedule = json.loads(schedule)
    if not isinstance(schedule, list) or len(schedule) > LIMITS["schedule"]:
        raise ValueError("daily_review_invalid_schedule")
    if len(canonical(schedule).encode()) > 262144 or any(
        not isinstance(s, dict)
        or not all(isinstance(s.get(k), str) for k in ("start", "end"))
        for s in schedule
    ):
        raise ValueError("daily_review_invalid_schedule")
    adjustments = await database.query_raw(
        """SELECT id,adjust_type,reason,old_value,new_value FROM schedule_adjust_logs
        WHERE agent_id=$1 AND created_at >= $2::timestamptz AT TIME ZONE 'UTC'
        AND created_at < $3::timestamptz AT TIME ZONE 'UTC'
        ORDER BY created_at,id LIMIT 101"""
        + suffix,
        agent_id,
        start.isoformat(),
        end.isoformat(),
    )
    proactive = await database.query_raw(
        """SELECT id,message,event_type FROM proactive_chat_logs
        WHERE agent_id=$1 AND user_id=$2 AND workspace_id=$3
        AND created_at >= $4::timestamptz AT TIME ZONE 'UTC'
        AND created_at < $5::timestamptz AT TIME ZONE 'UTC'
        ORDER BY created_at DESC,id DESC LIMIT 6"""
        + suffix,
        agent_id,
        user_id,
        workspace_id,
        start.isoformat(),
        end.isoformat(),
    )
    games = await database.query_raw(
        """SELECT id,COALESCE(result->>'game_title',game_key) AS title,status,
        result->>'user_outcome' AS outcome,COALESCE(duration_seconds,0) AS dur
        FROM game_sessions WHERE workspace_id=$1 AND user_id=$2 AND agent_id=$3
        AND COALESCE(ended_at,created_at) >= $4::timestamptz AT TIME ZONE 'UTC'
        AND COALESCE(ended_at,created_at) < $5::timestamptz AT TIME ZONE 'UTC'
        ORDER BY COALESCE(ended_at,created_at),id LIMIT 201"""
        + suffix,
        workspace_id,
        user_id,
        agent_id,
        start.isoformat(),
        end.isoformat(),
    )
    if len(adjustments) > LIMITS["adjustments"] or len(games) > LIMITS["games"]:
        raise ValueError("daily_review_input_limit_exceeded")
    value = {
        "format": FORMAT,
        "local_date": start.date().isoformat(),
        "timezone": start.tzinfo.key,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "scope": scope,
        "agent": agents[0],
        "schedule_id": schedules[0]["id"] if schedules else None,
        "schedule": schedule,
        "adjustments": adjustments,
        "proactive": proactive[:5],
        "proactive_truncated": len(proactive) > 5,
        "games": games,
    }
    if len(canonical(value).encode()) > 524288:
        raise ValueError("daily_review_input_too_large")
    return value


def input_version(value):
    return content_version(canonical(value))


def public_summary(value, sampled_at):
    return {
        "local_date": value["local_date"],
        "timezone": value["timezone"],
        "sampled_at": sampled_at,
        "schedule_slots": len(value["schedule"]),
        "adjustments": len(value["adjustments"]),
        "proactive_messages": len(value["proactive"]),
        "proactive_truncated": value["proactive_truncated"],
        "games": len(value["games"]),
    }


def render_games(rows):
    # A created/started session is not a finished match; only settled receipts
    # contribute an outcome, and unfinished sessions are explicitly labelled.
    if not rows:
        return "（昨日无游戏业务记录）"
    settled = [r for r in rows if r["status"] == "settled"]
    titles = list(dict.fromkeys(str(r.get("title") or "游戏") for r in rows))
    titles_text = "、".join(f"《{t[:100]}》" for t in titles[:2])
    if len(titles) > 2:
        titles_text += f"等{len(titles)}种"
    minutes = sum(max(0, int(r.get("dur") or 0)) for r in rows) // 60
    return (
        f"昨日游戏业务记录涉及{titles_text}，共{len(rows)}局，记录时长{minutes}分钟，其中已结算{len(settled)}局、"
        f"未结算{len(rows)-len(settled)}局；已结算结果："
        f"用户赢{sum(r['outcome']=='win' for r in settled)}局、"
        f"AI赢{sum(r['outcome']=='lose' for r in settled)}局、"
        f"平局{sum(r['outcome']=='draw' for r in settled)}局。"
        "这些是系统对局记录，不是现实生活事件。"
    )
