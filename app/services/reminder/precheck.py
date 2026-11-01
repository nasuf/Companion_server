"""Bounded user evidence for reminder state changes, never delivery evidence.

A classifier can propose a state; only the pinned user messages and the current
trigger version authorize the transaction. Assistant guesses are not user acts.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import re

from prisma import Json

from app.services.memory.evidence import (
    EvidenceSource,
    content_version,
    message_statement_time,
)
from app.services.memory.manual_write import MemoryWriteError
from app.services.reminder.mutations import mutate_reminder
from zoneinfo import ZoneInfo
from app.config import settings
from app.services.time_domain.reminders import resolve_reminder_targets


@dataclass(frozen=True)
class PrecheckContext:
    text: str = ""
    sources: tuple[EvidenceSource, ...] = ()
    messages: tuple[str, ...] = ()
    truncated: bool = False


async def load_context(
    database, *, agent_id: str, user_id: str, now: datetime
) -> PrecheckContext:
    rows = await database.query_raw(
        """SELECT m.id,m.content,m.created_at,m.conversation_id FROM messages m
        JOIN conversations c ON c.id=m.conversation_id
        JOIN chat_workspaces w ON w.id=c.workspace_id AND w.agent_id=c.agent_id AND w.user_id=c.user_id
        WHERE c.agent_id=$1 AND c.user_id=$2 AND NOT c.is_deleted AND c.archived_at IS NULL
          AND w.status='active' AND w.archived_at IS NULL AND m.role='user'
          AND m.created_at >= ($3::timestamptz AT TIME ZONE 'UTC')
          AND m.created_at <= ($4::timestamptz AT TIME ZONE 'UTC')
          AND NOT COALESCE(m.metadata,'{}'::jsonb) ? 'component_card'
          AND COALESCE(m.metadata->>'human_takeover','false') <> 'true'
          AND COALESCE(m.metadata->>'memory_confirmation','false') <> 'true'
        ORDER BY m.created_at DESC,m.id DESC LIMIT 41""",
        agent_id,
        user_id,
        (now - timedelta(minutes=30)).isoformat(),
        now.isoformat(),
    )
    # Do not decide state from an incomplete window or a clipped negation.
    if len(rows) > 40 or sum(len(r["content"] or "") for r in rows) > 4800:
        return PrecheckContext(truncated=True)
    rows = list(reversed(rows))
    sources = tuple(
        EvidenceSource(
            "message",
            r["id"],
            expected_role="user",
            expected_conversation_id=r["conversation_id"],
            expected_version=content_version(r["content"]),
            expected_statement_time=message_statement_time(r["created_at"]),
        )
        for r in rows
        if r["content"].strip()
    )
    texts = tuple(r["content"] for r in rows if r["content"].strip())
    return PrecheckContext("\n".join("用户: " + t for t in texts), sources, texts)


def reschedule_target(context: PrecheckContext, now: datetime) -> datetime:
    # Generated ISO values cannot supply a date absent from the user's statement.
    candidates = []
    for text, source in zip(context.messages, context.sources):
        markers = list(re.finditer(r"改到|改成|改为|挪到|推迟到|提前到|调整到", text))
        if not markers:
            continue
        if len(markers) != 1 or re.search(
            r'[“”"「」『』?？]|还是|或者|或是|如果|假如|也许|可能|(?:他|她|别人|同事|朋友).{0,6}说|不要改|不用改|别改|不改',
            text,
        ):
            raise MemoryWriteError(422, "reminder_precheck_ambiguous_time")
        result = resolve_reminder_targets(
            text[markers[0].end() :],
            statement_at=source.expected_statement_time,
            statement_end_at=source.expected_statement_time,
            now=now,
            allow_multiple=True,
        )
        if result.reason != "resolved" or len(result.targets) != 1:
            raise MemoryWriteError(422, "reminder_precheck_unresolved_time")
        candidates.extend(result.targets)
    if len(candidates) != 1:
        raise MemoryWriteError(422, "reminder_precheck_ambiguous_time")
    return candidates[0]


async def apply_state(
    database, *, trigger, state: str, context: PrecheckContext, now: datetime
):
    if state not in {"completed", "cancelled", "rescheduled"}:
        raise MemoryWriteError(422, "reminder_precheck_invalid_state")
    if (
        not isinstance(context, PrecheckContext)
        or context.truncated
        or not context.sources
    ):
        raise MemoryWriteError(409, "reminder_precheck_source_unavailable")
    data = dict(trigger.actionData or {})
    data["precheck_state"] = state
    data["precheck_observed_at"] = now.isoformat()
    memory_changes = {}
    changes = {"actionData": Json(data)}
    if state == "rescheduled":
        target = reschedule_target(context, now)
        changes["triggerTime"] = target
        memory_changes["occurTime"] = target
    elif state == "completed" and (data.get("recurrence") or "once") != "once":
        from app.services.proactive.triggers import (
            _next_occurrence,
            _next_habit_weekday_occurrence,
            _normalize_habit_weekdays,
        )

        current = message_statement_time(trigger.triggerTime).astimezone(
            ZoneInfo(settings.schedule_timezone)
        )
        date_key = current.date().isoformat()
        dates = sorted(
            set(str(x) for x in data.get("completed_dates", []) if isinstance(x, str))
            | {date_key}
        )
        data["completed_dates"] = dates[-366:]
        days = _normalize_habit_weekdays(data.get("habit_weekdays"))
        target = (
            _next_habit_weekday_occurrence(current, days)
            if data["recurrence"] == "weekly" and days
            else _next_occurrence(current, data["recurrence"])
        )
        if target is None or target <= now:
            raise MemoryWriteError(409, "reminder_precheck_occurrence_stale")
        changes.update(triggerTime=target, actionData=Json(data))
        memory_changes["occurTime"] = target
    else:
        data["completed_at" if state == "completed" else "cancelled_at"] = (
            now.isoformat()
        )
        changes.update(isActive=False, actionData=Json(data))
        memory_changes["isArchived"] = True
    return await mutate_reminder(
        database,
        trigger=trigger,
        trigger_changes=changes,
        memory_changes=memory_changes,
        operation="reminder_" + state,
        sources=context.sources,
    )
