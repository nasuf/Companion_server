"""Bounded, invocation-owned conversation clocks; never infer event validity.

Database timestamps without an offset are UTC. Proactive output is not proof
that a user returned. Missing or future clocks stay unknown in diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.config import settings
from app.services.temporal import parse_time, local_clock
from app.services.temporal_query import TemporalQuery, resolve_temporal_query

logger = logging.getLogger(__name__)

HISTORY_LIMIT = 120
SESSION_SECONDS = 3 * 3600


@dataclass(frozen=True)
class StatementEvidence:
    """Counts refer to unique server-owned turn IDs, never client timestamps."""
    expected: int
    verified: int = 0
    missing: int = 0
    invalid: int = 0
    conflicting: int = 0
    reason: str = "unknown"
    earliest: datetime | None = None
    latest: datetime | None = None

    def diagnostics(self, now: datetime) -> dict:
        return {
            "version": 1, "reason": self.reason, "expected_messages": self.expected,
            "verified_messages": self.verified, "missing_messages": self.missing,
            "invalid_messages": self.invalid, "conflicting_messages": self.conflicting,
            "earliest_at": self.earliest.isoformat() if self.earliest else None,
            "latest_at": self.latest.isoformat() if self.latest else None,
            "processing_delay_seconds": round((now - self.earliest).total_seconds(), 3)
                if self.earliest else None,
        }


def _statement_evidence(history: list[dict], excluded: set[str], reference: datetime,
                        received_at: object) -> StatementEvidence:
    expected = len(excluded)
    if expected > HISTORY_LIMIT:
        return StatementEvidence(expected, missing=expected, reason="message_count_limit")
    groups = {ident: [] for ident in excluded}
    for message in history:
        if message.get("id") in groups and not message.get("synthetic_current"):
            stamp = parse_time(message.get("createdAt"))
            valid = message.get("role") == "user" and stamp is not None and stamp <= reference
            groups[message["id"]].append(stamp if valid else None)
    verified, missing, invalid, conflicting = [], 0, 0, 0
    for stamps in groups.values():
        if not stamps:
            missing += 1
        elif len(set(stamps)) > 1:
            conflicting += 1
        elif stamps[0] is None:
            invalid += 1
        else:
            verified.append(stamps[0])
    counts = dict(expected=expected, verified=len(verified), missing=missing,
                  invalid=invalid, conflicting=conflicting)
    # A receipt can anchor one unsaved/synthetic message, never stand in for
    # missing fragments or override an invalid/conflicting persisted timestamp.
    received = parse_time(received_at)
    if expected <= 1 and not verified and not invalid and not conflicting:
        if received and received <= reference:
            return StatementEvidence(**counts, reason="server_received", earliest=received, latest=received)
    if conflicting or invalid or missing or not verified:
        reason = ("conflicting_message_times" if conflicting else "invalid_message_times" if invalid
                  else "incomplete_message_times" if expected else "unknown")
        return StatementEvidence(**counts, reason=reason)
    earliest, latest = min(verified), max(verified)
    try:
        zone = ZoneInfo(settings.schedule_timezone)
    except (ValueError, ZoneInfoNotFoundError):
        return StatementEvidence(**counts, reason="invalid_timezone")
    reason = ("multiple_message_dates" if earliest.astimezone(zone).date() != latest.astimezone(zone).date()
              else "persisted_message")
    return StatementEvidence(**counts, reason=reason, earliest=earliest, latest=latest)


def is_proactive(message: dict) -> bool:
    meta = message.get("metadata")
    return isinstance(meta, dict) and bool(meta.get("proactive") or meta.get("trigger_type"))


@dataclass(frozen=True)
class TurnClock:
    now: datetime
    last_user_at: datetime | None
    last_dialogue_at: datetime | None
    last_outbound_at: datetime | None
    inspected_messages: int
    unknown_timestamps: int
    statement_at: datetime | None = None
    anchor_source: str = "unknown"
    statement_end_at: datetime | None = None
    statement_evidence: StatementEvidence | None = None

    def gap(self, value: datetime | None) -> float | None:
        return round((self.now - value).total_seconds(), 3) if value else None

    def diagnostics(self) -> dict:
        return {
            "version": 1,
            "reference_at": self.now.isoformat(),
            "timezone": settings.schedule_timezone,
            "last_user_at": self.last_user_at.isoformat() if self.last_user_at else None,
            "last_dialogue_at": self.last_dialogue_at.isoformat() if self.last_dialogue_at else None,
            "last_outbound_at": self.last_outbound_at.isoformat() if self.last_outbound_at else None,
            "user_gap_seconds": self.gap(self.last_user_at),
            "dialogue_gap_seconds": self.gap(self.last_dialogue_at),
            "inspected_messages": self.inspected_messages,
            "unknown_timestamps": self.unknown_timestamps,
            "statement_evidence": self.statement_evidence.diagnostics(self.now) if self.statement_evidence else None,
        }


def build_turn_clock(messages: list[dict], *, exclude_ids: set[str] | None = None,
                     now: datetime | None = None, received_at: object = None) -> TurnClock:
    reference = parse_time(now) or datetime.now(timezone.utc)
    excluded = exclude_ids or set()
    users, dialogue, outbound = [], [], []
    unknown = 0
    history = messages[-HISTORY_LIMIT:]
    for msg in history:
        if msg.get("id") in excluded or msg.get("synthetic_current"):
            continue
        ts = parse_time(msg.get("createdAt"))
        if ts is None or ts > reference:
            unknown += 1
            continue
        if msg.get("role") == "user":
            users.append(ts)
            dialogue.append(ts)
        elif msg.get("role") == "assistant":
            outbound.append(ts)
            if not is_proactive(msg):
                dialogue.append(ts)
    evidence = _statement_evidence(history, excluded, reference, received_at)
    anchor = evidence.earliest if evidence.reason in {"persisted_message", "server_received"} else None
    return TurnClock(reference, max(users, default=None), max(dialogue, default=None),
                     max(outbound, default=None), len(history), unknown, anchor, evidence.reason,
                     evidence.latest if anchor else None, evidence)


def dated_dialogue(messages: list[dict], *, exclude_ids: set[str] | None = None,
                   max_chars: int = 1800) -> str:
    """Bounded raw evidence with absolute dates, retaining whole latest lines."""
    excluded = exclude_ids or set()
    lines, used = [], 0
    for msg in reversed(messages[-HISTORY_LIMIT:]):
        if msg.get("id") in excluded or msg.get("synthetic_current"):
            continue
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            continue
        text = str(msg.get("content") or "").strip()
        if not text:
            continue
        speaker = "用户" if role == "user" else "AI主动消息" if is_proactive(msg) else "AI"
        line = f"[{local_clock(msg.get('createdAt'))}] {speaker}: {text[:360]}"
        if used + len(line) + 1 > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(reversed(lines)) or "(无可用历史)"


def elapsed_text(seconds: float) -> str:
    """A duration to the minute ("1 小时 30 分钟"). format_gap_text rounds down
    to a single unit for greetings; an answer to "how long" needs both."""
    total = int(seconds)
    if total < 60:
        return f"{total} 秒"
    hours, minutes = divmod(total // 60, 60)
    if not hours:
        return f"{minutes} 分钟"
    days, hours = divmod(hours, 24)
    if not days:
        return f"{hours} 小时" + (f" {minutes} 分钟" if minutes else "")
    return f"{days} 天" + (f" {hours} 小时" if hours else "")


async def render_temporal_context(clock: TurnClock, policy: str,
                                  temporal_query: TemporalQuery | None = None) -> str:
    import json
    from app.services.prompting.store import get_prompt_text, PromptDisabledError
    from app.services.prompting.utils import render_template

    try:
        template = await get_prompt_text("chat.temporal_context")
    except PromptDisabledError:
        return ""
    query_context = None
    if temporal_query is not None:
        # Execution metadata is for routing/Trace, not evidence about whether
        # an answer exists in dialogue or memory. In particular, an unexecuted
        # count must not override an explicitly stated total. Keep date ranges
        # and ambiguity here; diagnostics below retain the complete operation.
        query_context = {key: value for key, value in temporal_query.prompt_context().items()
                         if key in {"status", "timezone", "ranges", "reason"}}
    last_dialogue = local_clock(clock.last_dialogue_at)
    if clock.last_dialogue_at is not None:
        # The model should not divide seconds itself: it copied the prompt's
        # worked example instead ("三个小时左右" for a 1.5-hour gap).
        last_dialogue += f"（距今约 {elapsed_text(clock.gap(clock.last_dialogue_at))}）"
    return render_template(template, {
        "now": local_clock(clock.now),
        "statement_at": local_clock(clock.statement_at),
        "query_context": json.dumps(query_context, ensure_ascii=False) if query_context is not None else "未请求日期解析",
        "last_user": local_clock(clock.last_user_at),
        "last_dialogue": last_dialogue,
        "last_outbound": local_clock(clock.last_outbound_at),
        "user_gap_seconds": clock.gap(clock.last_user_at) if clock.last_user_at else "未知",
        "dialogue_gap_seconds": clock.gap(clock.last_dialogue_at) if clock.last_dialogue_at else "未知",
        "followup_policy": policy,
    }) if template else ""


async def prepare_reply_time(messages: list[dict], *, exclude_ids: set[str],
                             current_message: str, diagnostics: dict,
                             now: datetime | None = None, clock: TurnClock | None = None,
                             temporal_query: TemporalQuery | None = None):
    """One clock/policy snapshot shared by graph and legacy execution paths."""
    from app.services.chat.followup_policy import derive_followup_policy

    clock = clock or build_turn_clock(messages, exclude_ids=exclude_ids, now=now)
    query = temporal_query or resolve_temporal_query(current_message, anchor=clock.statement_at,
                                   anchor_source=clock.anchor_source, anchor_end=clock.statement_end_at)
    policy = derive_followup_policy(messages, now=clock.now, current_message=current_message)
    try:
        instructions = await render_temporal_context(clock, policy.prompt_value(), query)
        prompt_status = "enabled" if instructions else "disabled"
    except Exception as exc:
        # An optional prompt read must not prevent a crisis or ordinary reply.
        logger.warning("[TEMPORAL] prompt unavailable: %s", type(exc).__name__)
        instructions, prompt_status = "", "unavailable"
    diagnostics["temporal_context"] = {**clock.diagnostics(), **policy.diagnostics(),
                                       "prompt_status": prompt_status,
                                       "query": query.diagnostics(),
                                       "suppressed_question_count": 0, "out_of_window_event_count": 0, "event_check_status": "not_requested"}
    return clock, policy, instructions


def scope_event_memories(memories, query: str, *, now: datetime, diagnostics: dict,
                         temporal_query: TemporalQuery | None = None):
    """Project point-event evidence into an explicit query date, without writes.

    Stable facts, unknown dates, multi-period comparisons and duration statements
    remain available. A record/access timestamp never supplies an event date.
    """
    import re
    from zoneinfo import ZoneInfo

    diagnostic = diagnostics.setdefault("temporal_context", {})
    diagnostic["out_of_window_event_count"] = 0
    if not memories or not re.search(r"[?？]|什么|哪|吗|有没有|记得|记不记得", query):
        return memories
    if re.search(r"相比|比较|以前|过去|上次|一直|长期|每天|持续|以来", query):
        return memories
    resolved = temporal_query or resolve_temporal_query(query, anchor=now, anchor_source="server_received")
    periods = resolved.parsed_times()
    if not periods:
        return memories
    zone = ZoneInfo(resolved.timezone)
    # A single explicit calendar day is unambiguous. Broader/overlapping intent
    # stays in the normal dated-evidence path rather than guessing its scope.
    dates = {p.start.astimezone(zone).date() for p in periods
             if p.start.astimezone(zone).date() == p.end.astimezone(zone).date()}
    if len(dates) != 1 or len(periods) != len([p for p in periods
            if p.start.astimezone(zone).date() == p.end.astimezone(zone).date()]):
        return memories
    target = next(iter(dates))
    diagnostic["requested_event_date"] = target.isoformat()
    kept = []
    for memory in memories:
        event = parse_time(memory.occur_time)
        duration = re.search(r"一直|长期|每天|持续|期间|以来|从.+到", memory.text)
        if (memory.main_category in {"生活", "情绪"} and event is not None
                and not duration and event.astimezone(zone).date() != target):
            diagnostic["out_of_window_event_count"] += 1
        else:
            kept.append(memory)
    diagnostic["in_window_event_count"] = sum(
        m.main_category in {"生活", "情绪"} and parse_time(m.occur_time) is not None
        and parse_time(m.occur_time).astimezone(zone).date() == target for m in kept)
    return kept


async def render_event_scope(diagnostics: dict) -> str:
    from app.services.prompting.store import get_prompt_text, PromptDisabledError
    from app.services.prompting.utils import render_template

    evidence = diagnostics.get("temporal_context", {})
    if not evidence.get("out_of_window_event_count"):
        return ""
    try:
        template = await get_prompt_text("chat.temporal_evidence")
        evidence["event_prompt_status"] = "enabled"
        return render_template(template, {
            "date": evidence["requested_event_date"],
            "matched": evidence.get("in_window_event_count", 0),
            "excluded": evidence["out_of_window_event_count"],
        })
    except PromptDisabledError:
        evidence["event_prompt_status"] = "disabled"
        return ""
    except Exception as exc:
        logger.warning("[TEMPORAL] evidence prompt unavailable: %s", type(exc).__name__)
        evidence["event_prompt_status"] = "unavailable"
        return ""


def dated_active_dialogue(messages: list[dict], *, now: datetime, exclude_ids: set[str]) -> str:
    recent = [m for m in messages[-HISTORY_LIMIT:]
              if (ts := parse_time(m.get("createdAt"))) is not None
              and 0 <= (now - ts).total_seconds() <= SESSION_SECONDS]
    return dated_dialogue(recent, exclude_ids=exclude_ids)
