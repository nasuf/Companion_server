"""Pure reminder target interpretation, separate from storage and delivery."""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import re

_YEARLESS_DATE = re.compile(r"(?<![\d年])(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]")


def precise_reminder_time(
    message: str, *, now: datetime
) -> tuple[bool, datetime | None]:
    """One day + clock expression bound into a single trigger time.

    The legacy parser returns 明天 and 下午3点 as unrelated results, so a
    reminder for 明天下午3点 fired today at 15:00 and again at midnight, and
    今晚8点 read as 08:00. Returns (True, time) when the message states exactly
    one hour/minute-precise time; time is None if that moment has passed. Returns
    (False, None) otherwise (offsets, several clocks, no clock), leaving the
    legacy parser in charge.
    """
    from app.services.time_domain.parser import normalize_clock_words
    from app.services.time_domain.clock import _TZ
    from app.services.time_domain.query import resolve_temporal_query

    text = normalize_clock_words(message)
    candidates = [text]
    # A reminder looks forward: a yearless date means its next occurrence.
    if len(_YEARLESS_DATE.findall(text)) == 1:
        year = now.astimezone(_TZ).year
        candidates = [
            _YEARLESS_DATE.sub(lambda m, y=y: f"{y}年{m.group()}", text)
            for y in (year, year + 1)
        ]
    for candidate in candidates:
        query = resolve_temporal_query(
            candidate, anchor=now, anchor_source="persisted_message"
        )
        if (
            query.status != "resolved"
            or len(query.expressions) != 1
            or query.expressions[0].precision not in {"hour", "minute"}
        ):
            return False, None
        start = query.expressions[0].start
        if start > now:
            return True, start
    return True, None


@dataclass(frozen=True)
class ReminderTimeResolution:
    has_expression: bool
    occur_time: datetime | None
    reason: str


@dataclass(frozen=True)
class ReminderTargets:
    has_expression: bool
    targets: tuple[datetime, ...]
    reason: str


def resolve_reminder_targets(
    message: str,
    *,
    statement_at: datetime | None,
    statement_end_at: datetime | None,
    now: datetime,
    allow_multiple: bool = False,
) -> ReminderTargets:
    """Interpret at the statement clock, then check validity at execution.

    An aggregate has no per-expression source binding here. Only accept a
    target that is identical at both ends of its verified statement interval.
    Missing evidence never falls back to execution time for a business write.
    """
    from app.services.time_domain.timestamps import parse_time
    from app.services.time_domain.parser import (
        parse_loose_offset,
        parse_with_statement_time,
    )

    anchor, end, execution = map(parse_time, (statement_at, statement_end_at, now))
    if execution is None:
        return ReminderTargets(True, (), "invalid_execution_clock")

    def candidate(at):
        bound, precise = precise_reminder_time(message, now=at)
        parsed = parse_with_statement_time(message, now=at)
        future = [item.start for item in parsed.event_times if item.is_future]
        if bound:
            targets = (precise,) if precise else ()
        elif future:
            targets = tuple(sorted(future)) if allow_multiple else (min(future),)
        else:
            loose = parse_loose_offset(message, at)
            targets = (loose,) if loose is not None else ()
        return bool(bound or parsed.event_times or targets), targets

    try:
        # Execution is used only for syntactic detection when evidence is absent.
        # Such a candidate is rejected below, never used for persistence.
        has_expression, targets = candidate(anchor or execution)
        if not has_expression:
            return ReminderTargets(False, (), "no_expression")
        if anchor is None:
            return ReminderTargets(True, (), "missing_statement_clock")
        if end is None or end < anchor or end > execution or anchor > execution:
            return ReminderTargets(True, (), "invalid_statement_interval")
        if end != anchor and candidate(end) != (has_expression, targets):
            return ReminderTargets(True, (), "ambiguous_statement_interval")
        if not targets:
            return ReminderTargets(True, (), "past_or_unresolved")
        if any(target <= execution for target in targets):
            return ReminderTargets(True, (), "expired_before_processing")
        return ReminderTargets(True, targets, "resolved")
    except (ValueError, OverflowError):
        return ReminderTargets(True, (), "invalid_expression")


def resolve_confirmation_time(
    message: str,
    *,
    statement_at: datetime | None,
    statement_end_at: datetime | None,
    now: datetime,
) -> ReminderTimeResolution:
    """Existing single-target pending contract, backed by the shared resolver."""
    result = resolve_reminder_targets(
        message, statement_at=statement_at, statement_end_at=statement_end_at, now=now
    )
    return ReminderTimeResolution(
        result.has_expression,
        result.targets[0] if result.targets else None,
        result.reason,
    )
