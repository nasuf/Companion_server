"""Read-only temporal query contract. No event dates are inferred from storage time.

The caller supplies a server-owned statement anchor. Expressions never use a
new wall clock during retrieval, and ambiguous ranges never narrow recall.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.config import settings
from app.services.temporal import parse_time
from app.services.temporal_operation import TemporalOperation, classify_temporal_operation

MAX_EXPRESSIONS = 8
MAX_QUERY_CHARS = 4000
_VAGUE = re.compile(r"前阵子|前些日子|前几天|过几天|过阵子|最近|以前|小时候|当时|月底|年初|年底")
_YEARLESS = re.compile(r"(?<!\d)\d{1,2}\s*月\s*\d{1,2}\s*[日号]")
_DATED = re.compile(r"(\d{4})\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})\s*[日号]?")
_APPROXIMATE = re.compile(r"[一二三四五六七八九十百两半\d]{1,4}\s*个?\s*(?:月|年)(?:之)?[前后]")
_ALIASES = {"昨晚": "昨天晚上", "昨夜": "昨天晚上", "今早": "今天早上",
            "今晨": "今天早晨", "今晚": "今天晚上", "明早": "明天早上", "明晚": "明天晚上"}
_ALIAS_PATTERN = re.compile("|".join(_ALIASES))
_IMPRECISE = re.compile(r"刚刚|刚才|不久前|前不久|一会儿|待会|稍后")
_JOIN = re.compile(r"\s*(?:到|至|—|~|～)\s*")
_DISTINCT = re.compile(r"和|以及|、|或|相比|还是|分别")
_HOLIDAY = re.compile(r"春节|端午|中秋|国庆|除夕|元旦|圣诞|清明|重阳|元宵")


def _unique_local_clock(value: datetime) -> bool:
    """A skipped or repeated DST clock cannot identify one exact instant."""
    instants = set()
    for fold in (0, 1):
        candidate = value.replace(fold=fold)
        utc = candidate.astimezone(timezone.utc)
        if utc.astimezone(value.tzinfo).replace(tzinfo=None) == value.replace(tzinfo=None):
            instants.add(utc)
    return len(instants) == 1


@dataclass(frozen=True)
class TemporalExpression:
    start: datetime
    end: datetime
    precision: str
    kind: str


@dataclass(frozen=True)
class TemporalQuery:
    anchor: datetime | None
    anchor_source: str
    timezone: str
    timezone_source: str
    status: str
    reason: str
    expressions: tuple[TemporalExpression, ...] = ()
    kind: str = "unspecified"
    operation: TemporalOperation = TemporalOperation()

    def diagnostics(self) -> dict:
        return {
            "version": 1, "anchor_at": self.anchor.isoformat() if self.anchor else None,
            "anchor_source": self.anchor_source, "timezone": self.timezone,
            "timezone_source": self.timezone_source, "status": self.status,
            "reason": self.reason, "kind": self.kind,
            "operation": self.operation.diagnostics(),
            "ranges": [{"start": p.start.isoformat(), "end": p.end.isoformat(),
                        "precision": p.precision, "kind": p.kind} for p in self.expressions],
        }

    def cache_token(self) -> str:
        # Do not key ordinary messages by a changing clock. Resolved ranges
        # include their date/zone; unknown anchors cannot reuse a dated result.
        value = self.diagnostics()
        value.pop("anchor_at")
        return "time02:" + hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]

    def retrieval_range(self) -> tuple[datetime, datetime] | None:
        if (self.status != "resolved" or len(self.expressions) != 1
                or self.operation.type in {"as_of", "current", "duration", "comparison", "multiple"}):
            return None
        p = self.expressions[0]
        return (p.start, p.end) if self.anchor and p.start <= self.anchor else None

    def parsed_times(self) -> list:
        from app.services.schedule_domain.time_parser import ParsedTime
        if (self.status != "resolved"
                or self.operation.type in {"as_of", "current", "duration", "comparison", "multiple"}):
            return []
        return [ParsedTime("", p.kind, p.start, p.end, 1.0,
                           bool(self.anchor and p.start > self.anchor)) for p in self.expressions]


def resolve_temporal_query(text: str, *, anchor: object, anchor_source: str,
                           timezone_name: str | None = None) -> TemporalQuery:
    from app.services.schedule_domain.time_parser import parse_time_expressions

    stamp = parse_time(anchor)
    zone_name = timezone_name or settings.schedule_timezone
    zone_source = "explicit_setting" if timezone_name else "configured_default"
    base = dict(anchor=stamp, anchor_source=anchor_source, timezone=zone_name,
                timezone_source=zone_source)
    if len(text) > MAX_QUERY_CHARS:
        return TemporalQuery(**base, status="unresolved", reason="input_limit")
    base["operation"] = classify_temporal_operation(text)
    try:
        zone = ZoneInfo(zone_name)
    except (ValueError, ZoneInfoNotFoundError):
        return TemporalQuery(**base, status="unresolved", reason="invalid_timezone")
    if not text.strip():
        return TemporalQuery(**base, status="not_requested", reason="no_expression")
    if stamp is None:
        return TemporalQuery(**base, status="unresolved", reason="missing_anchor")
    # A vague expression or an omitted year must not acquire a precise date
    # simply because a parser happens to recognize another substring.
    if _VAGUE.search(text) or _IMPRECISE.search(text):
        return TemporalQuery(**base, status="ambiguous", reason="vague_expression")
    text = _ALIAS_PATTERN.sub(lambda m: _ALIASES[m.group()], text)
    # Holiday cache may choose the nearest year or even shift a lunar holiday
    # by a Gregorian year. A read query needs explicit, qualified calendar
    # evidence before such a date can restrict recall.
    from app.services.schedule_domain import holiday_cache
    if _HOLIDAY.search(text) or any(name in text for name in holiday_cache.all_known_names()):
        return TemporalQuery(**base, status="ambiguous", reason="holiday_calendar_required")
    # Relative years are explicit when bound to a calendar date. Resolve them
    # before the legacy parser's overlapping year/month patterns can discard it.
    year = stamp.astimezone(zone).year
    text = re.sub(r"(今年|去年|前年)\s*(?=\d{1,2}\s*月\s*\d{1,2}\s*[日号])",
                  lambda m: str(year - {"今年": 0, "去年": 1, "前年": 2}[m.group(1)]) + "年", text)
    if _YEARLESS.search(_DATED.sub("", text)):
        return TemporalQuery(**base, status="ambiguous", reason="year_unspecified")
    if _APPROXIMATE.search(text):
        return TemporalQuery(**base, status="ambiguous", reason="approximate_duration")
    try:
        for match in _DATED.finditer(text):
            date(*(int(v) for v in match.groups()))
        parsed = parse_time_expressions(text, now=stamp, calendar_timezone=zone)
    except (ValueError, OverflowError):
        return TemporalQuery(**base, status="unresolved", reason="invalid_expression")
    if len(parsed) > MAX_EXPRESSIONS:
        return TemporalQuery(**base, status="unresolved", reason="expression_limit")
    if not parsed:
        return TemporalQuery(**base, status="not_requested", reason="no_expression")
    # "去年和上个月" are two requested periods, even if they overlap. Only
    # directly adjacent day/time fragments or explicit "A到B" may be combined.
    ordered = sorted(parsed, key=lambda p: text.find(p.original_text))
    if any(_DISTINCT.search(text[text.find(a.original_text) + len(a.original_text):
                                      text.find(b.original_text)])
           for a, b in zip(ordered, ordered[1:])):
        return TemporalQuery(**base, status="ambiguous", reason="multiple_periods")
    # Bind only adjacent calendar-day + time-of-day expressions. "昨天中午"
    # is one period; "昨天去了，今天中午又去" is not. Never combine across prose.
    bound_range = None
    if len(parsed) == 2:
        left, right = ordered
        left_end = text.find(left.original_text) + len(left.original_text)
        right_start = text.find(right.original_text)
        if _JOIN.fullmatch(text[left_end:right_start]):
            if left.start > right.end:
                return TemporalQuery(**base, status="unresolved", reason="reversed_range")
            bound_range = left.start, right.end
        days = [p for p in parsed if p.start.hour == 0 and p.start.minute == 0
                and p.end.hour == 23 and p.end.minute == 59 and p.start.date() == p.end.date()]
        if len(days) == 1:
            day = days[0]
            part = next(p for p in parsed if p is not day)
            adjacent = re.search(re.escape(day.original_text) + r"\s*(?:的)?\s*" + re.escape(part.original_text), text)
            if not bound_range and adjacent and re.search(r"点|时|早|上午|中午|下午|晚上|凌晨|深夜", part.original_text):
                delta = day.start.date() - stamp.astimezone(zone).date()
                bound_range = part.start + delta, part.end + delta
    # Use the intersection only when all fragments describe a compatible
    # period (e.g. 今天中午). Distinct periods require semantic binding later.
    start, end = bound_range or (max(p.start for p in parsed), min(p.end for p in parsed))
    if start > end:
        return TemporalQuery(**base, status="ambiguous", reason="multiple_periods")
    # Elapsed offsets have an unambiguous UTC anchor even during a repeated
    # local hour. Calendar wall times instead need explicit disambiguation.
    elapsed = len(parsed) == 1 and re.search(r"(?:分钟|小时)(?:之)?[前后]", parsed[0].original_text)
    if not elapsed and not all(_unique_local_clock(t) for t in (start, end)):
        return TemporalQuery(**base, status="ambiguous", reason="ambiguous_clock")
    local_start, local_end = start.astimezone(zone), end.astimezone(zone)
    precision = "day" if local_start.date() == local_end.date() and local_start.hour == 0 and local_end.hour == 23 else "range"
    expression = TemporalExpression(start, end, precision, parsed[0].type)
    if base["operation"].type == "unspecified":
        base["operation"] = TemporalOperation("planned" if start > stamp else "range")
    return TemporalQuery(**base, status="resolved", reason="explicit_range",
                         expressions=(expression,), kind="planned" if start > stamp else "range")
