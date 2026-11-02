"""Read-only temporal query contract. No event dates are inferred from storage time.

The caller supplies a server-owned statement anchor. Expressions never use a
new wall clock during retrieval, and ambiguous ranges never narrow recall.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.config import settings
from app.services.time_domain.timestamps import EndBoundary, exclusive_end, parse_time
from app.services.time_domain.operations import (
    TemporalOperation,
    classify_temporal_operation,
    is_elliptical_followup,
    is_now_followup,
    asks_own_plan,
    is_state_question,
)
from app.services.time_domain.binding import (
    SourceText, binding_diagnostics, binding_uncertainty, locate_parts,
    RANGE_CONNECTOR, is_range_connector,
)

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
_CLOCK_MINUTE = re.compile(r"[点时]\d{1,2}分?$")
_CLOCK_HOUR = re.compile(r"\d{1,2}[点时]$")
_ELAPSED = re.compile(r"(?:[一二三四五六七八九十百两半\d]{1,4}\s*个?\s*(?:分钟|小时)|半天)(?:之)?[前后]")
_HOLIDAY = re.compile(r"春节|端午|中秋|国庆|除夕|元旦|圣诞|清明|重阳|元宵")
# These references may date a statement, quotation, or stored record instead of
# its subject. Without semantic binding, retaining recall is safer than using
# that date to exclude events. This is NOT a statement-date search.
_REFERENCE_CLOCK = re.compile(
    r"提过|提起|提到|提及|说过|说的|说起|聊过|聊的|聊起|聊天|告诉|"
    r"讲过|讲的|讲起|分享过|发过|发的|问过|问的|记下|记录过|记录的|"
    r"转述|转发|原帖|日记|录音|日志|入库|导入|保存|读取|打开|重试|补传|"
    r"提及时|陈述时|接收时间"
)


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
    end_boundary: EndBoundary = "exclusive"


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
    date_basis: str = "event_candidate"
    binding: dict | None = None

    def allows_event_range(self) -> bool:
        return (
            self.date_basis == "event_candidate"
            and self.status == "resolved"
            and len(self.expressions) == 1
            and self.operation.type
            not in {"as_of", "current", "duration", "comparison", "multiple"}
        )

    def prompt_context(self) -> dict:
        """Existing prompt contract; operational filter policy stays in Trace."""
        return {
            "version": 1,
            "anchor_at": self.anchor.isoformat() if self.anchor else None,
            "anchor_source": self.anchor_source,
            "timezone": self.timezone,
            "timezone_source": self.timezone_source,
            "status": self.status,
            "reason": self.reason,
            "kind": self.kind,
            "operation": self.operation.diagnostics(),
            "ranges": [
                {
                    "start": p.start.isoformat(),
                    "end": p.end.isoformat(),
                    "precision": p.precision,
                    "kind": p.kind,
                    "end_boundary": p.end_boundary,
                }
                for p in self.expressions
            ],
        }

    def diagnostics(self) -> dict:
        window = self.retrieval_range()
        return {
            **self.prompt_context(),
            "retrieval_window": {
                "version": 1,
                "status": "eligible" if window else "not_eligible",
                "bounds": "[start,end)",
                "start": window[0].isoformat() if window else None,
                "end_exclusive": window[1].isoformat() if window else None,
            },
            "date_basis": {
                "version": 1,
                "kind": self.date_basis,
                "event_range_eligible": self.allows_event_range(),
            },
            "binding": self.binding
            or {
                "version": 1,
                "decision": "not_analyzed",
                "offset_unit": "unicode_codepoint",
                "spans": [],
            },
        }

    def cache_token(self) -> str:
        # Do not key ordinary messages by a changing clock. Resolved ranges
        # include their date/zone; unknown anchors cannot reuse a dated result.
        value = self.diagnostics()
        value.pop("anchor_at")
        return (
            "time02f:"
            + hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[
                :24
            ]
        )

    def retrieval_range(self) -> tuple[datetime, datetime] | None:
        if not self.allows_event_range():
            return None
        p = self.expressions[0]
        return (
            (parse_time(p.start), exclusive_end(p.end, p.end_boundary))
            if self.anchor and p.start <= self.anchor
            else None
        )

    def parsed_times(self) -> list:
        from app.services.time_domain.parser import ParsedTime

        if not self.allows_event_range():
            return []
        return [
            ParsedTime(
                "",
                p.kind,
                p.start,
                p.end,
                1.0,
                bool(self.anchor and p.start > self.anchor),
                p.end_boundary,
            )
            for p in self.expressions
        ]


def _past_state_question(text, context_text, parsed, end, stamp, zone) -> bool:
    """整段时间都在今天之前, 且问的是用户自己当时的状态 (或省略追问上一句这样问)。"""
    today_start = stamp.astimezone(zone).replace(hour=0, minute=0, second=0, microsecond=0)
    if end > today_start:
        return False
    return _state_question_or_followup(text, context_text, parsed)


def _state_question_or_followup(text, context_text, parsed) -> bool:
    if is_state_question(text):
        return True
    spans = [(i, i + len(p.original_text)) for p in parsed if (i := text.find(p.original_text)) >= 0]
    return bool(context_text) and is_elliptical_followup(text, spans) and is_state_question(context_text)


def resolve_temporal_query(
    text: str,
    *,
    anchor: object,
    anchor_source: str,
    timezone_name: str | None = None,
    anchor_end: object = None,
    context_text: str | None = None,
    bare_months: bool = False,
) -> TemporalQuery:
    """context_text: 省略式追问 ("那上个月呢") 的上一句用户消息, 只用来判断问的是
    不是状态; 时间仍只从 text 里解析。

    bare_months: 认 "九月份" 这类不带年份的月份。只给聊天读路径用 —— 提醒、作息、
    录入等写路径照旧不认, 免得 "提醒我12月续费" 静默建出一条午夜提醒。"""
    from app.services.time_domain.parser import (
        TimeExpressionLimitExceeded,
        parse_time_expressions,
    )

    stamp = parse_time(anchor)
    zone_name = timezone_name or settings.schedule_timezone
    zone_source = "explicit_setting" if timezone_name else "configured_default"
    base = dict(
        anchor=stamp,
        anchor_source=anchor_source,
        timezone=zone_name,
        timezone_source=zone_source,
    )
    if len(text) > MAX_QUERY_CHARS:
        return TemporalQuery(
            **base, status="unresolved", reason="input_limit", date_basis="unavailable"
        )
    base["date_basis"] = (
        "unbound_reference" if _REFERENCE_CLOCK.search(text) else "event_candidate"
    )
    base["operation"] = classify_temporal_operation(text)
    if (base["operation"].type == "unspecified" and context_text
            and is_now_followup(text) and is_state_question(context_text)):
        base["operation"] = TemporalOperation("current", inferred=True)
    try:
        zone = ZoneInfo(zone_name)
    except (ValueError, ZoneInfoNotFoundError):
        return TemporalQuery(**base, status="unresolved", reason="invalid_timezone")
    if not text.strip():
        return TemporalQuery(**base, status="not_requested", reason="no_expression")
    if stamp is None:
        return TemporalQuery(**base, status="unresolved", reason="missing_anchor")
    if anchor_end is not None:
        end = parse_time(anchor_end)
        if end is None or end < stamp:
            return TemporalQuery(
                **base, status="unresolved", reason="invalid_anchor_window"
            )
        if end.astimezone(zone).date() != stamp.astimezone(zone).date():
            return TemporalQuery(
                **base, status="ambiguous", reason="multiple_statement_dates"
            )
        # One combined input cannot bind an elapsed-time expression to its
        # originating fragment. Calendar words remain usable on the same day.
        if end > stamp and _ELAPSED.search(text):
            return TemporalQuery(
                **base, status="ambiguous", reason="multiple_statement_instants"
            )
    # A single explicitly named as-of day is the requested reference clock,
    # not proof that the fact happened on that day. A leading "DATE当时"
    # also permits questions about a recording studio or a story being told;
    # broad record/mention keywords must not unbind this explicit anchor.
    # The answer checker still distinguishes telling from the narrated event.
    as_of_dates = list(_DATED.finditer(text))
    leading_references = list(re.finditer(_DATED.pattern + r"\s*(?:的\s*)?当时", text))
    leading_reference = (
        len(leading_references) == 1
        and not text[: leading_references[0].start()].strip()
    )
    # A terminal clarification names the event-reference day, not when the
    # user asked a past question. Remove only this syntactic cue for the
    # statement-clock guard; preserve the original text and source offsets.
    reference_text = re.sub(
        r"问的是(?=\s*" + _DATED.pattern + r"\s*[?？。!！]?\s*$)", "", text
    )
    if (
        base["operation"].type == "as_of"
        and (len(as_of_dates) == 1 or leading_reference)
        and not any(m.group() != "当时" for m in _VAGUE.finditer(text))
        and not _IMPRECISE.search(text)
        and (
            (
                leading_reference
                and re.match(
                    r"\s*(?:我|你|他|她|它)", text[leading_references[0].end() :]
                )
            )
            or not _REFERENCE_CLOCK.search(reference_text)
        )
        and (leading_reference or not _YEARLESS.search(_DATED.sub("", text)))
        and not re.search(
            r"[“”‘’「」『』\"']|如果|假如|假设|要是|并非|(?<!是)不是|别按|不要按", text
        )
    ):
        match = as_of_dates[0]
        bound = resolve_temporal_query(
            match.group(),
            anchor=stamp,
            anchor_source=anchor_source,
            timezone_name=zone_name,
            anchor_end=anchor_end,
        )
        if bound.status == "resolved":
            return replace(
                bound,
                operation=base["operation"],
                timezone_source=zone_source,
                binding={
                    "version": 1,
                    "decision": "explicit_as_of_reference",
                    "offset_unit": "unicode_codepoint",
                    "spans": [{"start": match.start(), "end": match.end()}],
                },
            )
    # A vague expression or an omitted year must not acquire a precise date
    # simply because a parser happens to recognize another substring.
    # An immediately dated “当时” refers to that date; an unbound “当时”
    # remains ambiguous. Keep the original text/offsets and all other vague
    # expressions, rather than stripping deictics across clauses.
    vague = any(
        not (
            match.group() == "当时"
            and re.search(_DATED.pattern + r"\s*(?:的\s*)?$", text[: match.start()])
        )
        for match in _VAGUE.finditer(text)
    )
    if vague or _IMPRECISE.search(text):
        return TemporalQuery(**base, status="ambiguous", reason="vague_expression")
    original_text = text
    source = SourceText.original(text).substitute(
        _ALIAS_PATTERN, lambda m: _ALIASES[m.group()]
    )
    text = source.text
    # Holiday cache may choose the nearest year or even shift a lunar holiday
    # by a Gregorian year. A read query needs explicit, qualified calendar
    # evidence before such a date can restrict recall.
    from app.services.schedule_domain import holiday_cache

    if _HOLIDAY.search(text) or any(
        name in text for name in holiday_cache.all_known_names()
    ):
        return TemporalQuery(
            **base, status="ambiguous", reason="holiday_calendar_required"
        )
    # Relative years are explicit when bound to a calendar date. Resolve them
    # before the legacy parser's overlapping year/month patterns can discard it.
    year = stamp.astimezone(zone).year
    source = source.substitute(
        re.compile(r"(今年|去年|前年)\s*(?=\d{1,2}\s*月\s*\d{1,2}\s*[日号])"),
        lambda m: str(year - {"今年": 0, "去年": 1, "前年": 2}[m.group(1)]) + "年",
    )
    # Only an adjacent, explicitly connected range may inherit its left year.
    # Do not borrow a year from a separate statement/event or assume rollover.
    source = source.substitute(
        re.compile(
            r"(\d{4})年(\d{1,2}月\d{1,2}[日号]\s*"
            + RANGE_CONNECTOR + r"\s*)(\d{1,2}月\d{1,2}[日号])"
        ),
        lambda m: m.group(1) + "年" + m.group(2) + m.group(1) + "年" + m.group(3),
    )
    text = source.text
    if _YEARLESS.search(_DATED.sub("", text)):
        return TemporalQuery(**base, status="ambiguous", reason="year_unspecified")
    if _APPROXIMATE.search(text):
        return TemporalQuery(**base, status="ambiguous", reason="approximate_duration")
    try:
        for match in _DATED.finditer(text):
            date(*(int(v) for v in match.groups()))
        parsed = parse_time_expressions(
            text,
            now=stamp,
            calendar_timezone=zone,
            all_occurrences=True,
            max_results=MAX_EXPRESSIONS,
            bare_months=bare_months,
        )
    except TimeExpressionLimitExceeded:
        base["date_basis"] = "unbound_expression"
        base["binding"] = binding_diagnostics(source, [], "expression_limit")
        return TemporalQuery(**base, status="unresolved", reason="expression_limit")
    except (ValueError, OverflowError):
        return TemporalQuery(**base, status="unresolved", reason="invalid_expression")
    if len(parsed) > MAX_EXPRESSIONS:
        base["date_basis"] = "unbound_expression"
        base["binding"] = binding_diagnostics(source, [], "expression_limit")
        return TemporalQuery(**base, status="unresolved", reason="expression_limit")
    if not parsed:
        return TemporalQuery(**base, status="not_requested", reason="no_expression")
    parts, error = locate_parts(source, parsed, MAX_EXPRESSIONS)
    if error:
        base["date_basis"] = "unbound_expression"
        base["binding"] = binding_diagnostics(source, [], error)
        return TemporalQuery(**base, status="unresolved", reason=error)
    uncertainty = binding_uncertainty(original_text)
    base["binding"] = binding_diagnostics(
        source, parts, uncertainty or "single_expression"
    )
    if uncertainty:
        base["date_basis"] = "unbound_expression"
        return TemporalQuery(**base, status="ambiguous", reason="unbound_expression")
    # Never intersect dates separated by prose, even when their periods overlap.
    # Separate occurrences of the same spelling are separate evidence positions.
    ordered = [part.parsed for part in parts]
    # Bind only adjacent calendar-day + time-of-day expressions. "昨天中午"
    # is one period; "昨天去了，今天中午又去" is not. Never combine across prose.
    bound_range = None
    end_boundary = ordered[0].end_boundary
    decision = "single_expression"
    if len(parts) == 2:
        left, right = ordered
        left_end, right_start = parts[0].end, parts[1].start
        if is_range_connector(text[left_end:right_start]):
            if left.start > right.start:
                base["binding"] = binding_diagnostics(source, parts, "reversed_range")
                return TemporalQuery(
                    **base, status="unresolved", reason="reversed_range"
                )
            bound_range = left.start, right.end
            end_boundary = right.end_boundary
            decision = "explicit_range"
        days = [
            p
            for p in ordered
            if p.start.hour == 0
            and p.start.minute == 0
            and p.end.hour == 23
            and p.end.minute == 59
            and p.start.date() == p.end.date()
        ]
        if len(days) == 1:
            day = days[0]
            part = next(p for p in ordered if p is not day)
            adjacent = day is left and re.fullmatch(
                r"\s*(?:的)?\s*", text[left_end:right_start]
            )
            if (
                not bound_range
                and adjacent
                and re.search(
                    r"点|时|早|上午|中午|下午|晚上|凌晨|深夜", part.original_text
                )
            ):
                delta = day.start.date() - stamp.astimezone(zone).date()
                bound_range = part.start + delta, part.end + delta
                end_boundary = part.end_boundary
                decision = "adjacent_day_time"
    if len(parts) > 1 and bound_range is None:
        if base["date_basis"] == "event_candidate":
            base["date_basis"] = "unbound_expression"
        base["binding"] = binding_diagnostics(source, parts, "separate_expressions")
        return TemporalQuery(**base, status="ambiguous", reason="multiple_periods")
    base["binding"] = binding_diagnostics(
        source,
        parts,
        "reference_unbound" if base["date_basis"] == "unbound_reference" else decision,
    )
    start, end = bound_range or (ordered[0].start, ordered[0].end)
    if _CLOCK_MINUTE.search(ordered[-1].original_text):
        # Refine read precision only after binding the calendar date. Refining
        # on the statement's day first can carry its DST transition into a
        # different event day. The legacy scheduling parser remains unchanged.
        minute_start = ordered[-1].start if decision == "explicit_range" else start
        if not _unique_local_clock(minute_start):
            return TemporalQuery(**base, status="ambiguous", reason="ambiguous_clock")
        end = (minute_start.astimezone(timezone.utc) + timedelta(minutes=1)).astimezone(
            zone
        )
        end_boundary = "exclusive"
    if start > end:
        return TemporalQuery(**base, status="ambiguous", reason="multiple_periods")
    # Elapsed offsets have an unambiguous UTC anchor even during a repeated
    # local hour. Calendar wall times instead need explicit disambiguation.
    elapsed = len(parsed) == 1 and re.search(
        r"(?:分钟|小时)(?:之)?[前后]", parsed[0].original_text
    )
    if not elapsed and not all(_unique_local_clock(t) for t in (start, end)):
        return TemporalQuery(**base, status="ambiguous", reason="ambiguous_clock")
    local_start, local_end = start.astimezone(zone), end.astimezone(zone)
    precision = (
        "day"
        if local_start.date() == local_end.date()
        and local_start.hour == 0
        and local_end.hour == 23
        else "range"
    )
    if decision in {"single_expression", "adjacent_day_time"}:
        if _CLOCK_MINUTE.search(ordered[-1].original_text):
            precision = "minute"
        elif _CLOCK_HOUR.search(ordered[-1].original_text):
            precision = "hour"
    # Overflow or a non-positive real-time window is not a usable SQL range.
    try:
        if exclusive_end(end, end_boundary) <= parse_time(start):
            raise ValueError("invalid_interval")
    except (ValueError, OverflowError):
        return TemporalQuery(**base, status="unresolved", reason="invalid_expression")
    expression = TemporalExpression(start, end, precision, parsed[0].type, end_boundary)
    if base["operation"].type == "unspecified":
        if start > stamp:
            # 只有问自己的安排才查计划记忆: 问 AI 的、问天气的、陈述句都不算
            own = asks_own_plan(text) or _state_question_or_followup(text, context_text, parsed)
            base["operation"] = TemporalOperation("planned" if own else "range")
        elif _past_state_question(text, context_text, parsed, exclusive_end(end, end_boundary), stamp, zone):
            # 过去某段时间的状态 ("上个月我在哪上班") 按有效期回答, 不按事件日期筛。
            base["operation"] = TemporalOperation("as_of", inferred=True)
        elif exclusive_end(end, end_boundary) > stamp and _state_question_or_followup(
                text, context_text, parsed):
            # 包含现在的一段时间问状态 ("我这周是不是在休假" / 问完去年再问 "那今年呢")
            base["operation"] = TemporalOperation("current", inferred=True)
        else:
            base["operation"] = TemporalOperation("range")
    return TemporalQuery(
        **base,
        status="resolved",
        reason="explicit_range",
        expressions=(expression,),
        kind="planned" if start > stamp else "range",
    )
