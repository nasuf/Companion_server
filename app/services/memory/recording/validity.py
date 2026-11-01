"""Source-anchored temporal assertions, never model-supplied timestamps.

An assertion means that the speaker said this. It is not independent proof of
the fact, and an unknown endpoint is not an infinite guarantee. Keep this pure
so extraction replay can freeze its result before any transactional write.
"""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.services.time_domain.clock import ensure_aware
from app.services.time_domain.parser import normalize_clock_words
from app.services.time_domain.query import resolve_temporal_query
from app.services.time_domain.timestamps import exclusive_end

KINDS = frozenset({"point", "state", "stable", "plan", "recurring"})
STATES = frozenset(
    {"planned", "occurred", "ongoing", "completed", "cancelled", "uncertain"}
)
_ALLOWED_STATES = {
    "point": {"occurred", "uncertain"},
    "state": {"ongoing", "completed", "uncertain"},
    "stable": {"ongoing", "uncertain"},
    "plan": {"planned", "cancelled", "completed", "uncertain"},
    "recurring": {"ongoing", "cancelled", "uncertain"},
}
_UNCERTAIN = re.compile(
    r"如果|假如|假设|要是|万一|可能|也许|大概|好像|据说|听说|[?？]|是不是|是否|有没有|会不会|能不能|想不想|吗[呀啊呢。！!…\s]*$"
)
_ATTRIBUTION = re.compile(
    r"[“”\"「」『』]|(?:他|她|别人|同事|朋友|姐姐|哥哥|妈妈|爸爸).{0,4}(?:说|告诉|觉得)|我(?:的)?(?:妈妈|爸爸|朋友|同事|家人|姐姐|哥哥|弟弟|妹妹|儿子|女儿)"
)
_NEGATION = re.compile(
    r"没有|并未|不曾|不是|尚未|未曾|否认|"
    r"没(?:说|讲|提|告诉|住|去|在|参加|工作|实习|开始|结束|完成)|"
    r"不(?:打算|准备|计划|想|会|要|去|住|在|参加)"
)
_PLAN = re.compile(r"计划|打算|准备|安排|预定|预约")
_CANCEL = re.compile(r"取消|作废|不去了|不去啦")
_COMPLETE = re.compile(r"完成|结束|痊愈|搬离|离开|到.+(?:为止|期间)|从.+到")
_ONGOING = re.compile(r"现在|目前|还在|仍在|正在|开始|一直|始终")
_SELF = re.compile(
    r"我(?!们|的|妈|爸|哥|姐|弟|妹|儿|女|朋友|同事)|我的(?:疗程|行程|工作|旅行|考试)|本人"
)
_KEYS = frozenset(
    {"kind", "status", "source_index", "quote", "valid_from_quote", "valid_to_quote"}
)


def _unknown(reason: str) -> dict:
    return {"version": 1, "state": "unknown", "reason": reason}


def _bound(quote: object, source_quote: str, anchor: datetime, *, end: bool):
    if quote in (None, ""):
        return None, None
    if (
        not isinstance(quote, str)
        or len(quote) > 100
        or not quote.strip()
        or quote not in source_quote
    ):
        raise ValueError("unanchored_bound")
    # Models may quote the complete boundary phrase. Remove only boundary
    # markers from the already matched original, never synthesize a date.
    quote = re.sub(
        r"^(?:到|至|截止|截至)\s*" if end else r"^(?:自从|从|自)\s*", "", quote.strip()
    )
    quote = re.sub(r"\s*(?:为止|截止)$" if end else r"\s*(?:开始|起)$", "", quote)
    if not quote or quote not in source_quote:
        raise ValueError("unanchored_bound")
    # A point-event date cannot silently become a fact's validity start.
    if source_quote.count(quote) != 1:
        raise ValueError("ambiguous_bound_occurrence")
    offset = source_quote.index(quote)
    before, after = (
        source_quote[max(0, offset - 4) : offset],
        source_quote[offset + len(quote) :],
    )
    marker = r"(?:到|至|截止|截至)\s*$" if end else r"(?:从|自|自从)\s*$"
    if not re.search(marker, before) and not (
        not end and re.match(r"\s*(?:起|开始|起至|起到)", after)
    ):
        raise ValueError("unbound_validity_date")
    parsed = resolve_temporal_query(
        normalize_clock_words(quote), anchor=anchor, anchor_source="persisted_message"
    )
    if end and parsed.reason == "year_unspecified":
        # The right endpoint can inherit the explicitly written year of its
        # adjacent left endpoint. It cannot borrow the statement's year or
        # cross prose referring to another event. Reuse the shared range parser.
        left = re.search(
            r"\d{4}年\d{1,2}月\d{1,2}[日号]\s*(?:到|至|—|~|～)\s*$",
            source_quote[:offset],
        )
        if left:
            joined = resolve_temporal_query(
                left.group() + quote, anchor=anchor, anchor_source="persisted_message"
            )
            if joined.status == "resolved" and len(joined.expressions) == 1:
                period = joined.expressions[0]
                return exclusive_end(period.end, period.end_boundary).isoformat(), "day"
    if parsed.status != "resolved" or len(parsed.expressions) != 1:
        raise ValueError("ambiguous_bound")
    expression = parsed.expressions[0]
    if expression.precision in {"range", "day_part"}:
        raise ValueError("imprecise_bound")
    # 'through Tuesday' includes that calendar day; 'until 08:00' ends at 08:00.
    value = (
        exclusive_end(expression.end, expression.end_boundary)
        if end and expression.precision in {"day", "week", "month", "year"}
        else expression.start
    )
    return ensure_aware(value).isoformat(), expression.precision


def resolve_validity(memory: dict, origins, indices: tuple[int, ...]) -> dict:
    proposal = memory.get("temporal_assertion")
    if proposal is None:
        return _unknown("not_proposed")
    if not isinstance(proposal, dict) or set(proposal) - _KEYS:
        return _unknown("invalid_contract")
    kind, status = proposal.get("kind"), proposal.get("status")
    index, quote = proposal.get("source_index"), proposal.get("quote")
    if (
        not isinstance(kind, str)
        or kind not in KINDS
        or not isinstance(status, str)
        or status not in STATES
        or type(index) is not int
        or index not in indices
        or not 1 <= index <= len(origins)
        or not isinstance(quote, str)
        or not 1 <= len(quote) <= 500
    ):
        return _unknown("invalid_contract")
    source = origins[index - 1]
    text = source.input_text or ""
    if (
        source.kind != "message"
        or source.expected_role not in {"user", "assistant"}
        or source.expected_statement_time is None
        or source.expected_version is None
        or quote not in text
        or not _SELF.search(quote)
    ):
        return _unknown("unanchored_subject")
    if sha256(text.encode("utf-8")).hexdigest() != source.expected_version:
        # Card renderers append descriptions (and possibly first-person song
        # titles) that are not the speaker's original message. Keep the memory
        # extraction, but do not create a temporal claim from rendered text.
        return _unknown("rendered_source_not_plaintext")
    # Inspect the whole source, not a cherry-picked substring inside a quote.
    if _UNCERTAIN.search(text) or _ATTRIBUTION.search(text) or _NEGATION.search(text):
        return _unknown("non_assertive_source")
    if status not in _ALLOWED_STATES[kind]:
        return _unknown("inconsistent_state")
    if (
        (status == "cancelled" and not _CANCEL.search(quote))
        or (status == "completed" and not _COMPLETE.search(quote))
        or (
            status == "ongoing"
            and not (
                _ONGOING.search(quote)
                or kind == "recurring"
                and re.search(r"每天|每日|每周|每月|每年|每星期|每隔", quote)
            )
        )
        or (
            status == "occurred"
            and not re.search(r"了|过|完成|拿到|参加|做了|去的|做的|到的", quote)
        )
        or (
            kind == "recurring"
            and not re.search(r"每天|每日|每周|每月|每年|每星期|每隔", quote)
        )
        or (kind != "plan" and _PLAN.search(quote))
    ):
        return _unknown("state_not_supported")
    try:
        anchor = ensure_aware(source.expected_statement_time)
        start, start_precision = _bound(
            proposal.get("valid_from_quote"), quote, anchor, end=False
        )
        end, end_precision = _bound(
            proposal.get("valid_to_quote"), quote, anchor, end=True
        )
        if (
            start
            and end
            and datetime.fromisoformat(start) >= datetime.fromisoformat(end)
        ):
            raise ValueError("inverted_interval")
        if kind == "point" and (start or end):
            raise ValueError("point_is_not_validity")
        if (
            status not in {"planned", "uncertain"}
            and start
            and datetime.fromisoformat(start) > anchor
        ):
            raise ValueError("future_is_not_current")
        if status == "completed" and end and datetime.fromisoformat(end) > anchor:
            raise ValueError("future_is_not_completed")
        if status == "planned" and not (
            _PLAN.search(quote) or (start and datetime.fromisoformat(start) > anchor)
        ):
            raise ValueError("plan_not_supported")
    except (ValueError, TypeError):
        return _unknown("invalid_validity_bounds")
    from app.config import settings

    return {
        "version": 1,
        "state": "asserted",
        "kind": kind,
        "status": status,
        "source_index": index,
        "source_version": source.expected_version,
        "source_quote": quote,
        "valid_from_quote": proposal.get("valid_from_quote"),
        "valid_to_quote": proposal.get("valid_to_quote"),
        "source_role": source.expected_role,
        "statement_at": anchor.isoformat(),
        "valid_from": start,
        "valid_to": end,
        "from_precision": start_precision,
        "to_precision": end_precision,
        "timezone": settings.schedule_timezone,
        "end_boundary": "exclusive",
        "truth": "speaker_assertion",
    }


def public_validity(value: object) -> dict:
    """Validate persisted metadata too; old/invalid records remain unknown."""
    if (
        not isinstance(value, dict)
        or type(value.get("version")) is not int
        or value["version"] != 1
    ):
        return _unknown("not_collected")
    if value.get("state") != "asserted":
        return _unknown("not_established")
    if (
        not isinstance(value.get("kind"), str)
        or value["kind"] not in KINDS
        or not isinstance(value.get("status"), str)
        or value["status"] not in STATES
        or value["status"] not in _ALLOWED_STATES[value["kind"]]
        or value.get("truth") != "speaker_assertion"
        or value.get("end_boundary") != "exclusive"
    ):
        return _unknown("invalid_metadata")
    try:
        zone = value["timezone"]
        if not isinstance(zone, str) or len(zone) > 64:
            raise ValueError()
        ZoneInfo(zone)
        dates = {}
        for key in ("valid_from", "valid_to", "statement_at"):
            raw = value.get(key)
            if raw is None and key != "statement_at":
                dates[key] = None
                continue
            if not isinstance(raw, str) or len(raw) > 40:
                raise ValueError()
            parsed = datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                raise ValueError()
            dates[key] = parsed.isoformat()
        if value["kind"] == "point" and (dates["valid_from"] or dates["valid_to"]):
            raise ValueError()
        if (
            dates["valid_from"]
            and dates["valid_to"]
            and datetime.fromisoformat(dates["valid_from"])
            >= datetime.fromisoformat(dates["valid_to"])
        ):
            raise ValueError()
        for field in ("from_precision", "to_precision"):
            if value.get(field) is not None and (
                not isinstance(value[field], str)
                or value[field]
                not in {"instant", "minute", "hour", "day", "week", "month", "year"}
            ):
                raise ValueError()
    except (ValueError, TypeError, KeyError, ZoneInfoNotFoundError):
        return _unknown("invalid_metadata")
    return {
        **{
            key: value.get(key)
            for key in (
                "version",
                "state",
                "kind",
                "status",
                "timezone",
                "end_boundary",
                "truth",
                "from_precision",
                "to_precision",
            )
        },
        **dates,
    }


def valid_at(value: object, at: datetime) -> str:
    """Three-valued interval membership; open unknown bounds cannot prove yes."""
    record = public_validity(value)
    if record["state"] != "asserted" or record["kind"] == "point":
        return "unknown"
    at = ensure_aware(at)
    start, end = record["valid_from"], record["valid_to"]
    if (
        start
        and at < datetime.fromisoformat(start)
        or end
        and at >= datetime.fromisoformat(end)
    ):
        return "outside"
    return "inside" if start and end else "unknown"
