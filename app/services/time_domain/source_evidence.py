"""Bounded source-date syntax and calendar support for historical answers.

These helpers never establish subject, entailment or event truth. They reuse the
shared calendar parser and require a separate semantic answer check.
"""

from __future__ import annotations

import re
from zoneinfo import ZoneInfo

from app.services.time_domain.timestamps import parse_time

MAX_SOURCE_RECORDS = 20


def source_date_spans(text: str, *, anchor, timezone_name: str) -> list[dict]:
    """Calendar syntax candidates only: never assert the subject or truth.

    Read literal parser spans so a correction/negative assertion can retain its
    date. The separate semantic check must still bind it to the answered fact.
    Only adjacent range endpoints are joined. A year may be shared by an
    ordered date list in one sentence; otherwise missing years remain unknown.
    """
    from app.services.time_domain.parser import (
        parse_time_expressions,
        TimeExpressionLimitExceeded,
    )
    from app.services.time_domain.binding import SourceText, locate_parts, is_range_connector
    from app.services.time_domain.query import resolve_temporal_query
    from app.services.time_domain.timestamps import exclusive_end

    stamp = parse_time(anchor)
    if not stamp or not isinstance(text, str) or len(text) > 4000:
        return []
    try:
        parts, error = locate_parts(
            SourceText.original(text),
            parse_time_expressions(
                text,
                now=stamp,
                calendar_timezone=ZoneInfo(timezone_name),
                all_occurrences=True,
                max_results=8,
            ),
            8,
        )
        if error:
            return []
        spans, i = [], 0
        while i < len(parts):
            begin, end = parts[i].start, parts[i].end
            if i + 1 < len(parts) and is_range_connector(text[end : parts[i + 1].start]):
                end = parts[i + 1].end
                i += 1
            quote = text[begin:end]
            parsed = resolve_temporal_query(
                quote,
                anchor=stamp,
                anchor_source="persisted_message",
                timezone_name=timezone_name,
            )
            year_reference = None
            if parsed.reason == "year_unspecified" and re.fullmatch(
                r"\d{1,2}月\d{1,2}[日号]", quote
            ):
                # This is a calendar reference within an ordered date list,
                # not borrowing the statement year. Never cross a sentence,
                # quoted material, relative-year language or a year rollover.
                sentence_start = (
                    max(
                        (text.rfind(mark, 0, begin) for mark in "。！？!?；;\n"),
                        default=-1,
                    )
                    + 1
                )
                sentence_end = min(
                    (
                        pos
                        for mark in "。！？!?；;\n"
                        if (pos := text.find(mark, end)) >= 0
                    ),
                    default=len(text),
                )
                sentence = text[sentence_start:sentence_end]
                prefix = text[sentence_start:begin]
                years = set(re.findall(r"(\d{4})年", sentence))
                earlier = re.findall(r"(\d{4})年(\d{1,2})月(\d{1,2})[日号]", prefix)
                excluded = re.search(
                    r"[“”‘’\"'「」『』]|今年|去年|前年|明年|后年|次年|翌年|来年|那年|如果|假如|假设",
                    sentence,
                )
                month, day = map(int, re.findall(r"\d+", quote))
                if (
                    len(years) == 1
                    and earlier
                    and not excluded
                    and not re.search(
                        r"说|提到|讲|记得|记起|聊|告诉|回忆|回顾|想到|提起", prefix
                    )
                ):
                    year, prior_month, prior_day = map(int, earlier[-1])
                    if (month, day) >= (prior_month, prior_day):
                        parsed = resolve_temporal_query(
                            f"{year}年{quote}",
                            anchor=stamp,
                            anchor_source="persisted_message",
                            timezone_name=timezone_name,
                        )
                        year_reference = f"{year}年"
            if parsed.status == "resolved" and len(parsed.expressions) == 1:
                period = parsed.expressions[0]
                # Clock-only syntax is not a date. Anchoring it to when the
                # source was said would silently turn mention time into event time.
                if period.precision in {
                    "instant",
                    "minute",
                    "hour",
                    "day_part",
                    "range",
                } and not re.search(r"年|月|日|号|天|周|星期|礼拜", quote):
                    i += 1
                    continue
                spans.append(
                    {
                        "quote": quote,
                        **(
                            {"year_reference": year_reference} if year_reference else {}
                        ),
                        "precision": period.precision,
                        "start": period.start.isoformat(),
                        "end_exclusive": exclusive_end(
                            period.end, period.end_boundary
                        ).isoformat(),
                    }
                )
            i += 1
        return spans
    except (ValueError, OverflowError, TimeExpressionLimitExceeded):
        return []


def attach_source_date_spans(records: list[dict], *, timezone_name: str) -> None:
    """Attach bounded literal calendars; a nearby deictic keeps its origin.

    This is a reference option for the semantic checker, not an asserted event.
    Never borrow dates from a different storage side, distant turn, ambiguous
    date list, quotation or hypothetical. No new source is fetched here.
    """
    # Reference syntax can occur in a cancellation/correction as well. Those
    # are not asserted facts here; the semantic verifier sees the entire text.
    uncertain_reference = re.compile(
        r"[“”‘’「」『』\"']|如果|假如|假设|假使|倘若|要是|梦见|梦到"
    )
    for record in records:
        record["literal_date_spans"] = [
            dict(span, origin_index=record["index"])
            for span in source_date_spans(
                record.get("source_excerpt") or "",
                anchor=record.get("statement_at"),
                timezone_name=timezone_name,
            )
        ]
    for record in records:
        text = record.get("source_excerpt") or ""
        stamp = parse_time(record.get("statement_at"))
        if (
            record.get("evidence_type") != "recent_statement"
            or record["literal_date_spans"]
            or stamp is None
            or not re.search(r"那天|当天|那一天|同一天|这次|该次|那次", text)
            or uncertain_reference.search(text)
        ):
            continue
        prior = [
            r
            for r in records
            if r.get("evidence_type") == "recent_statement"
            and r.get("source") == record.get("source")
            and (at := parse_time(r.get("statement_at"))) is not None
            and 0 < (stamp - at).total_seconds() <= 1800
        ]
        if not prior:
            continue
        latest = max(parse_time(r["statement_at"]) for r in prior)
        closest = [r for r in prior if parse_time(r["statement_at"]) == latest]
        if len(closest) != 1:
            continue
        origin = closest[0]
        options = list(
            {
                (x["start"], x["end_exclusive"]): x
                for x in origin["literal_date_spans"]
            }.values()
        )
        if (
            len(options) == 1
            and options[0]["precision"] == "day"
            and options[0]["origin_index"] == origin["index"]
            and not uncertain_reference.search(origin.get("source_excerpt") or "")
        ):
            record["literal_date_spans"] = [
                dict(options[0], reference="recent_day_reference")
            ]


def validate_as_of_support(verdict: dict, payload: dict, matches: list[dict]) -> bool:
    """Calendar support is necessary, not sufficient, for a historical answer.

    The semantic checker still verifies subject, negation, plans and conflicts.
    It cannot grant a missing interval endpoint by returning a bare boolean.
    Quotes are local exact source substrings, never generated timestamps.
    """
    from app.services.time_domain.timestamps import exclusive_end

    support = verdict.get("time_support")
    if not isinstance(support, list) or len(support) > MAX_SOURCE_RECORDS:
        raise ValueError("invalid_time_support")
    if not support:
        return False
    requested = payload["requested_validity_range"]
    start, end = parse_time(requested["start"]), parse_time(requested["end"])
    if start is None or end is None:
        raise ValueError("invalid_requested_period")
    boundary = requested.get("end_boundary")
    if boundary not in {"exclusive", "inclusive_second", "inclusive_microsecond"}:
        raise ValueError("unknown_requested_boundary")
    end = exclusive_end(end, boundary)
    matched = {item["index"] for item in matches}
    # Citation associations may include background or a superseded value.
    # They are not a list of independent dated claims. The semantic checker
    # verifies every claim in the answer and names its direct time evidence;
    # calendar validation checks those named sources, not unrelated citations.
    seen = set()
    for item in support:
        if not isinstance(item, dict) or set(item) not in (
            {"index", "quote"},
            {"index", "date_index"},
            {"index", "date_index", "relation"},
        ):
            raise ValueError("invalid_time_support")
        index = item["index"]
        if type(index) is not int or index not in matched or index in seen:
            raise ValueError("invalid_time_support")
        seen.add(index)
        record = payload["memories"][index]
        if "date_index" in item:
            date_index = item["date_index"]
            relation = item.get("relation", "contains")
            if relation not in {"contains", "ended_before"}:
                raise ValueError("invalid_time_support_relation")
            if record.get("validity_membership") == "inside" and date_index is None:
                if relation != "contains":
                    return False
                continue
            options = record.get("literal_date_spans") or []
            if type(date_index) is not int or not 0 <= date_index < len(options):
                raise ValueError("invalid_date_support_index")
            window = options[date_index]
            if relation == "ended_before":
                # Only a bounded stated episode can have an evidenced ending.
                # The checker must restrict the reply to that episode, never
                # assert that no new episode or later event could exist.
                if (
                    window.get("precision") != "range"
                    or parse_time(window["end_exclusive"]) > start
                ):
                    return False
                continue
            if relation != "contains":
                raise ValueError("invalid_time_support_relation")
            if (
                parse_time(window["start"]) > start
                or parse_time(window["end_exclusive"]) < end
            ):
                return False
            continue
        # Read older managed prompt responses during the publication transition.
        quote = item["quote"]
        if record.get("validity_membership") == "inside" and quote is None:
            continue
        source = record.get("source_excerpt") or ""
        if quote is None:
            # A single literal date is unambiguous syntax, not proof of a fact.
            # The semantic verdict above still validates its use in the reply.
            spans = source_date_spans(
                source,
                anchor=record.get("statement_at"),
                timezone_name=payload["calendar"]["timezone"],
            )
        else:
            if not isinstance(quote, str) or not 1 <= len(quote) <= 150:
                raise ValueError("unanchored_time_support")
            offsets = [i for i, char in enumerate(source) if not char.isspace()]
            normalized = "".join(source[i] for i in offsets)
            needle = "".join(char for char in quote if not char.isspace())
            if not needle or normalized.count(needle) != 1:
                raise ValueError("unanchored_time_support")
            offset = normalized.index(needle)
            quote = source[offsets[offset] : offsets[offset + len(needle) - 1] + 1]
            spans = source_date_spans(
                quote,
                anchor=record.get("statement_at"),
                timezone_name=payload["calendar"]["timezone"],
            )
        if len(spans) != 1:
            return False
        window = spans[0]
        if (
            parse_time(window["start"]) > start
            or parse_time(window["end_exclusive"]) < end
        ):
            return False
    return bool(seen)


def calendar_evidence_options(payload: dict) -> list[dict]:
    """Issue request-local labels for calendar containment, never semantic truth.

    Recompute bounds from source spans; model-provided relations or labels never
    create dates. Labels are separate from memory indices to avoid confusing an
    undated supporting attribute with a second independently dated assertion.
    """
    from app.services.time_domain.timestamps import exclusive_end

    requested = payload["requested_validity_range"]
    start, end = parse_time(requested["start"]), parse_time(requested["end"])
    if start is None or end is None or start > end:
        raise ValueError("invalid_requested_period")
    boundary = requested.get("end_boundary")
    if boundary not in {"exclusive", "inclusive_second", "inclusive_microsecond"}:
        raise ValueError("unknown_requested_boundary")
    end = exclusive_end(end, boundary)
    if start >= end:
        raise ValueError("invalid_requested_period")
    records = payload["memories"]
    if len(records) > MAX_SOURCE_RECORDS:
        raise ValueError("invalid_source_record_count")
    options = []
    for record in records:
        if record.get("validity_membership") == "inside":
            options.append(
                {
                    "record_index": record["index"],
                    "date_index": None,
                    "relation": "contains",
                }
            )
        spans = record.get("literal_date_spans") or []
        if len(spans) > 8:
            raise ValueError("invalid_source_date_count")
        for index, span in enumerate(spans):
            begin, finish = parse_time(span["start"]), parse_time(span["end_exclusive"])
            if begin is None or finish is None or begin >= finish:
                raise ValueError("invalid_source_period")
            relation = (
                "contains"
                if begin <= start and finish >= end
                else (
                    "ended_before"
                    if span.get("precision") == "range" and finish <= start
                    else None
                )
            )
            if relation:
                options.append(
                    {
                        "record_index": record["index"],
                        "date_index": index,
                        "relation": relation,
                    }
                )
    return [{"label": f"T{index}", **option} for index, option in enumerate(options)]


def validate_calendar_labels(verdict: dict, payload: dict, matches: list[dict]) -> bool:
    """Resolve only issued labels; then reuse the strict date-bound validator."""
    labels = verdict.get("calendar_evidence_labels")
    if (
        not isinstance(labels, list)
        or len(labels) > MAX_SOURCE_RECORDS
        or any(type(label) is not str for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("invalid_calendar_labels")
    options = {option["label"]: option for option in calendar_evidence_options(payload)}
    if any(label not in options for label in labels):
        raise ValueError("unknown_calendar_label")
    for label in labels:
        option = options[label]
        if not validate_as_of_support(
            {
                "time_support": [
                    {
                        "index": option["record_index"],
                        "date_index": option["date_index"],
                        "relation": option["relation"],
                    }
                ]
            },
            payload,
            matches,
        ):
            return False
    return bool(labels)
