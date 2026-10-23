"""Per-fact clocks derived only from version-pinned message inputs.

The model proposes attribution, never timestamps or ownership. A valid quote
proves presence in the input, not that the fact itself is true.
"""
from dataclasses import dataclass, replace
from datetime import datetime
from zoneinfo import ZoneInfo
from app.config import settings
import json
import re

from app.services.schedule_domain.time_parser import _parse_cn_number

from app.services.temporal_query import resolve_temporal_query
from app.services.temporal import exclusive_end
from app.services.schedule_domain.time_service import ensure_aware


@dataclass(frozen=True)
class FactClock:
    statement_time: datetime | None
    occur_time: datetime | None
    basis: dict
    indices: tuple[int, ...]

    def bind(self, origins):
        # Retain the complete model input as transaction guards. Only selected
        # rows are labelled as proposed fact sources; other rows are batch input.
        return tuple(replace(source, temporal_basis={**self.basis,
            'source_role': 'fact' if i in self.indices else 'batch'})
            for i, source in enumerate(origins, 1))


def source_table(origins) -> str:
    return json.dumps([{'index': i, 'statement_time': source.expected_statement_time.isoformat(),
                        'text': source.input_text}
                       for i, source in enumerate(origins, 1)], ensure_ascii=False)


def resolve_fact_clock(memory: dict, origins) -> FactClock:
    basis = {'version': 1, 'status': 'source_unresolved'}
    raw = memory.get('source_indices')
    if (not isinstance(raw, list) or not 1 <= len(raw) <= len(origins)
            or any(type(i) is not int or not 1 <= i <= len(origins) for i in raw)
            or len(set(raw)) != len(raw)):
        return FactClock(None, None, basis, ())
    indices = tuple(sorted(raw))
    times = [origins[i-1].expected_statement_time for i in indices]
    if any(t is None for t in times):
        return FactClock(None, None, basis, indices)
    start, end = min(times), max(times)
    basis.update(statement_start=start.isoformat(), statement_end=end.isoformat())
    statement = start if start == end else None
    quote, index = memory.get('time_quote'), memory.get('time_source_index')
    if quote is None or quote == '':
        basis['status'] = 'time_not_stated'
        return FactClock(statement, None, basis, indices)
    if (not isinstance(quote, str) or len(quote) > 200 or not quote.strip()
            or type(index) is not int or index not in indices
            or quote not in (origins[index-1].input_text or '')):
        basis['status'] = 'time_quote_invalid'
        return FactClock(statement, None, basis, indices)
    # Normalization only changes a validated time quote, never the source body.
    # Reuse the parser's number grammar; Chinese clock words must not degrade
    # silently into a broad afternoon range.
    normalized = re.sub(r"([零一二三四五六七八九十两]{1,3})(点|时)",
                        lambda m: str(_parse_cn_number(m[1])) + m[2], quote)
    normalized = re.sub(r"(?<=[点时])([零一二三四五六七八九十两]{1,3})分",
                        lambda m: str(_parse_cn_number(m[1])) + "分", normalized)
    normalized = re.sub(r"(?<=[点时])半", "30分", normalized)
    parsed = resolve_temporal_query(normalized, anchor=origins[index-1].expected_statement_time,
                                    anchor_source='persisted_message')
    # A reminder is prospective. Preserve the existing month/day convention,
    # but make its assumed year visible; ordinary recollections never borrow it.
    calendar_inferred = False
    if (parsed.reason == 'year_unspecified' and memory.get('sub_category') == '提醒'
            and re.fullmatch(r"\s*\d{1,2}\s*月\s*\d{1,2}\s*[日号].*", normalized)):
        stamp = origins[index-1].expected_statement_time.astimezone(ZoneInfo(settings.schedule_timezone))
        for year in (stamp.year, stamp.year + 1):
            candidate = resolve_temporal_query(f"{year}年{normalized.strip()}",
                anchor=stamp, anchor_source='persisted_message')
            if (candidate.status == 'resolved' and len(candidate.expressions) == 1
                    and candidate.expressions[0].start > stamp):
                parsed, calendar_inferred = candidate, True
                break
    if parsed.status != 'resolved' or len(parsed.expressions) != 1:
        basis['status'] = 'time_ambiguous'
        return FactClock(statement, None, basis, indices)
    event = parsed.expressions[0]
    occur = ensure_aware(event.start)
    basis.update(timezone=parsed.timezone, precision=event.precision,
                 status='calendar_inferred' if calendar_inferred else 'resolved', event_start=occur.isoformat(),
                 event_end=exclusive_end(event.end, event.end_boundary).isoformat())
    return FactClock(statement, occur, basis, indices)


def uncertain_composite_sources(sources, reason):
    """A merged/split text has no single proven event clock."""
    return tuple(replace(s, temporal_basis={'version': 1, 'status': reason,
                 'source_role': 'batch'}) for s in sources)


_STATUSES = frozenset({'source_unresolved', 'time_not_stated', 'time_quote_invalid',
                      'time_ambiguous', 'resolved', 'calendar_inferred', 'merge_unresolved', 'split_unresolved'})


def public_temporal_basis(value):
    """Bounded public metadata; never echo arbitrary stored keys or source text."""
    if value is None:
        return None
    invalid = {'status': 'invalid_metadata'}
    if (not isinstance(value, dict) or type(value.get('version')) is not int
            or value['version'] != 1 or not isinstance(value.get('status'), str)
            or value['status'] not in _STATUSES
            or not isinstance(value.get('source_role'), str)
            or value['source_role'] not in {'fact', 'batch'}):
        return invalid
    result = {k: value[k] for k in ('version', 'status', 'source_role')}
    for prefix in ('statement', 'event'):
        keys = [prefix + '_start', prefix + '_end']
        if not any(k in value for k in keys):
            if prefix == 'event' and value['status'] in {'resolved', 'calendar_inferred'}:
                return invalid
            continue
        try:
            dates = [datetime.fromisoformat(value[k]) for k in keys
                     if isinstance(value.get(k), str) and len(value[k]) <= 40]
            if len(dates) != 2 or any(d.tzinfo is None for d in dates) or dates[1] < dates[0]:
                return invalid
        except (ValueError, TypeError):
            return invalid
        result.update(zip(keys, (d.isoformat() for d in dates)))
    if 'timezone' in value:
        try:
            if not isinstance(value['timezone'],str) or len(value['timezone'])>64:
                return invalid
            ZoneInfo(value['timezone'])
        except (ValueError,KeyError):
            return invalid
        result['timezone']=value['timezone']
    if 'precision' in value:
        if not isinstance(value['precision'], str) or value['precision'] not in {'instant','minute','hour','day','week','month','year','range','day_part'}:
            return invalid
        result['precision']=value['precision']
    return result
