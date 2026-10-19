"""Bounded interpretation of provider hours. Unrecognised rules stay unknown.

Raw provider text is retained for audit, never copied into the public fallback.
This is a visit hint, not a guarantee of admission or a replacement for a dated
event session. In particular a closed venue must not cancel its future concert.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

_TZ = ZoneInfo("Asia/Shanghai")
_DAY = "一二三四五六日"
_WEEK = re.compile(r"(?:周|星期|礼拜)([一二三四五六日天])(?:\s*[-–—~～至到]\s*(?:周|星期|礼拜)?([一二三四五六日天]))?")
_DATE = re.compile(r"(?<!\d)(?:(\d{4})[/年-])?(\d{1,2})[/月](\d{1,2})日?(?:\s*[-–—~～至]\s*(?:(\d{4})[/年-])?(\d{1,2})[/月](\d{1,2})日?)?")
_HOURS = re.compile(r"(?<!\d)([0-2]?\d)[:：]([0-5]\d)\s*[-–—~～至到]\s*([0-2]?\d)[:：]([0-5]\d)")
_CLOSED = re.compile(r"不开放|不营业|闭馆|闭园|闭店|歇业|暂停营业|暂停开放|休息|休馆")
_HOLIDAY = re.compile(r"节假日|法定假日|春节|清明|劳动节|五一|端午|中秋|国庆|元旦|暑期|寒假|节日")


def _days(text: str) -> set[int]:
    days: set[int] = set()
    for match in _WEEK.finditer(text):
        start = _DAY.index(match[1].replace("天", "日"))
        end = _DAY.index((match[2] or match[1]).replace("天", "日"))
        days.update((start + n) % 7 for n in range((end - start) % 7 + 1))
    return days


def _date_applies(match: re.Match, day: date, checked_year: int) -> bool | None:
    try:
        year = int(match[1]) if match[1] else checked_year
        start = date(year, int(match[2]), int(match[3]))
        end = date(int(match[4]) if match[4] else year,
                   int(match[5] or match[2]), int(match[6] or match[3]))
        if end < start:
            end = end.replace(year=end.year + 1)
            if not match[1] and not match[4] and day < start:
                start = start.replace(year=start.year - 1)
                end = end.replace(year=end.year - 1)
        # Yearless exceptions from a previous query year cannot roll forward.
        return start <= day <= end
    except ValueError:
        return None


def _holiday_might_apply(text: str, day: date) -> bool:
    # Reuse the project's holiday data rather than hard-coding vacation spans.
    # Unsupported years/seasonal rules remain uncertain, never presumed expired.
    from chinese_calendar import get_holiday_detail
    try:
        holiday, name = get_holiday_detail(day)
    except (ValueError, NotImplementedError):
        return True
    for label, expected in (("国庆", "National Day"), ("元旦", "New Year's Day"),
                            ("劳动节", "Labour Day"), ("五一", "Labour Day"),
                            ("春节", "Spring Festival"), ("清明", "Tomb-sweeping Day"),
                            ("端午", "Dragon Boat Festival"), ("中秋", "Mid-autumn Festival")):
        if label in text:
            return name == expected
    if "法定" in text:
        return bool(name)
    if "节假日" in text or "假日" in text:
        return holiday
    return True


def _rules(raw: str, day: date, checked_year: int) -> tuple[list[tuple[int, list[tuple[int, int]], bool]], int]:
    rules, uncertain = [], 0
    # A comma between weekdays belongs to one rule. A comma after an action or
    # end time can separate two rules ("周一闭馆，周二至周日9:00-17:00").
    raw = re.sub(r"(闭馆|闭园|休息|休馆|不开放|不营业|\d[:：]\d{2})[,，]\s*(?=(?:周|星期|礼拜))", r"\1；", raw)
    for clause in re.split(r"[；;\n]+", raw[:2000]):
        clause = clause.strip()
        if not clause:
            continue
        clause = re.sub(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)", r"\1/\2/\3", clause)
        dates = list(_DATE.finditer(clause))
        if len(dates) > 1:
            uncertain = 4
            continue
        if dates:
            applies = _date_applies(dates[0], day, checked_year)
            if applies is None:
                uncertain = 4
                continue
            if not applies:
                continue
        weekdays = _days(clause)
        if weekdays and day.weekday() not in weekdays:
            continue
        if _HOLIDAY.search(clause):
            if _holiday_might_apply(clause, day):
                uncertain = max(uncertain, 3 if dates and not dates[0][5] else 2)
            continue
        closed = bool(_CLOSED.search(clause))
        hours = []
        for match in _HOURS.finditer(clause):
            start, end = int(match[1]) * 60 + int(match[2]), int(match[3]) * 60 + int(match[4])
            if start >= 1440 or end > 1440 or start == end:
                uncertain = 4
                continue
            hours.append((start, end))
        if not closed and not hours:
            # Explicit 24-hour access, but never "非全天" or "全天不开放".
            if re.fullmatch(r"(?:每天|每日|全天)?\s*(?:24小时|全天)(?:营业|开放)?", clause):
                hours = [(0, 1440)]
            else:
                uncertain = 4
                continue
        if closed and hours:
            uncertain = 4
            continue
        # Dated exceptions override recurring rules; exact days outrank ranges.
        priority = (3 if dates and not dates[0][5] else 2 if dates else 1)
        rules.append((priority, hours, closed))
    return rules, uncertain


def assess_opening_hours(card: dict[str, Any], now: datetime) -> dict[str, Any]:
    metadata = card.get("discovery_metadata") or {}
    local = now.replace(tzinfo=_TZ) if now.tzinfo is None else now.astimezone(_TZ)
    result: dict[str, Any] = {"status": "unknown", "reason": "missing", "assessed_at": local.isoformat(),
                              "reference_date": local.date().isoformat(), "intervals": []}
    if metadata.get("kind") == "event" or card.get("starts_at"):
        return {**result, "status": "not_applicable", "reason": "use_event_session"}
    raw = str(metadata.get("opening_hours") or "").strip()
    if not raw:
        return result
    try:
        checked = datetime.fromisoformat(str(metadata.get("checked_at") or local.isoformat()).replace("Z", "+00:00"))
        checked_year = checked.astimezone(_TZ).year if checked.tzinfo else checked.year
    except ValueError:
        return {**result, "reason": "invalid_reference_date"}
    if checked_year != local.year:
        return {**result, "reason": "outdated_reference_year"}
    rules, uncertain = _rules(raw, local.date(), checked_year)
    if uncertain and (not rules or max(rule[0] for rule in rules) <= uncertain):
        return {**result, "reason": "unresolved_rules"}
    # A night-opening interval can extend from the previous day's rule.
    previous, previous_uncertain = _rules(raw, local.date() - timedelta(days=1), checked_year)
    carry = []
    if previous and not previous_uncertain:
        priority = max(r[0] for r in previous)
        winners = [r for r in previous if r[0] == priority]
        if not any(r[2] for r in winners):
            carry = [(0, end) for _, hours, _ in winners for start, end in hours if 0 < end < start]
    if not rules and not carry:
        return {**result, "reason": "no_applicable_rule"}
    if rules:
        priority = max(r[0] for r in rules)
        winners = [r for r in rules if r[0] == priority]
        if any(r[2] for r in winners) and any(not r[2] for r in winners):
            return {**result, "reason": "conflicting_rules"}
        if all(r[2] for r in winners):
            return {**result, "status": "closed", "reason": "closed_day"}
        intervals = [(start, end if end > start else 1440) for _, hours, _ in winners for start, end in hours]
    else:
        intervals = []
    intervals = sorted(set(intervals + carry))
    minute = local.hour * 60 + local.minute
    if not rules and not any(start <= minute < end for start, end in carry):
        return {**result, "reason": "no_applicable_rule"}
    opened = any(start <= minute < end for start, end in intervals)
    reason = "within_hours" if opened else "before_opening" if intervals and minute < intervals[0][0] else "after_closing"
    return {**result, "status": "open" if opened else "closed", "reason": reason,
            "intervals": [[f"{start // 60:02}:{start % 60:02}", f"{end // 60:02}:{end % 60:02}"] for start, end in intervals]}


def visit_context(card: dict[str, Any], now: datetime) -> str:
    assessment = assess_opening_hours(card, now)
    if assessment["status"] == "not_applicable":
        return "临时活动按本场日期和场次安排，不用场馆日常营业状态代替活动时间。"
    moment = assessment["assessed_at"][:16].replace("T", " ")
    hours = "、".join("–".join(pair) for pair in assessment["intervals"])
    if assessment["status"] == "unknown":
        return f"当前北京时间 {moment}。营业安排不能可靠判断，不断言现在开放或邀请立即出发；可建议另找方便时段并在出发前确认。"
    if assessment["status"] == "closed":
        return f"当前北京时间 {moment}，按参考营业规则目前未开放{('；当天参考时段 ' + hours) if hours else ''}。不邀请现在、今晚或马上去，可推荐另选开放时段。"
    return f"当前北京时间 {moment}，按参考规则在开放时段内（{hours}）；不保证实时营业、可入场或来得及到达，出发前仍需确认。"


def public_hours(card: dict[str, Any], now: datetime) -> str:
    assessment = assess_opening_hours(card, now)
    hours = "、".join("–".join(pair) for pair in assessment["intervals"])
    if hours:
        return f"开放参考（{assessment['reference_date']}）：{hours}；出发前确认当天安排。"
    if assessment["reason"] == "closed_day":
        return f"开放参考（{assessment['reference_date']}）：当天不开放，建议另选开放时段。"
    return "出发前确认当天开放安排。"


def visit_issues(text: str, card: dict[str, Any], now: datetime) -> list[str]:
    assessment = assess_opening_hours(card, now)
    if assessment["status"] == "not_applicable":
        return []
    for sentence in re.split(r"[。！？!?\n]+", text):
        if re.search(r"别去|不去|不建议|不能|(?<!要)不要|另选|改天|明天|下次|等到|已经闭馆|已过.*时段", sentence):
            continue
        immediate = re.search(r"(?:现在|这会儿|马上|立刻|今晚|今天就).{0,30}(?:去|出发|逛|看展)", sentence)
        current_open = re.search(r"(?:现在|这会儿|今晚).{0,8}(?:开放|营业)|(?:今天|今晚).{0,8}(?:开放|营业).{0,35}(?:去|逛|看展)", sentence)
        if assessment["status"] == "closed" and (immediate or current_open):
            return ["按当前参考营业规则尚未开放或已闭馆，删除立即出发的邀请，改为另选开放时段。"]
        if assessment["status"] == "unknown" and current_open:
            return ["营业安排无法可靠判断，不能断言现在或今晚开放，请改为出发前确认开放时间。"]
    return []
