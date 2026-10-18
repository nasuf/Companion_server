"""Deterministic checks for user-visible activity copy.

The model reviewers miss errors that code can decide exactly. Production traces
show a 4-day-away concert written as "明年十月中旬" and "要不要一起去凑个热闹"
passing review twice. These checks run before any model review, and what they
find goes back to the writer as revision feedback.

Every pattern here is deliberately narrow: a false positive turns a good message
into the generic fallback, which is the failure this module exists to reduce.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from app.config import settings

_TZ = ZoneInfo(settings.schedule_timezone)
_WEEKDAY_CN = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
_WEEKDAY_INDEX = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
# A wrap-around month/day ("1月5日" written in December) is read as next year
# only when it is this close; a farther date names something already past.
_NEXT_YEAR_WINDOW_DAYS = 120
# A long run of dates contains every weekday; weekday words cannot be wrong.
_WEEKDAY_CHECK_MAX_SPAN_DAYS = 14

# Data-availability explanations belong in private diagnostics, never in the
# copy. "开放时间不太固定" is the same leak in disguise: the facts say the hours
# were not provided, not that they vary.
INPUT_DISCLOSURE = re.compile(
    r"用户(?:记忆|喜好|偏好)库|(?:素材|资料|原文|信息|介绍|简介|偏好|时段|时间|价格|票价)[^。！？]{0,16}"
    r"(?:未(?:在[^。！？]{0,8})?(?:提供|提及|列出|写明|注明|说明|明确|确定|公布)|未定|不详|未知"
    r"|没有(?:提供|提到|详细列出|写明|注明|说明)|没(?:有)?写(?![死过完])(?:明|清楚?|全|细|太细|具体)?"
    r"|没(?:有)?(?:细说|说清楚?|具体说|详细说))"
    r"|没(?:有)?(?:写|提|说|给)(?:具体|详细|明确)?的?(?:开放|营业|开门)(?:时间|时段)"
    r"|没有(?:特别)?提到的偏好|没(?:有)?(?:特别|什么|具体)?的?(?:偏好|喜好)(?!的话|[^。！？，]{0,4}(?:也|都)(?:没关系|可以|行))"
    r"|(?:偏好|喜好|聊天)记录"
    r"|(?:尚不清楚|暂不清楚|不清楚|未知)[^。！？]{0,8}(?:开放时间|开放时段|营业时间)"
    r"|(?:开放|营业|开门)(?:时间|时段)?[^。！？，]{0,4}(?:不(?:太|大)?固定|不确定|不一定|说不准)"
    # "这种地方时间不一定固定" (10-17); "时间不一定要长" is not about opening hours.
    r"|(?:时间|时段)[^。！？，]{0,4}不(?:太|大|一定)?固定"
    r"|没(?:有)?(?:具体|特别|专门|怎么|太)?(?:聊|说|提|讲)(?:起|到)?过"
)
DISCLOSURE_FEEDBACK = (
    "删除描述输入数据缺失的句子，只写活动推荐和普通出行建议。"
    "不说没聊过、没提过什么，也不评价开放时间是否固定；需要提醒就直接说出发前确认开放时间。"
)

# The assistant is an online companion: it recommends, it does not go along.
# Users going with their own friends ("约上朋友一起去") is fine and not matched,
# and neither are "我们去年…", "我们过去聊过", "那边见到的" or looking at the
# introduction together online ("一起看看").
CO_PRESENCE = re.compile(
    r"(?:我们|咱们|咱俩|我俩)(?:就|也|要|到时候?|一块儿?|一起){0,2}(?:去(?!年)|见个?面|现场|赴约|碰头|集合|出发)"
    r"|(?:我|咱)(?:也|就)?(?:陪你|带你|和你|跟你)(?:一起|一块儿?)?(?:去(?!年)|过去|逛|走|赴约)"
    r"|(?:要不要|不如|那就|干脆|到时候?)(?:我们|咱们)?一起(?:去(?!年)|过去|逛|凑)"
    r"|(?:现场|门口|到时候)见(?:面|啦|吧|哦|喔)?(?=[，。！？!?~～\s]|$)|不见不散"
    # "陪你去" without a subject is the assistant; "叫上朋友陪你去" is not.
    r"|(?<!朋友)(?<!家人)(?<!同事)(?<!室友)(?<!对象)(?<!伙伴)陪你(?:一起)?(?:去(?!年)|过去|逛)"
)
# A recommendation comes from seeing a listing. "刚路过解放路那边" invents a
# visit (production 10-17, a tea room); "要是刚路过" is the user's conditional.
AI_VISIT = re.compile(
    r"(?<![你您是果若要])(?:我|刚|刚刚)(?:才|也)?路过(?!的话)"
    r"|我(?:之前|上次|以前|也|刚)?(?:去过|到过|逛过|来过|喝过|吃过)(?!吗)"
)
# Chat-voice copy only (it has a repair round). With no verified user quote
# nothing can be said about the user's past or habits: production 10-17 wrote
# "之前看你关注过这场", "想着你之前挺爱听流行歌的" and "你平时做软件费神"
# with no evidence at all. "你之前要是听过" and "你最近要是有空" are conditional.
_USER_CITED = re.compile(
    r"你(?:之前|上次|以前|前两天|前几天|那天|昨天|最近)?(?:跟我|和我)?"
    r"(?:说|提|聊|讲|关注|收藏|问|分享)(?:起|到)?过"
    r"|你(?:之前|上次|以前|平时|平常|一直|向来|总是|经常|最近|前两天|前几天|那天|昨天)"
    r"(?!要是|如果|若|有空|方便|想|愿意)"
    r"|(?<!我)之前(?:你)?(?:提|说|聊)(?:到|起)?过"
)
# A place name says nothing about the district or transport ("反正就在市区，
# 过去也方便" passed review on 10-17 for a stadium).
_EASE_CLAIM = re.compile(
    r"市中心|就在市区|在市区里|交通(?:也|很|挺|还)?(?:方便|便利)|过去(?:也|很|挺|还)?方便"
    r"|位置(?:也|很|挺|还|比较)?(?:方便|便利|优越)"
)
_EASE_FACT = re.compile(r"市中心|市区|交通|地铁|公交")
# Category is not environment: a library or a temple is not "quiet" unless the
# facts say so (the reviewer passed "安静角落" and "挺清静" on 10-17).
_QUIET_CLAIM = re.compile(r"安静|清静|清幽|宁静|静谧|幽静|僻静")
# Crowds are a current state no provider fact covers ("不用挤人潮扎堆的路线"
# passed review on 10-18 for a riverside pier).
_CROWD_CLAIM = re.compile(r"人潮|人少|人不多|人很多|人挺多|人多|扎堆|人流|排队|拥挤|冷清")

# "下周三之前把票买好" is a deadline, not the session date.
_NOT_SESSION = r"(?!\s*(?:之前|以前|之后|以后|前(?!往)|后(?!天)))"
# All-day/anytime access and free admission are the claims the reviewer prompt
# lists first, and whether the facts state them at all is decidable. A
# rubber-stamp review must not be able to publish "免费全天营业".
# In recommendation copy a bare "全天" is an access claim ("全天免费", "全天都能去").
_ALL_DAY_CLAIM = re.compile(
    r"全天|24\s*小时|通宵(?:营业|开放)"
    r"|(?:随时|任何时候)(?:都)?(?:可以|能|可)?(?:拐|走|顺道|顺路)?(?:去|过去|进去|进来|来|进场|入园|逛)"
)
_ALL_DAY_FACT = re.compile(r"全天|24\s*小时|0?0:00\s*[-—~至]\s*24:00|通宵")
# Facts that negate access ("开放时段未提供，不能推断全天或随时营业", "非全天开放")
# must not count as stating it.
_NEGATED_ACCESS = re.compile(r"(?:不能|不要|不得|无法|不可)(?:推断|认为|写成|说)[^。；，,;]*|(?:非|不是|并非)全天")
_FREE_CLAIM = re.compile(r"免费|不收费|不要钱|零门票|无需门票|不用门票")
_FREE_FACT = re.compile(r"免费|不收费|免票|无需门票|(?<!\d)0(?:\.0+)?\s*元")

_MONTH_DAY = re.compile(r"(?<!\d)(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]" + _NOT_SESSION)
# "这周一定要去" and "下周一起去" are not Monday.
_RELATIVE_WEEK = re.compile(
    r"(这|本|下下|下)个?(?:周|星期|礼拜)([二三四五六日天末]|一(?![定起直些样般]))" + _NOT_SESSION)
_BARE_WEEKEND = re.compile(r"(?<![这本下个上])周末")
# "17号晚上" without a month. Street numbers ("伯先路12号", "团山路33号") and
# "2号线" / "3号馆" are not dates.
_DAY_ONLY = re.compile(
    r"(?<![月\d路街巷道弄里村镇区楼栋幢座室号])(\d{1,2})\s*[号日](?!\s*[线楼馆门口厅座房院店位])" + _NOT_SESSION)
_DAY_ONLY_WINDOW_DAYS = 45
# Longest alternatives first: "大后天" contains "后天", "明天晚上" contains "明天".
_RELATIVE_DAY = re.compile(r"(?:大后天|后天|今天晚上|今晚|今夜|明天晚上|明天|明晚|明儿)" + _NOT_SESSION)
_RELATIVE_DAY_OFFSET = {
    "今天晚上": 0, "今晚": 0, "今夜": 0,
    "明天晚上": 1, "明天": 1, "明晚": 1, "明儿": 1,
    "后天": 2, "大后天": 3,
}


def local_now() -> datetime:
    """Seam for tests; the NTP-corrected clock is the one the rest of chat uses."""
    from app.services.schedule_domain.time_service import get_current_time

    return get_current_time().now.astimezone(_TZ)


def today_text(now: datetime) -> str:
    return f"{now:%Y-%m-%d}（{_WEEKDAY_CN[now.weekday()]}）"


def _local(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        moment = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(_TZ)


def event_span(card: dict[str, Any]) -> tuple[date, date] | None:
    """Local dates of a dated session; fixed places have none."""
    metadata = card.get("discovery_metadata") or {}
    event = metadata.get("event") or {}
    start = _local(card.get("starts_at") or event.get("starts_at"))
    if start is None:
        return None
    end = _local(card.get("ends_at") or event.get("ends_at")) or start
    return start.date(), max(start.date(), end.date())


def _relative_day_name(day: date, today: date) -> str:
    delta = (day - today).days
    named = {0: "今天", 1: "明天", 2: "后天"}.get(delta)
    if named:
        return named
    return f"今天之后第 {delta} 天" if delta > 0 else f"{-delta} 天前"


def _week_name(day: date, today: date) -> str | None:
    weeks = ((day - timedelta(days=day.weekday())) - (today - timedelta(days=today.weekday()))).days // 7
    prefix = {0: "本周", 1: "下周"}.get(weeks)
    return prefix + _WEEKDAY_CN[day.weekday()][-1] if prefix else None


def date_relation(card: dict[str, Any], now: datetime) -> str:
    """Code computes weekdays and relative days; models get them wrong."""
    span = event_span(card)
    if span is None:
        return "固定地点，不对应某一天"
    today = now.date()

    def describe(day: date) -> str:
        parts = [_WEEKDAY_CN[day.weekday()], _relative_day_name(day, today)]
        week = _week_name(day, today)
        if week:
            parts.append(week)
        return f"{day:%Y-%m-%d}（{'，'.join(parts)}）"

    start, end = span
    return describe(start) if start == end else f"{describe(start)} 至 {describe(end)}"


def _future_month_day(month: int, day: int, today: date) -> date | None:
    try:
        candidate = date(today.year, month, day)
        if candidate >= today:
            return candidate
        candidate = date(today.year + 1, month, day)
    except ValueError:
        return None
    return candidate if (candidate - today).days <= _NEXT_YEAR_WINDOW_DAYS else None


def _future_day_of_month(day: int, today: date) -> date | None:
    """The next date with this day-of-month, if it is close enough to mean a session."""
    first = today.replace(day=1)
    for month_start in (first, (first + timedelta(days=32)).replace(day=1)):
        try:
            candidate = month_start.replace(day=day)
        except ValueError:
            continue
        if candidate >= today:
            return candidate if (candidate - today).days <= _DAY_ONLY_WINDOW_DAYS else None
    return None


def date_issues(text: str, card: dict[str, Any], now: datetime) -> list[str]:
    """Only claims that can be checked exactly against a dated session."""
    span = event_span(card)
    if span is None:
        return []
    start, end = span
    today = now.date()
    years = {start.year, end.year}
    issues: list[str] = []

    def outside(day: date) -> bool:
        return not start <= day <= end

    # Only "明年": "去年"/"今年" routinely describe the user's past, not the session.
    if "明年" in text and today.year + 1 not in years:
        issues.append(f"活动在{start.year}年，不能写成“明年”。")
    def in_span(month: int, day: int) -> bool:
        # A long run across New Year: "12月20日" in January still names its start.
        for year in years:
            try:
                if not outside(date(year, month, day)):
                    return True
            except ValueError:
                pass
        return False

    for month, day in _MONTH_DAY.findall(text):
        if in_span(int(month), int(day)):
            continue
        mentioned = _future_month_day(int(month), int(day), today)
        if mentioned and outside(mentioned):
            issues.append(f"“{month}月{day}日”与活动日期 {start:%Y-%m-%d} 不符。")
    span_days = (end - start).days
    if span_days <= 31:  # On a longer run every day-of-month is plausible.
        for day in dict.fromkeys(_DAY_ONLY.findall(text)):
            mentioned = _future_day_of_month(int(day), today)
            if mentioned and outside(mentioned):
                issues.append(f"“{day}号”是 {mentioned:%Y-%m-%d}，与活动日期 {start:%Y-%m-%d} 不符。")
    monday = today - timedelta(days=today.weekday())
    for prefix, weekday in _RELATIVE_WEEK.findall(text):
        week_start = monday + timedelta(weeks={"这": 0, "本": 0, "下": 1, "下下": 2}[prefix])
        targets = ([week_start + timedelta(days=5), week_start + timedelta(days=6)] if weekday == "末"
                   else [week_start + timedelta(days=_WEEKDAY_INDEX[weekday])])
        if all(outside(target) for target in targets):
            phrase = f"{prefix}周{weekday}"
            issues.append(f"“{phrase}”是 {targets[0]:%Y-%m-%d}，与活动日期 {start:%Y-%m-%d} 不符。")
    if (_BARE_WEEKEND.search(text) and span_days <= _WEEKDAY_CHECK_MAX_SPAN_DAYS
            and all((start + timedelta(days=i)).weekday() < 5 for i in range(span_days + 1))):
        issues.append(f"活动日期 {start:%Y-%m-%d} 不在周末，不能写成“周末”。")
    for word in dict.fromkeys(_RELATIVE_DAY.findall(text)):
        target = today + timedelta(days=_RELATIVE_DAY_OFFSET[word])
        if outside(target):
            issues.append(f"“{word}”是 {target:%Y-%m-%d}，与活动日期 {start:%Y-%m-%d} 不符。")
    return issues


def access_issues(text: str, facts_text: str) -> list[str]:
    issues: list[str] = []
    facts_text = _NEGATED_ACCESS.sub("", facts_text)
    claim = _ALL_DAY_CLAIM.search(text)
    if claim and not _ALL_DAY_FACT.search(facts_text):
        issues.append(f"资料没有写“{claim.group(0)}”，不能这么说；需要提醒就说出发前确认开放时间。")
    claim = _FREE_CLAIM.search(text)
    if claim and not _FREE_FACT.search(facts_text):
        issues.append(f"资料没有写“{claim.group(0)}”，不能写门票或价格情况。")
    return issues


def voice_issues(
    text: str, facts_text: str, user_references: int, *, labels: tuple[str, ...] = (),
) -> list[str]:
    issues: list[str] = []
    # Category names are search buckets; one was "安静角落". They never prove
    # what the place is like.
    for label in filter(None, labels):
        facts_text = facts_text.replace(label, "")
    cited = _USER_CITED.search(text) if not user_references else None
    if cited:
        issues.append(f"没有用户原文可依据，删除“{cited.group(0)}”开头这类说用户之前提过、平时怎样的话，"
                      "直接从这个地方本身说起；也不要改写成“你之前没提过”这类说法。")
    quiet = _QUIET_CLAIM.search(text)
    if quiet and not _QUIET_CLAIM.search(facts_text):
        issues.append(f"资料没有写环境“{quiet.group(0)}”，删除这类环境描述，不要从类别推断环境。")
    crowd = _CROWD_CLAIM.search(text)
    if crowd and not _CROWD_CLAIM.search(facts_text):
        issues.append(f"资料没有写人流情况，删除“{crowd.group(0)}”这类说法。")
    ease = _EASE_CLAIM.search(text)
    if ease and not _EASE_FACT.search(facts_text):
        issues.append(f"资料没有写“{ease.group(0)}”，地点名字不能说明位置或交通。")
    return issues


def copy_issues(
    text: str, card: dict[str, Any], now: datetime, *, facts_text: str | None = None,
    user_references: int | None = None,
) -> list[str]:
    """Revision feedback for one draft; empty means no deterministic objection.

    ``facts_text`` is everything the copy may rely on; with it, access claims
    (全天/随时/免费) are checked against what the facts actually state.
    ``user_references`` (verified quotes available) adds the environment,
    crowd, location and cited-user checks. Every field of copy gets them now:
    card copy drops only the offending field instead of the whole card.
    """
    issues: list[str] = []
    if INPUT_DISCLOSURE.search(text):
        issues.append(DISCLOSURE_FEEDBACK)
    together = CO_PRESENCE.search(text)
    if together:
        issues.append(
            f"删除“{together.group(0)}”这类暗示你会和用户一起去或线下见面的说法；"
            "推荐的是用户自己去，你只在线上陪伴。"
        )
    visit = AI_VISIT.search(text)
    if visit:
        issues.append(f"删除“{visit.group(0)}”：你没有去过或路过那里，只是看到了这个推荐。")
    if re.search(r"https?://|www\.", text):
        issues.append("删除网址。")
    issues.extend(date_issues(text, card, now))
    if facts_text is not None:
        issues.extend(access_issues(text, facts_text))
        if user_references is not None:
            issues.extend(voice_issues(text, facts_text, user_references,
                                       labels=(str(card.get("category") or ""),)))
    return issues


def _issue_text(item: Any) -> str:
    """Reviewers also answer with objects ({"claim": ..., "reason": ...}).

    Keeping only strings dropped them (production 10-18): the detail fell back
    with no recorded reason, so it was never repaired.
    """
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return ""
    def first(*keys: str) -> str:
        return next((item[k].strip() for k in keys if isinstance(item.get(k), str) and item[k].strip()), "")
    claim = first("claim", "text", "quote", "sentence")
    reason = first("reason", "problem", "missing", "evidence")
    return f"“{claim}”：{reason}" if claim and reason else claim or reason


def bounded_issues(items: Any, *, limit: int = 3, length: int = 120) -> list[str]:
    """Diagnostics for the admin view: short, few, never raw model payloads."""
    if not isinstance(items, list):
        return []
    return [text[:length] for text in map(_issue_text, items) if text][:limit]
