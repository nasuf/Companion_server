"""Agent 的固定作息: 起床、上下班、睡觉、休息日 (TIME12)。

为什么单独有一份: 人设档案里的作息 ("晚上9点下班") 建号时写进 L1 记忆, 每日作息
却由另一次生成随手排, 两边互不知道。2026-10-30 生产: 人设写 21:00 下班, 当天作息
排 19:30 下班回家, 两段同时进了主回复, 模型在 19:55 说"我都是九点下班的"又说
"提前溜了十分钟"。真人每天的上下班时间是稳定的, 所以这里定一份固定作息作为
唯一来源: 人设写明的项原样采用, 没写的由模型按职业补齐一次后存到 ai_agents.routine,
之后每日作息必须服从它 (见 schedule.generate_daily_schedule), 主回复也按它把
"平时几点下班 / 今天现在处于什么阶段" 算好再交给模型 (describe_now)。

时刻一律用"距当天 00:00 的分钟数"; 跨午夜的值 >= 1440 (凌晨 1 点下班 = 1500)。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from app.services.time_domain.parser import _parse_cn_number

logger = logging.getLogger(__name__)

ROUTINE_VERSION = 1
DAY = 24 * 60
FIELDS = ("wake", "work_start", "work_end", "sleep")
_WEEKDAY_CN = "一二三四五六日"

# ── 人设文字解析 ────────────────────────────────────────────────────────

_PERIOD = r"(凌晨|清晨|早上|早晨|上午|中午|下午|傍晚|晚上|每晚|夜里|半夜|深夜|早|晚)?"
_NUM = r"(\d{1,2}|[零一二两三四五六七八九十]{1,3})"
_CLOCK = (
    _PERIOD + r"\s*(?:(\d{1,2})[:：](\d{2})|" + _NUM
    + r"\s*[点时](?:\s*(半|一刻|三刻|(\d{1,2}|[一二三四五]?十[一二三四五六七八九]?|[一二三四五六七八九](?=\s*分))\s*分?))?)"
)
_CLOCK_RE = re.compile(_CLOCK)
# 时刻之后几个字内出现的动作决定它是哪一项。顺序即优先级 (中午的"休息"是午休, 单独排除)。
_ACTIONS = (
    ("work_end", re.compile(r"下班|收工|打烊|关店|闭店|关门|离店|结束工作|下课|放学|收摊|下播|交班")),
    ("work_start", re.compile(r"到店|到岗|上班|开门|开店|开工|开始工作|营业|出摊|上课|到公司|到工作室|"
                              r"开播|接班|到(?:健身房|医院|学校|单位|门店|店里|工地)|"
                              r"开始(?:直播|画|上课|营业|干活|接单|做|写|录|练|带|揉面|备料|备菜|上工)")),
    ("wake", re.compile(r"起床|(?<!提)醒|起来|^\s*(?:左右|多|才|就)?起(?!不来|营业|开始|上班|接单|开门|开工|到)")),
    ("sleep", re.compile(r"(?<![午小])睡|入眠|关灯|上床")),
)
_SLEEP_REST_RE = re.compile(r"休息")
_RANGE_RE = re.compile(
    _CLOCK + r"\s*(?:到|至|-|—|~)\s*" + _CLOCK
    + r"[^，。；]{0,4}?(上班|工作|营业|上课|值班|出摊|开工|做手工|做手作|创作|干活)"
)
_REST_WORD_RE = re.compile(r"休息|休|不上班|放假|轮休|调休|歇业|店休")
_REST_NEGATION_RE = re.compile(r"[不没]\s*(?:休息|休|放假)")
_WEEKDAY_TOKEN_RE = re.compile(r"(?:周|星期|礼拜)([一二三四五六日天](?:[、和及与]?[一二三四五六日天])*)")
# "播到晚上七点" "画到晚上十点" "卖到上午十点半": 做事做到几点 = 收工
_UNTIL_RE = re.compile(r"(?:播|画|干|卖|忙|做|写|练|教|营业|开|上班|工作)到" + _CLOCK)
_NINE_TO_FIVE_RE = re.compile(r"(?:朝|早)([一二三四五六七八九十]{1,2})晚([一二三四五六七八九十]{1,2})(?![点时])")
# "睡到下午两点" 是起床时刻; "Wake up 7:15"
_WAKE_AT_RE = re.compile(r"(?:睡到|wake up\s*)" + _CLOCK, re.IGNORECASE)
# 说的是某一次或不固定的事, 不是作息: "有一次通宵到凌晨四点" "偶尔十一点前睡"
_ONE_OFF_RE = re.compile(r"有一次|有次|那次|上次|那天|偶尔|有时候?|没有固定|不固定|想起就")
# 时间段本身没带"上班"之类的词, 但这句话在讲工作 ("在琴行教课，上午十点到下午五点")
_WORK_CONTEXT_RE = re.compile(r"上班|教课|上课|工作|营业|出摊|值班|开店|后厨|店里|公司|单位|工作室|直播|接单|干活|坐诊|"
                              r"[白夜早晚中]班")
# 后面跟着午休/吃饭的时间段是中间休息, 不是上班时间
_PLAIN_RANGE_RE = re.compile(_CLOCK + r"\s*(?:到|至|-|—|~)\s*" + _CLOCK
                             + r"(?!\s*(?:午休|休息|午睡|睡觉|吃饭|午饭|晚饭|健身|运动))")
_WEEKEND_REST_RE = re.compile(r"周末[^，。；]{0,3}(?:休息|休|不上班|放假|双休)|双休")
_PM_PERIODS = {"下午", "傍晚", "晚上", "每晚", "夜里", "半夜", "深夜", "晚"}
# 只有夜里的"休息"才是睡觉; 下午"休息半小时"是歇一会儿。
_NIGHT_PERIODS = {"晚上", "每晚", "夜里", "半夜", "深夜", "晚"}


def _routine_text(raw) -> str:
    """去掉讲某一次 / 不固定的分句, 剩下的才是作息。"""
    kept = []
    for clause in re.split(r"[，。；,;！!？?（）()]", str(raw or "").strip()):
        marker = _ONE_OFF_RE.search(clause)
        # "每天7点起床偶尔赖床": 标记之前的仍是作息
        kept.append(clause[:marker.start()] if marker else clause)
    return "，".join(c for c in kept if c)


def _clock_minute(groups: tuple, *, kind: str | None = None) -> int | None:
    period, hh, mm, hour_word, frac, minute_word = groups
    if hh is not None:
        hour, minute = int(hh), int(mm)
    else:
        hour = _parse_cn_number(hour_word or "")
        if hour is None:
            return None
        minute = {"半": 30, "一刻": 15, "三刻": 45}.get(frac or "", None)
        if minute is None:
            minute = (int(minute_word) if minute_word and minute_word.isdigit()
                      else (_parse_cn_number(minute_word) or 0) if minute_word else 0)
    if hour > 24 or minute >= 60:
        return None
    if period in ("下午", "傍晚") and hour < 12:
        hour += 12
    elif period in _NIGHT_PERIODS and 6 <= hour <= 12:
        hour += 12  # 晚上12点 = 24:00; 夜里 0-5 点留给 _normalize 记作次日
    elif period == "中午" and hour < 6:
        hour += 12
    elif not period and kind == "work_end" and 1 <= hour <= 11:
        hour += 12  # "6点下班" 指傍晚
    elif not period and kind == "sleep" and 7 <= hour <= 12:
        hour += 12
    return hour * 60 + minute


def _normalize(field_name: str, minute: int) -> int:
    """睡觉/下班落在清晨之前的, 视为当晚过了午夜。"""
    if field_name in ("sleep", "work_end") and minute < 6 * 60:
        return minute + DAY
    if minute >= DAY and field_name in ("wake", "work_start"):
        return minute - DAY
    return minute


_NOT_CLOCK_BEFORE = set("早晚多少快慢有差好")
_NOT_CLOCK_AFTER = ("一线", "也", "儿", "都", "点")


def _is_clock(line: str, match: re.Match) -> bool:
    """"早一点睡" "两点一线" "一点也不想起床" 里的"X点"不是钟点。"""
    if match.group(1) in ("早", "晚") and match.group(4) in ("一", "两", "1"):
        return False  # "早一点睡" "晚一点再说" 是早些/晚些
    if match.group(1) or match.group(2):  # 带时段词或 HH:MM
        return True
    before = line[match.start() - 1] if match.start() else ""
    after = line[match.end():match.end() + 2]
    if before in _NOT_CLOCK_BEFORE or after.startswith(_NOT_CLOCK_AFTER):
        return False
    return True


@dataclass
class ParsedRoutine:
    values: dict[str, int] = field(default_factory=dict)
    rest_weekdays: set[int] | None = None
    evidence: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    def _set(self, name: str, minute: int, line: str) -> None:
        minute = _normalize(name, minute)
        if name in self.values:
            if abs(self.values[name] - minute) > 30:
                self.conflicts.append(f"{name}:{fmt_clock(self.values[name])}≠{fmt_clock(minute)}")
            return
        self.values[name] = minute
        if line not in self.evidence:
            self.evidence.append(line[:80])


def parse_routine_lines(lines: list[str]) -> ParsedRoutine:
    """人设里的作息句 → 各项时刻。只认明确写出的时刻; 先出现的句子优先。"""
    parsed = ParsedRoutine()
    # 一天里分几段干活 ("上午9点到12点做手工，下午2点到6点做手工") 时取最早开始、最晚结束。
    spans: list[tuple[int, int, str]] = []
    for raw in lines:
        line = _routine_text(raw)
        idiom = _NINE_TO_FIVE_RE.search(line)
        if idiom:
            start_h, end_h = _parse_cn_number(idiom.group(1)), _parse_cn_number(idiom.group(2))
            if start_h and end_h:
                spans.append((start_h * 60, (end_h + (12 if end_h <= 12 else 0)) * 60, line))
        matches = list(_RANGE_RE.finditer(line))
        if not matches and _WORK_CONTEXT_RE.search(line):
            matches = list(_PLAIN_RANGE_RE.finditer(line))
        for match in matches:
            first, second = match.groups()[0:6], match.groups()[6:12]
            start = _clock_minute(first, kind="work_start")
            # "上午8点到11点" 的结束时刻沿用开始时刻的时段词, 不按"下班多在傍晚"去猜。
            end = _clock_minute(second if second[0] else (first[0],) + second[1:], kind="work_start")
            if start is None or end is None:
                continue
            while end <= start:
                end += 12 * 60
            spans.append((start, end, line))
    if spans:
        parsed._set("work_start", min(s for s, _, _ in spans), spans[0][2])
        parsed._set("work_end", max(e for _, e, _ in spans), spans[-1][2])
    for raw in lines:
        line = _routine_text(raw)
        if not line:
            continue
        for match in _WAKE_AT_RE.finditer(line):
            minute = _clock_minute(match.groups()[:6], kind="wake")
            if minute is not None and "wake" not in parsed.values:
                parsed._set("wake", minute, line)
        for match in _UNTIL_RE.finditer(line):
            minute = _clock_minute(match.groups()[:6], kind="work_end")
            if minute is not None and "work_end" not in parsed.values:
                parsed._set("work_end", minute, line)
        for match in _CLOCK_RE.finditer(line):
            if not _is_clock(line, match):
                continue
            tail = line[match.end():match.end() + 8]
            # 只看到下一个时刻为止, "7点起床，8点到店" 里的"到店"不属于 7 点。
            following = _CLOCK_RE.search(tail)
            if following:
                tail = tail[:following.start()]
            tail = re.split(r"[，。；,;]", tail, maxsplit=1)[0]
            # 同一时刻后面跟着几个动作 ("3点起床到店") 时取最先出现的那个。
            hits = [(m.start(), name) for name, pattern in _ACTIONS if (m := pattern.search(tail))]
            if hits:
                name = min(hits)[1]
                minute = _clock_minute(match.groups()[:6], kind=name)
                if minute is not None:
                    parsed._set(name, minute, line)
            else:
                period = match.group(1) or ""
                if _SLEEP_REST_RE.search(tail) and period in _NIGHT_PERIODS:
                    minute = _clock_minute(match.groups()[:6], kind="sleep")
                    if minute is not None:
                        parsed._set("sleep", minute, line)
        rest: set[int] = set()
        for clause in re.split(r"[，。；,;！!？?]", line):
            if not _REST_WORD_RE.search(clause) or _REST_NEGATION_RE.search(clause):
                continue
            for group in _WEEKDAY_TOKEN_RE.findall(clause):
                rest |= {_WEEKDAY_CN.index("日" if ch == "天" else ch) for ch in group if ch in _WEEKDAY_CN or ch == "天"}
            if _WEEKEND_REST_RE.search(clause):
                rest |= {5, 6}
        if rest and parsed.rest_weekdays is None:
            parsed.rest_weekdays = rest
            if line not in parsed.evidence:
                parsed.evidence.append(line[:80])
    return parsed


# ── 固定作息 ─────────────────────────────────────────────────────────

def fmt_clock(minute: int | None) -> str:
    if minute is None:
        return "?"
    prefix = "次日 " if minute >= DAY else ""
    minute %= DAY
    return f"{prefix}{minute // 60:02d}:{minute % 60:02d}"


def _plain_clock(minute: int) -> str:
    minute %= DAY
    return f"{minute // 60}:{minute % 60:02d}"


def parse_clock_text(value: Any, field_name: str) -> int | None:
    match = re.fullmatch(r"\s*(?:次日\s*)?(\d{1,2})[:：](\d{2})\s*", str(value or ""))
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour > 24 or minute >= 60:
        return None
    total = hour * 60 + minute
    if str(value).strip().startswith("次日"):
        total += DAY
    return _normalize(field_name, total)


@dataclass(frozen=True)
class AgentRoutine:
    wake: int
    work_start: int | None
    work_end: int | None
    sleep: int
    rest_weekdays: frozenset[int]
    holiday_off: bool
    sources: dict[str, str]
    evidence: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()

    @property
    def has_work(self) -> bool:
        return self.work_start is not None and self.work_end is not None

    def to_json(self, *, derived_at: str) -> dict:
        return {
            "version": ROUTINE_VERSION,
            "wake": fmt_clock(self.wake), "sleep": fmt_clock(self.sleep),
            "work_start": fmt_clock(self.work_start) if self.work_start is not None else None,
            "work_end": fmt_clock(self.work_end) if self.work_end is not None else None,
            "rest_weekdays": sorted(d + 1 for d in self.rest_weekdays),
            "holiday_off": self.holiday_off,
            "sources": dict(self.sources),
            "evidence": list(self.evidence),
            "conflicts": list(self.conflicts),
            "derived_at": derived_at,
        }

    @classmethod
    def from_json(cls, data: Any) -> "AgentRoutine | None":
        if not isinstance(data, dict) or data.get("version") != ROUTINE_VERSION:
            return None
        wake = parse_clock_text(data.get("wake"), "wake")
        sleep = parse_clock_text(data.get("sleep"), "sleep")
        start = parse_clock_text(data.get("work_start"), "work_start") if data.get("work_start") else None
        end = parse_clock_text(data.get("work_end"), "work_end") if data.get("work_end") else None
        rest = data.get("rest_weekdays")
        if wake is None or sleep is None or not isinstance(rest, list):
            return None
        if (start is None) != (end is None):
            return None
        if start is not None and end <= start:
            end += DAY
        try:
            rest_set = frozenset(int(d) - 1 for d in rest if 1 <= int(d) <= 7)
        except (TypeError, ValueError):
            return None
        sources = data.get("sources") if isinstance(data.get("sources"), dict) else {}
        return cls(wake, start, end, sleep, rest_set, bool(data.get("holiday_off")),
                   {str(k): str(v) for k, v in sources.items()},
                   tuple(str(e) for e in data.get("evidence") or [])[:8],
                   tuple(str(c) for c in data.get("conflicts") or [])[:8])

    def usual_text(self) -> str:
        parts = [f"早上{_plain_clock(self.wake)}左右起床"]
        if self.has_work:
            parts.append(f"{_plain_clock(self.work_start)}开始上班，{_day_word(self.work_end)}{_plain_clock(self.work_end)}下班")
        parts.append(f"{_day_word(self.sleep)}{_plain_clock(self.sleep)}左右睡觉")
        text = "，".join(parts)
        if self.has_work:
            if self.rest_weekdays:
                text += "；" + "、".join(f"周{_WEEKDAY_CN[d]}" for d in sorted(self.rest_weekdays)) + "休息"
            else:
                text += "；没有固定休息日"
            if self.holiday_off:
                text += "，法定节假日放假"
        return text


def _day_word(minute: int) -> str:
    """只给过了午夜的时刻加"凌晨", 其余直接写 24 小时制时刻。"""
    return "凌晨" if minute >= DAY else ""


def validate_routine_values(values: dict[str, int | None], rest: set[int]) -> str | None:
    """补全结果的合理性; 不合理返回原因。"""
    wake, sleep = values.get("wake"), values.get("sleep")
    start, end = values.get("work_start"), values.get("work_end")
    if wake is None or sleep is None:
        return "missing_wake_or_sleep"
    if not (3 * 60 <= wake <= 14 * 60):
        return "wake_out_of_range"
    if not (19 * 60 <= sleep <= DAY + 6 * 60) or sleep - wake < 12 * 60:
        return "sleep_out_of_range"
    if (start is None) != (end is None):
        return "half_work_window"
    if start is not None:
        if end <= start:
            return "work_end_before_start"
        if end - start > 16 * 60 or end - start < 2 * 60:
            return "work_length_out_of_range"
        if start < wake:
            return "work_before_wake"
        if sleep <= end:
            return "sleep_before_work_end"
    if len(rest) > 4:
        return "too_many_rest_days"
    return None


# ── 某一天 ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DayRoutine:
    day: date
    is_workday: bool
    reason: str  # workday | rest_weekday | holiday | makeup_workday | no_work
    wake: int
    sleep: int
    work_start: int | None
    work_end: int | None


def day_plan(routine: AgentRoutine, day: date, *, holiday: Any = None,
             workday_swap: bool = False) -> DayRoutine:
    legal = holiday is not None and getattr(holiday, "type", None) == "legal"
    if not routine.has_work:
        is_work, reason = False, "no_work"
    elif routine.holiday_off and workday_swap:
        is_work, reason = True, "makeup_workday"
    elif routine.holiday_off and legal:
        is_work, reason = False, "holiday"
    elif day.weekday() in routine.rest_weekdays:
        is_work, reason = False, "rest_weekday"
    else:
        is_work, reason = True, "workday"
    wake, sleep = routine.wake, routine.sleep
    if reason in ("rest_weekday", "holiday"):
        # 休息日多睡一会儿, 但不改动固定的上下班时间。
        wake = min(wake + 60, 11 * 60)
    return DayRoutine(day, is_work, reason, wake, sleep,
                      routine.work_start if is_work else None,
                      routine.work_end if is_work else None)


_REASON_TEXT = {
    "rest_weekday": "今天是你的固定休息日",
    "holiday": "今天法定节假日放假",
    "no_work": "你没有固定上班时间",
}


def render_constraint_rules(day: DayRoutine, routine: AgentRoutine) -> str:
    lines = []
    if day.is_workday:
        lines.append(
            f"- 今天上班：{fmt_clock(day.work_start)} 开始工作，{fmt_clock(day.work_end)} 结束工作离开；"
            "上班时段（含开工准备、收尾）完整落在这两个时刻之间，之外不安排上班，也不提前下班或加班。"
        )
    else:
        lines.append(f"- {_REASON_TEXT.get(day.reason, '今天休息')}，全天不安排上班。")
    lines.append(f"- 约 {fmt_clock(day.wake)} 起床，约 {fmt_clock(day.sleep)} 睡觉（误差半小时内）。")
    lines.append(f"- 你平时的作息：{routine.usual_text()}。")
    return "\n".join(lines)


# ── 校验与兜底 ──────────────────────────────────────────────────────

KIND_WORK = "工作"
KIND_SLEEP = "睡眠"
KINDS = ("睡眠", "起床", "通勤", "工作", "用餐", "休闲", "家务", "社交", "其他")
_WORK_TOLERANCE = 30
_EDGE_TOLERANCE = 60
_REST_DAY_WORK_ALLOWANCE = 120


def _spans(schedule: list[dict]) -> list[tuple[int, int, dict]]:
    from app.services.schedule_domain.schedule import _slot_span
    out = []
    for slot in schedule:
        span = _slot_span(slot)
        if span:
            out.append((span[0], span[1], slot))
    return sorted(out, key=lambda item: item[0])


def has_kinds(schedule: list[dict] | None) -> bool:
    return bool(schedule) and all(isinstance(s, dict) and s.get("kind") in KINDS for s in schedule)


def work_window(schedule: list[dict]) -> tuple[int, int] | None:
    work = [(s, e) for s, e, slot in _spans(schedule) if slot.get("kind") == KIND_WORK]
    if not work:
        return None
    return min(s for s, _ in work), max(e for _, e in work)


def check_conformance(schedule: list[dict], day: DayRoutine) -> list[str]:
    """每日作息与固定作息的偏差; 空列表表示一致。"""
    if not has_kinds(schedule):
        return ["missing_kind"]
    spans = _spans(schedule)
    if not spans:
        return ["no_valid_slots"]
    violations = []
    window = work_window(schedule)
    if day.is_workday:
        if window is None:
            violations.append("no_work_on_workday")
        else:
            start, end = window
            if abs(start - day.work_start) > _WORK_TOLERANCE:
                violations.append(f"work_start {fmt_clock(start)}≠{fmt_clock(day.work_start)}")
            if abs(end - day.work_end) > _WORK_TOLERANCE:
                violations.append(f"work_end {fmt_clock(end)}≠{fmt_clock(day.work_end)}")
    elif day.reason != "no_work" and window is not None:
        # 休息日顺手处理点工作上的事很正常, 只拦整段上班。
        worked = sum(e - s for s, e, slot in spans if slot.get("kind") == KIND_WORK)
        if worked > _REST_DAY_WORK_ALLOWANCE:
            violations.append("work_on_rest_day")
    awake = [(s, e) for s, e, slot in spans if slot.get("kind") != KIND_SLEEP]
    if awake:
        first = min(s for s, _ in awake)
        if abs(first - day.wake) > _EDGE_TOLERANCE:
            violations.append(f"wake {fmt_clock(first)}≠{fmt_clock(day.wake)}")
        sleeps = [s for s, _, slot in spans if slot.get("kind") == KIND_SLEEP and s >= 12 * 60]
        bedtime = min(sleeps) if sleeps else max(e for _, e in awake)
        if abs(bedtime - day.sleep) > _EDGE_TOLERANCE:
            violations.append(f"sleep {fmt_clock(bedtime)}≠{fmt_clock(day.sleep)}")
    return violations


def _slot(start: int, end: int, event: str, status: str, kind: str) -> dict:
    return {"start": fmt_clock(start), "end": fmt_clock(end), "event": event,
            "status": status, "kind": kind}


def template_schedule(day: DayRoutine) -> list[dict]:
    """生成失败时按固定作息排一天; 只用通用说法, 不编地点和具体事件。"""
    out: list[dict] = []
    cursor = day.wake
    out.append(_slot(cursor, cursor + 30, "起床洗漱", "忙碌", "起床"))
    cursor += 30
    if day.is_workday:
        start, end = day.work_start, day.work_end
        if start - cursor >= 30:
            breakfast_end = min(cursor + 40, start - 20)
            out.append(_slot(cursor, breakfast_end, "吃早饭", "空闲", "用餐"))
            if start - 20 > breakfast_end:
                out.append(_slot(breakfast_end, start - 20, "自由时间", "空闲", "休闲"))
            out.append(_slot(start - 20, start, "出门去上班", "空闲", "通勤"))
        elif start > cursor:
            out.append(_slot(cursor, start, "出门去上班", "空闲", "通勤"))
        noon_start, noon_end = 12 * 60, 13 * 60
        if start < noon_start - 60 and end > noon_end + 60:
            out.append(_slot(start, noon_start, "上班", "很忙碌", "工作"))
            out.append(_slot(noon_start, noon_end, "午饭休息", "空闲", "用餐"))
            out.append(_slot(noon_end, end, "上班", "忙碌", "工作"))
        else:
            out.append(_slot(start, end, "上班", "很忙碌", "工作"))
        cursor = end
        if day.sleep - cursor > 40:
            out.append(_slot(cursor, cursor + 40, "下班回家", "空闲", "通勤"))
            cursor += 40
    else:
        if 12 * 60 - cursor > 60:
            out.append(_slot(cursor, 12 * 60, "在家慢慢收拾、做点自己的事", "空闲", "休闲"))
            cursor = 12 * 60
    if day.sleep - cursor > 90:
        meal_start = max(cursor, 18 * 60) if day.sleep - max(cursor, 18 * 60) > 90 else cursor
        if meal_start > cursor:
            out.append(_slot(cursor, meal_start, "自由时间", "空闲", "休闲"))
        out.append(_slot(meal_start, meal_start + 60, "吃饭", "空闲", "用餐"))
        cursor = meal_start + 60
    if day.sleep - 30 > cursor:
        out.append(_slot(cursor, day.sleep - 30, "自由时间", "空闲", "休闲"))
        cursor = day.sleep - 30
    if day.sleep > cursor:
        out.append(_slot(cursor, day.sleep, "洗漱准备睡觉", "空闲", "家务"))
    next_wake = day.wake + DAY if day.wake + DAY > day.sleep else day.sleep + 6 * 60
    out.append(_slot(day.sleep, next_wake, "睡觉", "睡眠", "睡眠"))
    return out


# ── 主回复里的"现在" ──────────────────────────────────────────────────

def _slot_kind_at(schedule: list[dict] | None, minute: int) -> str | None:
    if not schedule or not has_kinds(schedule):
        return None
    hits = [(e - s, slot.get("kind")) for s, e, slot in _spans(schedule) if s <= minute < e]
    return min(hits)[1] if hits else None


def _duration_text(minutes: int) -> str:
    minutes = max(0, int(minutes))
    hours, rest = divmod(minutes, 60)
    if hours and rest:
        return f"{hours}小时{rest}分钟"
    if hours:
        return f"{hours}小时"
    return f"{rest}分钟"


def describe_now(routine: AgentRoutine, schedule: list[dict] | None, now: datetime,
                 *, holiday: Any = None, workday_swap: bool = False,
                 yesterday_holiday: Any = None, yesterday_swap: bool = False) -> dict:
    """平时 / 今天 / 此刻三句, 以及给 Trace 的结构化字段。时刻差都在这里算好。"""
    from datetime import timedelta

    day = day_plan(routine, now.date(), holiday=holiday, workday_swap=workday_swap)
    minute = now.hour * 60 + now.minute
    window_source = "routine"
    window = (day.work_start, day.work_end) if day.is_workday else None
    # 上不上班按固定作息定; 当天作息只用来细化上班时段 (含被用户调整过的那一截)。
    if window and schedule and has_kinds(schedule):
        window = work_window(schedule) or window
        window_source = "schedule"
    weekday = "一二三四五六日"[now.weekday()]
    # 凌晨还在昨晚那个跨午夜的班上 (按昨天上不上班算, 不按今天)。
    prev = day_plan(routine, now.date() - timedelta(days=1),
                    holiday=yesterday_holiday, workday_swap=yesterday_swap)
    overnight_end = prev.work_end - DAY if prev.is_workday and prev.work_end and prev.work_end > DAY else None
    if window:
        start, end = window
        today = f"今天周{weekday}，上班 {_plain_clock(start)}–{_day_word(end)}{_plain_clock(end)}"
    else:
        today = f"今天周{weekday}，{_REASON_TEXT.get(day.reason, '今天休息')}，不上班"
    if overnight_end is not None and minute < overnight_end:
        phase, now_line = "at_work", f"还在上昨晚的班，离下班还有{_duration_text(overnight_end - minute)}"
    elif window:
        start, end = window
        if minute < start:
            phase, now_line = "before_work", f"还没上班，再过{_duration_text(start - minute)}上班"
        elif minute < end and _slot_kind_at(schedule, minute) not in (None, KIND_WORK):
            # 上班时段中间排了别的 (午饭 / 用户要的陪聊调整)
            phase, now_line = "work_break", f"上班中间空下来一会儿，{_plain_clock(end)}下班"
        elif minute < end:
            phase, now_line = "at_work", f"正在上班，离下班还有{_duration_text(end - minute)}"
        else:
            phase, now_line = "after_work", f"已经下班{_duration_text(minute - end)}了"
    else:
        phase, now_line = "day_off", "今天不上班"
    clock = f"{now.hour}:{now.minute:02d}"
    return {
        "usual": routine.usual_text(),
        "today": today,
        "now_line": f"现在{clock}，{now_line}",
        "phase": phase,
        "day_reason": day.reason,
        "work_window": [fmt_clock(window[0]), fmt_clock(window[1])] if window else None,
        "work_window_source": window_source if window else None,
    }


def public_routine(routine: "AgentRoutine | None") -> dict | None:
    """给接口展示的固定作息: 时刻、休息日、各项来源和此前发现的冲突, 不含人设原句。"""
    if routine is None:
        return None
    data = routine.to_json(derived_at="")
    data.pop("evidence", None)
    data.pop("derived_at", None)
    data["usual"] = routine.usual_text()
    return data


def routine_json_dumps(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


# ── 读取与补全 ──────────────────────────────────────────────────────

_ROUTINE_CUE_RE = re.compile(r"起床|醒|下班|上班|到店|开门|打烊|收工|睡|休息|营业|上课|放学|值班|出摊")


async def _persona_lines(agent_id: str) -> list[str]:
    """人设里写明作息的句子: 建号时冻结的人设档案优先, 其次人设种子记忆。"""
    from app.db import db

    lines: list[str] = []
    try:
        rows = await db.query_raw(
            """SELECT payload_text FROM memory_profile_origins
               WHERE agent_id = $1 ORDER BY created_at DESC LIMIT 1""",
            agent_id,
        )
        if rows:
            payload = json.loads(rows[0]["payload_text"])
            lifestyle = ((payload.get("profile") or {}).get("lifestyle") or {})
            routine = lifestyle.get("routine")
            if isinstance(routine, str):
                routine = [routine]
            lines += [str(x) for x in routine or [] if isinstance(x, str)]
    except Exception as exc:
        logger.warning("[ROUTINE] profile origin unavailable agent=%s: %s", agent_id[:8], type(exc).__name__)
    try:
        rows = await db.query_raw(
            """SELECT m.content FROM memories_ai m
               JOIN chat_workspaces w ON w.id = m.workspace_id
               WHERE w.agent_id = $1 AND m.provenance = 'profile_seed' AND m.is_archived = false
                 AND m.sub_category IN ('生活习惯', '工作')
               ORDER BY m.importance DESC, m.created_at ASC LIMIT 40""",
            agent_id,
        )
        lines += [r["content"] for r in rows if _ROUTINE_CUE_RE.search(r.get("content") or "")]
    except Exception as exc:
        logger.warning("[ROUTINE] persona memories unavailable agent=%s: %s", agent_id[:8], type(exc).__name__)
    seen: set[str] = set()
    return [x for x in lines if not (x in seen or seen.add(x))]


def _known_text(parsed: ParsedRoutine) -> str:
    names = {"wake": "起床", "work_start": "上班", "work_end": "下班", "sleep": "睡觉"}
    out = [f"- {names[k]}：{fmt_clock(v)}" for k, v in parsed.values.items()]
    if parsed.rest_weekdays is not None:
        out.append("- 休息日：" + ("、".join(f"周{_WEEKDAY_CN[d]}" for d in sorted(parsed.rest_weekdays)) or "无"))
    return "\n".join(out) or "（人设没有写明具体时间）"


async def derive_routine(agent: Any) -> AgentRoutine | None:
    """人设写明的项原样采用; 缺的由模型按职业补一次。失败返回 None (不落库, 下次重试)。"""
    from app.services.llm.models import get_utility_model, invoke_json
    from app.services.prompting.store import get_prompt_text

    lines = await _persona_lines(agent.id)
    parsed = parse_routine_lines(lines)
    values: dict[str, int | None] = dict(parsed.values)
    rest = set(parsed.rest_weekdays) if parsed.rest_weekdays is not None else None
    sources = {k: "persona" for k in parsed.values}
    if rest is not None:
        sources["rest_weekdays"] = "persona"
    holiday_off = False
    missing = [k for k in FIELDS if k not in values] + ([] if rest is not None else ["rest_weekdays"])
    if missing:
        template = await get_prompt_text("schedule.routine_derive")
        prompt = template.format(
            occupation=getattr(agent, "occupation", None) or "未知",
            age=getattr(agent, "age", None) or "未知",
            city=getattr(agent, "city", None) or "未知",
            known=_known_text(parsed),
            persona_lines="\n".join(f"- {x}" for x in lines[:12]) or "（无）",
            overview=(getattr(agent, "lifeOverview", None) or "（无）")[:400],
        )
        result = await invoke_json(get_utility_model(), prompt)
        if not isinstance(result, dict):
            raise ValueError("routine_derive returned non-object")
        no_fixed_work = result.get("no_fixed_work") is True
        for name in FIELDS:
            if name in values:
                continue
            if name in ("work_start", "work_end") and no_fixed_work:
                values[name] = None
                sources[name] = "derived"
                continue
            minute = parse_clock_text(result.get(name), name)
            if minute is None:
                raise ValueError(f"routine_derive invalid {name}")
            values[name] = minute
            sources[name] = "derived"
        if rest is None:
            raw = result.get("rest_weekdays")
            if not isinstance(raw, list):
                raise ValueError("routine_derive invalid rest_weekdays")
            rest = {int(d) - 1 for d in raw if isinstance(d, int) and 1 <= d <= 7}
            sources["rest_weekdays"] = "derived"
        holiday_off = result.get("holiday_off") is True
        sources["holiday_off"] = "derived"
    start, end = values.get("work_start"), values.get("work_end")
    if start is not None and end is not None and end <= start:
        values["work_end"] = end + DAY
    reason = validate_routine_values(values, rest or set())
    if reason:
        raise ValueError(f"routine invalid: {reason}")
    return AgentRoutine(values["wake"], values.get("work_start"), values.get("work_end"), values["sleep"],
                        frozenset(rest or set()), holiday_off, sources,
                        tuple(parsed.evidence[:8]), tuple(parsed.conflicts[:8]))


async def get_agent_routine(agent_id: str) -> AgentRoutine | None:
    """已存档的固定作息 (只读, 不触发补全)。"""
    from app.db import db

    try:
        rows = await db.query_raw("SELECT routine FROM ai_agents WHERE id = $1", agent_id)
    except Exception as exc:
        logger.warning("[ROUTINE] read failed agent=%s: %s", agent_id[:8], type(exc).__name__)
        return None
    if not rows:
        return None
    data = rows[0].get("routine")
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError:
            return None
    return AgentRoutine.from_json(data)


async def ensure_agent_routine(agent: Any) -> AgentRoutine | None:
    """读存档; 没有就推导并写入 (只在仍为空时写, 并发推导以先写入者为准)。"""
    from datetime import UTC
    from app.db import db

    existing = await get_agent_routine(agent.id)
    if existing is not None:
        return existing
    try:
        routine = await derive_routine(agent)
    except Exception as exc:
        logger.warning("[ROUTINE] derive failed agent=%s: %s", str(agent.id)[:8], exc)
        return None
    if routine is None:
        return None
    payload = routine_json_dumps(routine.to_json(derived_at=datetime.now(UTC).isoformat()))
    try:
        await db.execute_raw(
            "UPDATE ai_agents SET routine = $1::jsonb WHERE id = $2 AND (routine IS NULL OR routine = 'null'::jsonb)",
            payload, agent.id,
        )
    except Exception as exc:
        logger.warning("[ROUTINE] persist failed agent=%s: %s", str(agent.id)[:8], type(exc).__name__)
        return routine
    return await get_agent_routine(agent.id) or routine
