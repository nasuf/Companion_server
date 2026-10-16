"""AI作息系统。

管理AI的生活画像、每日作息表、状态查询和作息调整。
生活画像由LLM生成，作息表基于模板+个性化。
"""

from __future__ import annotations

import json
import hashlib
import logging
import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from prisma import Json

from app.config import settings
from app.db import db
from app.redis_client import get_redis
from app.services.llm.models import get_utility_model, invoke_json, invoke_text
from app.services.prompting.store import get_prompt_text
from app.services.schedule_domain.time_service import (
    classify_day_kind,
    is_holiday,
    is_workday_swap,
)
from app.services.schedule_domain.time_parser import (
    ParsedTime,
    has_explicit_time,
    parse_time_expressions,
)
from app.services.mbti import get_mbti, signal as mbti_signal

_WEEKDAY_CN = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def _mbti_brief(mbti: dict | None) -> str:
    """Spec §2.1 prompt 参考信息 "性格"：输出 MBTI 类型 + 简要描述."""
    if not isinstance(mbti, dict) or not mbti:
        return "温和友善"
    mbti_type = mbti.get("type") or ""
    summary = mbti.get("summary") or ""
    brief = f"{mbti_type} {summary}".strip()
    return brief or "温和友善"


logger = logging.getLogger(__name__)

_SCHEDULE_TZ = ZoneInfo(settings.schedule_timezone)


def _local_now() -> datetime:
    """Return current time in the configured schedule timezone."""
    return datetime.now(_SCHEDULE_TZ)

# 作息时段模板（基准，根据人格微调）
_BASE_SCHEDULE_TEMPLATE = [
    {"start": "07:00", "end": "08:00", "activity": "起床洗漱", "type": "routine"},
    {"start": "08:00", "end": "09:00", "activity": "吃早餐", "type": "routine"},
    {"start": "09:00", "end": "12:00", "activity": "工作/学习", "type": "work"},
    {"start": "12:00", "end": "13:00", "activity": "午饭", "type": "routine"},
    {"start": "13:00", "end": "14:00", "activity": "午休", "type": "rest"},
    {"start": "14:00", "end": "18:00", "activity": "工作/学习", "type": "work"},
    {"start": "18:00", "end": "19:00", "activity": "晚饭", "type": "routine"},
    {"start": "19:00", "end": "21:00", "activity": "自由时间", "type": "leisure"},
    {"start": "21:00", "end": "22:00", "activity": "看剧/看书", "type": "leisure"},
    {"start": "22:00", "end": "23:00", "activity": "准备睡觉", "type": "routine"},
    {"start": "23:00", "end": "07:00", "activity": "睡觉", "type": "sleep"},
]


async def generate_life_overview(
    mbti: dict | None,
    age: int = 22,
    occupation: str | None = None,
    gender: str | None = None,
) -> str:
    """生成AI角色的生活画像（spec 第一部分 §2.1 + 指令模版 P20）。

    Spec 要求输出 200 字以内纯文本。7 个中文性格维度由 MBTI 4 维反推近似值：
      liveliness = E(0-100)         rationality = T
      imagination = N              sensitivity = 100 - T
      planning = J                 spontaneity = 100 - J
      humor = (E+N)/2（复合）

    Returns: 纯文本生活画像。
    """
    e = round(mbti_signal(mbti, "E") * 100)
    n = round(mbti_signal(mbti, "N") * 100)
    t = round(mbti_signal(mbti, "T") * 100)
    j = round(mbti_signal(mbti, "J") * 100)

    prompt = (await get_prompt_text("schedule.life_overview")).format(
        age=age,
        gender=gender or "未设定",
        occupation=occupation or "自由职业",
        liveliness=e,
        rationality=t,
        sensitivity=100 - t,
        planning=j,
        spontaneity=100 - j,
        imagination=n,
        humor=round((e + n) / 2),
    )

    model = get_utility_model()
    text = (await invoke_text(model, prompt)).strip()

    if not text:
        raise ValueError("Life overview generation returned empty text")
    return text


async def generate_and_save_life_overview(agent: Any) -> str:
    """从 agent 对象提取信息，生成并保存生活画像。返回文本描述。"""
    description = await generate_life_overview(
        get_mbti(agent),
        age=agent.age or 22,
        occupation=agent.occupation,
        gender=getattr(agent, "gender", None),
    )
    await save_life_overview(agent.id, description)
    return description


async def generate_daily_schedule(
    agent_id: str,
    name: str,
    mbti: dict | None,
    life_overview: str | None = None,
    date: datetime | None = None,
    user_id: str | None = None,
    *,
    age: int | None = None,
    occupation: str | None = None,
    city: str | None = None,
) -> list[dict]:
    """生成每日作息表。基于生活画像+模板+个性化。节日强制LLM路径。

    Spec Part 1 §2.1 要求 prompt 注入:
        姓名 / 年龄 / 职业 / 性格(MBTI) / 生活画像 / 日期属性 [+ 用户记忆]
    age/occupation 如果调用方已知可直接传; 缺时从 DB 懒查 agent 记录.
    """
    # 绑 ContextVar 让作息生成 LLM 用该 agent 的模型 override.
    from app.services.runtime_config import bind_agent_context
    await bind_agent_context(agent_id)

    date = date or _local_now()
    weekday = _WEEKDAY_CN[date.weekday()]

    # 检测节日 + 当日属性
    holiday = is_holiday(date.date())
    workday_swap = is_workday_swap(date.date())
    day_kind = classify_day_kind(date.date(), holiday, workday_swap=workday_swap)

    # Always load the authoritative profile. Caller hints may be stale and may
    # never override the city used to ground generated scenes.
    agent_row = None
    try:
        agent_row = await db.aiagent.find_unique(where={"id": agent_id})
        if agent_row is not None:
            age = getattr(agent_row, "age", None)
            occupation = getattr(agent_row, "occupation", None)
            city = getattr(agent_row, "city", None)
        else:
            city = None
    except Exception as e:
        city = None
        logger.debug("Agent profile lookup failed: %s", type(e).__name__)

    personality_brief = _mbti_brief(mbti)

    base_fmt = {
        "name": name,
        "age": age if age is not None else "未知",
        "occupation": occupation or "普通人",
        "city": city or "未知（只安排通用活动，不编造地名）",
        "personality_brief": personality_brief,
        "overview": life_overview or "",
        "date": date.strftime("%Y-%m-%d"),
        "weekday": weekday,
        "day_kind": day_kind,
    }

    if life_overview:
        try:
            # Spec Part 1 §2.1: 70% 概率使用不带用户记忆指令；30% 使用带用户记忆指令。
            use_memory_variant = bool(user_id) and random.random() < 0.30
            memory_summary = ""
            if use_memory_variant:
                from app.services.workspace.workspaces import get_active_workspace
                workspace = await get_active_workspace(user_id=user_id, agent_id=agent_id)
                memory_summary = await _get_user_memory_summary(
                    user_id, workspace_id=workspace.id if workspace else None,
                )
                if not memory_summary:
                    use_memory_variant = False

            if use_memory_variant:
                prompt_key = "schedule.daily_schedule_with_memory"
                prompt = (await get_prompt_text(prompt_key)).format(
                    **base_fmt,
                    user_memories="\n".join(
                        f"- {m}" for m in memory_summary.split("；") if m.strip()
                    ),
                )
            else:
                prompt_key = "schedule.daily_schedule"
                prompt = (await get_prompt_text(prompt_key)).format(**base_fmt)

            model = get_utility_model()
            schedule = await invoke_json(model, prompt)
            if isinstance(schedule, list) and len(schedule) >= 5:
                if not all(isinstance(slot, dict) for slot in schedule):
                    raise ValueError("Schedule entries must be objects")
                if agent_row is None:
                    from types import SimpleNamespace
                    agent_row = SimpleNamespace(name=name, city=city, age=age, occupation=occupation)
                schedule = await _ground_schedule(agent_row, schedule)
                await _cache_schedule(agent_id, date, schedule)
                logger.info(f"[SCHEDULE] {agent_id} {date.date()} gen via {prompt_key}")
                return schedule
        except Exception as e:
            logger.warning(f"LLM schedule generation failed, falling back to template: {e}")

    # 兜底: 无 life_overview 或 LLM 失败时用模板
    schedule = _personalize_template(mbti, date, holiday=holiday, workday_swap=workday_swap)
    await _cache_schedule(agent_id, date, schedule)
    return schedule


def _personalize_template(
    mbti: dict | None,
    date: datetime,
    *,
    holiday=None,
    workday_swap: bool | None = None,
) -> list[dict]:
    """根据 MBTI 派生信号微调基准模板。

    法定节日将 work 替换为 leisure/rest；调休上班日按工作日处理。
    """
    schedule = [slot.copy() for slot in _BASE_SCHEDULE_TEMPLATE]
    lively = mbti_signal(mbti, "E")
    planned = mbti_signal(mbti, "J")

    is_workday_swap_day = (
        is_workday_swap(date.date()) if workday_swap is None else workday_swap
    )
    is_weekend = date.weekday() >= 5 and not is_workday_swap_day
    is_legal_holiday = holiday is not None and holiday.type == "legal"

    for slot in schedule:
        # E 程度高：更多社交活动
        if slot["type"] == "leisure" and lively >= 0.7:
            if random.random() < 0.4:
                slot["activity"] = random.choice(["和朋友聊天", "刷社交媒体", "出去逛逛"])
                slot["type"] = "social"

        # J 程度高：早起
        if slot["activity"] == "起床洗漱" and planned >= 0.7:
            slot["start"] = "06:30"
            slot["end"] = "07:30"

        # 周末晚起 + 晚餐也顺延
        if is_weekend and slot["activity"] == "起床洗漱":
            slot["start"] = "09:00"
            slot["end"] = "10:00"
        if is_weekend and slot["activity"] == "吃早餐":
            slot["start"] = "10:00"
            slot["end"] = "11:00"
        if is_weekend and slot["type"] == "work":
            slot["activity"] = random.choice(["看书", "追剧", "逛街", "玩游戏", "画画"])
            slot["type"] = "leisure"

        # 法定节日：work → leisure/rest
        if is_legal_holiday and slot["type"] == "work":
            slot["activity"] = random.choice(["休息放松", "出去玩", "看书", "和朋友聚餐", "追剧"])
            slot["type"] = "leisure"

    return schedule


async def _get_user_memory_summary(
    user_id: str | None, limit: int = 5, *, workspace_id: str | None = None,
) -> str:
    """Borrow active user interests, never their identity/location or history."""
    if not user_id or not workspace_id:
        return ""
    try:
        from app.services.memory.storage import repo as memory_repo
        memories = await memory_repo.find_many(
            source="user",
            where={"userId": user_id, "workspaceId": workspace_id, "level": 1,
                   "isArchived": False, "mainCategory": "偏好"},
            order={"importance": "desc"},
            take=limit,
        )
        if not memories:
            return ""
        return "；".join(m.content for m in memories if m.content)
    except Exception as e:
        logger.warning(f"Failed to load user memories for schedule: {e}")
        return ""


async def _ground_schedule(agent: Any, schedule: list[dict]) -> list[dict]:
    clean, _ = await _ground_schedule_result(agent, schedule)
    return clean


async def _ground_schedule_result(
    agent: Any, schedule: list[dict], *, policy_template: str | None = None,
) -> tuple[list[dict], bool]:
    from app.services.persona_grounding import GroundingUnavailable, verify_generated_locations
    texts = [str(s.get("event") or s.get("activity") or "") for s in schedule]
    verified = True
    try:
        rejected = await verify_generated_locations(agent, texts, kind="schedule", policy_template=policy_template)
    except GroundingUnavailable:
        verified = False
        # Unknown specific geography must not become live state. Keep times and
        # workload; a neutral activity is preferable to inventing a journey.
        from app.services.persona_grounding import needs_location_verification
        rejected = [i for i, text in enumerate(texts)
                    if needs_location_verification(text, implicit_self=True)]
    clean = [s.copy() for s in schedule]
    for i in rejected:
        clean[i]["event"] = clean[i]["activity"] = "日常活动"
    return clean, verified


async def _cache_schedule(agent_id: str, date: datetime, schedule: list[dict]) -> None:
    """缓存当日作息到Redis并持久化到DB。"""
    redis = await get_redis()
    await redis.set(_schedule_key(agent_id, date), json.dumps(schedule, ensure_ascii=False), ex=86400 * 2)

    # 持久化到 DB（用于历史查询）
    #
    # 必须去掉时区。列是 @db.Date, 而驱动会先把带时区的 datetime 归一到 UTC:
    # 2026-07-29 00:00+08:00 → 2026-07-28 16:00 UTC → 截断成 07-28。于是每天的
    # 作息都被写到前一天那行上, 撞唯一键 (agentId, date) 把它覆盖掉 —— 整个历史
    # 错位一天, 而且当天永远查不到自己的表。
    #
    # 热路径读 Redis (键按本地日期字符串拼), 所以聊天一直是对的, 这个错位才藏了
    # 这么久。但 Redis miss 时 conversations.py 会回落查 DB, 那时拿到的是昨天的
    # 安排。
    date_only = date.replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=None,
    )
    try:
        await db.aidailyschedule.upsert(
            where={"agentId_date": {"agentId": agent_id, "date": date_only}},
            data={
                "create": {
                    "agent": {"connect": {"id": agent_id}},
                    "date": date_only,
                    "scheduleData": Json(schedule),
                },
                "update": {
                    "scheduleData": Json(schedule),
                },
            },
        )
    except Exception as e:
        logger.warning(f"Failed to persist schedule to DB for {agent_id}: {e}")


async def save_life_overview(agent_id: str, description: str) -> None:
    """保存生活画像到DB和Redis（spec §2.1：纯文本，不含结构化字段）。

    历史遗留的 lifeOverviewData JSON 列保留不写入（仍在 prisma schema 里以避免
    DB 迁移，但不再使用）。
    """
    # 先写 DB，成功后才写 Redis（避免 split-brain）
    try:
        await db.aiagent.update(
            where={"id": agent_id},
            data={"lifeOverview": description},
        )
    except Exception as e:
        logger.warning(f"Failed to save life overview to DB for {agent_id}: {e}")
        return

    redis = await get_redis()
    await redis.set(f"life_overview:{agent_id}", description, ex=86400 * 30)


async def get_life_overview(agent_id: str) -> str | None:
    """获取生活画像文本。先查 Redis，miss 则查 DB 并回填缓存。"""
    redis = await get_redis()
    data = await redis.get(f"life_overview:{agent_id}")
    if data:
        return data.decode() if isinstance(data, bytes) else data

    # Redis miss → 查 DB fallback
    try:
        agent = await db.aiagent.find_unique(where={"id": agent_id})
        if agent and agent.lifeOverview:
            # 回填 Redis 缓存
            await redis.set(f"life_overview:{agent_id}", agent.lifeOverview, ex=86400 * 30)
            return agent.lifeOverview
    except Exception as e:
        logger.warning(f"Failed to load life overview from DB for {agent_id}: {e}")

    return None


def _schedule_key(agent_id: str, date: datetime) -> str:
    return f"schedule:{agent_id}:{date.strftime('%Y%m%d')}"


def _adj_key(agent_id: str) -> str:
    return f"schedule_adj:{agent_id}:{_local_now().strftime('%Y%m%d')}"


async def has_cached_schedule(agent_id: str, date: datetime | None = None) -> bool:
    """当天作息是否已经生成过 (只看键在不在, 不做 grounding, 给批量任务用)。"""
    redis = await get_redis()
    return bool(await redis.exists(_schedule_key(agent_id, date or _local_now())))


async def get_cached_schedule(agent_id: str, date: datetime | None = None) -> list[dict] | None:
    """从Redis获取缓存的当日作息。"""
    date = date or _local_now()
    redis = await get_redis()
    data = await redis.get(_schedule_key(agent_id, date))
    if data:
        try:
            schedule = json.loads(data)
            if not isinstance(schedule, list) or not all(isinstance(s, dict) for s in schedule):
                return None
            agent = await db.aiagent.find_unique(where={"id": agent_id})
            from app.services.persona_grounding import canonical_facts, needs_location_verification
            policy = ""
            if any(needs_location_verification(
                str(slot.get("event") or slot.get("activity") or ""), implicit_self=True,
            ) for slot in schedule):
                try:
                    # Read the same current Web-managed policy as the verifier.
                    # A disabled/unavailable policy cannot reuse an old verdict.
                    policy = await get_prompt_text("persona.grounding_check")
                except Exception:
                    return await _ground_schedule(agent, schedule)
            fingerprint = hashlib.sha256(json.dumps(
                [canonical_facts(agent), schedule, policy], ensure_ascii=False, sort_keys=True,
            ).encode()).hexdigest()
            checked_key = f"{_schedule_key(agent_id, date)}:grounded:v2:{fingerprint}"
            checked = await redis.get(checked_key)
            if checked:
                return json.loads(checked)
            clean, verified = await _ground_schedule_result(agent, schedule, policy_template=policy or None)
            # Original DB/Redis schedules remain evidence. This is a derived,
            # bounded read cache, invalidated by any profile or schedule change.
            await redis.set(checked_key, json.dumps(clean, ensure_ascii=False), ex=86400 if verified else 300)
            return clean
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def get_current_status(schedule: list[dict], now: datetime | None = None) -> dict:
    """根据作息表查询当前时段状态。

    Spec Part 1 §2.1：LLM 直接输出 4 个状态之一（空闲/忙碌/很忙碌/睡眠）。
    为了兼容旧缓存数据（字段 activity/type 的 6 值枚举），读取时统一规范化。

    返回 {"event": str, "activity": str (= event, 永不为空),
          "status": "idle"|"busy"|"very_busy"|"sleep",
          "status_source": "slot"|"unscheduled_night"|"unscheduled_gap",
          "upcoming": str}

    没有命中任何一段时不再一律当空闲, 见 `_unscheduled_status`。顶部状态、当前状态
    回复、主回复的状态约束、回复延迟都从这里取, 夜间判定只能在这一处改。

    `upcoming` 是接下来一两段安排的可读串 (可能为空)。放在这里而不是让调用方各自
    去算, 是因为聊天路径有两处构建 ai_status, 分开算下次新增调用点会漏 —— 而漏掉
    的表现是 AI 谈起未来时又开始现编, 不报错。
    """
    now = now or _local_now()
    minute = now.hour * 60 + now.minute
    upcoming = format_upcoming(get_upcoming_slots(schedule, now))

    spans = [(span, slot) for slot in schedule if (span := _slot_span(slot))]
    for (start, end), slot in spans:
        # 第二个条件是跨午夜段在今天凌晨的那一截 (23:00-07:00 在 01:00 也算命中)
        if start <= minute < end or start <= minute + _DAY_MINUTES < end:
            return {**_slot_to_status(slot), "status_source": "slot", "upcoming": upcoming}

    return {**_unscheduled_status(spans, minute), "upcoming": upcoming}


_DAY_MINUTES = 24 * 60
_CLOCK_RE = re.compile(r"(\d{1,2})\s*[:：]\s*(\d{2})")
_NEXT_DAY_MARKERS = ("次日", "第二天", "翌日", "隔天")


def _clock_minutes(value: Any) -> tuple[int, bool] | None:
    """"07:30" → (450, False); "次日 07:30" → (450, True); 认不出返回 None。

    作息表的时刻是 LLM 写的: 生产 2026-10 的 172 份里有 44 份把结束时间写成
    「次日07:30」「次日 07:30」。旧实现按字符串比较, 「次」的码位碰巧大于任何数字,
    当天的比较歪打正着能用, 所以一直没人发现它其实不是时刻。
    """
    text = str(value or "").strip()
    match = _CLOCK_RE.search(text)
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if minute >= 60 or hour > 24 or (hour == 24 and minute):
        return None
    return hour * 60 + minute, any(marker in text for marker in _NEXT_DAY_MARKERS)


def _slot_span(slot: dict) -> tuple[int, int] | None:
    """slot → 以当天 00:00 起算的 [start, end) 分钟数, 跨午夜的 end 落在 1440 之后。"""
    start = _clock_minutes(slot.get("start"))
    end = _clock_minutes(slot.get("end"))
    if start is None or end is None:
        return None
    start_minute = start[0] + (_DAY_MINUTES if start[1] else 0)
    end_minute = end[0] + (_DAY_MINUTES if end[1] else 0)
    if end_minute <= start_minute:
        end_minute += _DAY_MINUTES
    return start_minute, end_minute


def _unscheduled_status(spans: list[tuple[tuple[int, int], dict]], minute: int) -> dict:
    """作息表没写到的时刻。

    生成指令要的是「从早晨起床到晚上睡觉，不含凌晨时段」, 所以第一段之前、最后
    一段之后本来就是睡着的。旧实现一律回落「自由时间/空闲」, 而且返回值缺
    activity 键: 每天凌晨到起床、最后一段结束到午夜, 顶部显示「空闲」, 当前状态
    回复拿到「你现在在，有空聊天。」这种空活动, 模型只好现编 —— 生产上就是前一句
    还说在睡觉, 半小时后改口「上班摸鱼呢」。生产 2026-10 有 58% 的作息表不覆盖
    夜间, 这不是个例。

    只有两段之间漏掉的空档 (LLM 偶尔少写一刻钟) 才算自由时间。
    """
    if spans:
        first_start = min(start for (start, _), _ in spans)
        last_end = max(end for (_, end), _ in spans)
        if minute < first_start or minute >= last_end:
            return _status_with_activity("sleep", "", source="unscheduled_night")
    return _status_with_activity("idle", "", source="unscheduled_gap")


# 作息段没写事件时按状态补一个, 保证 activity 永远非空 —— 下游有的拿它拼句子
# (「你现在在{activity}」), 有的拿它判断要不要注入状态约束段 (空就整段跳过)。
_DEFAULT_ACTIVITY = {
    "sleep": "睡觉",
    "very_busy": "手头的事",
    "busy": "忙手头的事",
    "idle": "自由时间",
}


def _status_with_activity(status: str, event: str, *, source: str) -> dict:
    event = str(event or "").strip() or _DEFAULT_ACTIVITY.get(status, "自由时间")
    return {"event": event, "activity": event, "status": status, "status_source": source}


# spec 的 4 个中文状态 → 代码内部 ASCII 枚举
_SPEC_STATUS_MAP = {
    "空闲": "idle",
    "忙碌": "busy",
    "很忙碌": "very_busy",
    "很忙": "very_busy",  # 宽松匹配
    "睡眠": "sleep",
}

# 兼容旧缓存数据：6 值 type 枚举 → 4 值 status
_LEGACY_TYPE_MAP = {
    "sleep": "sleep",
    "work": "busy",
    "routine": "busy",
    "rest": "idle",
    "leisure": "idle",
    "social": "idle",
}


def _slot_to_status(slot: dict) -> dict:
    """规范化 slot 到 spec 字段 {event, status}。

    优先读取 spec 字段；回退到旧的 6 值 type + activity 字段以兼容历史缓存。
    """
    # Spec 字段（LLM 新输出）
    event = slot.get("event") or slot.get("activity", "")
    raw_status = slot.get("status")

    if raw_status in ("idle", "busy", "very_busy", "sleep"):
        status = raw_status
    elif isinstance(raw_status, str) and raw_status in _SPEC_STATUS_MAP:
        status = _SPEC_STATUS_MAP[raw_status]
    else:
        # 回退：旧缓存仍是 type 字段（6 值枚举）
        legacy_type = slot.get("type", "leisure")
        status = _LEGACY_TYPE_MAP.get(legacy_type, "idle")
        # 旧 work 时段在核心小时段细分为 very_busy（保留旧行为）
        if status == "busy" and legacy_type == "work":
            start = slot.get("start", "00:00")
            if "09:00" <= start < "12:00" or "14:00" <= start < "17:00":
                status = "very_busy"

    # Keep `activity` alias for backward-compat with existing consumers
    # (proactive/context.py, proactive/sender.py). These will migrate to
    # `event` over time.
    event = str(event or "").strip() or _DEFAULT_ACTIVITY.get(status, "自由时间")
    return {"event": event, "activity": event, "status": status}


_STATUS_LABELS = {
    "idle": "空闲",
    "busy": "忙碌",
    "very_busy": "很忙碌",
    "sleep": "睡眠",
}

# Deprecated: kept only for the public `type_label()` function used by api.
# Spec Part 1 §2.1 doesn't use `type` at all — schedule slots now carry only
# {event, status}. Callers should use `status_label` instead.
_LEGACY_TYPE_LABELS = {
    "leisure": "休闲",
    "work": "工作",
    "routine": "日常",
    "sleep": "睡眠",
    "social": "社交",
    "rest": "休息",
}


def status_label(status: str) -> str:
    """中文状态标签。"""
    return _STATUS_LABELS.get(status, status)


def type_label(slot_type: str) -> str:
    """中文类型标签（已废弃；新 schedule 只有 event/status，type 字段不再使用）。"""
    return _LEGACY_TYPE_LABELS.get(slot_type, slot_type)


# --- 作息查询意图识别 ---

_SCHEDULE_QUERY_KEYWORDS = {
    "current": [
        "在干嘛", "在做什么", "做什么呢", "干什么呢", "干嘛呢",
        "忙不忙", "有空吗", "忙吗", "在忙吗",
        "现在呢", "你呢", "在吗",
        "最近怎么样", "最近好吗", "今天怎么样",
        "有没有想我", "想我了吗",
    ],
    "routine": ["几点起", "几点睡", "作息", "日程", "时间表", "今天有什么安排"],
    "date": ["明天", "后天", "周末", "下午"],
}


@dataclass(frozen=True)
class ScheduleQueryScope:
    """Structured scope for a schedule query.

    Intent detection answers "is this a schedule query"; this scope answers
    "which part of the schedule should be used".  Keeping it here prevents
    date/current branching from being re-implemented in chat handlers.
    """

    query_type: str
    target_date: datetime | None = None
    date_label: str = "相关日期"
    explicit_time_text: str | None = None
    source: str = "keyword"


_SCHEDULE_QUERY_CUES = tuple(dict.fromkeys(
    _SCHEDULE_QUERY_KEYWORDS["current"]
    + _SCHEDULE_QUERY_KEYWORDS["routine"]
    + [
        "忙", "有空", "空不空", "方便吗", "有时间", "安排",
        "日程", "计划", "排满", "满不满",
    ]
))

_DATE_ANCHOR_WORDS = {
    "今天", "明天", "后天", "大后天", "昨天", "前天", "大前天",
}
_PERIOD_LABELS = {
    "早上": "早上",
    "早晨": "早上",
    "上午": "上午",
    "中午": "中午",
    "下午": "下午",
    "晚上": "晚上",
    "凌晨": "凌晨",
}
_CURRENT_STATE_SCOPE_PHRASES = (
    "最近怎么样",
    "最近好吗",
    "今天怎么样",
)


def _has_schedule_query_cue(message: str) -> bool:
    if not any(cue in message for cue in _SCHEDULE_QUERY_CUES):
        return False
    if "你" in message or "Hillow" in message:
        return True
    return any(marker in message for marker in (
        "吗", "呢", "？", "?", "几", "什么", "有没有", "忙不忙", "空不空",
    ))


def _duration_seconds(parsed: ParsedTime) -> float:
    return (parsed.end - parsed.start).total_seconds()


def _pick_schedule_time(parsed_times: list[ParsedTime]) -> ParsedTime | None:
    if not parsed_times:
        return None

    def _is_date_anchor(item: ParsedTime) -> bool:
        text = item.original_text
        return (
            text in _DATE_ANCHOR_WORDS
            or "周" in text
            or "月" in text
            or "年" in text
            or "日" in text
            or "号" in text
        )

    date_anchors = [item for item in parsed_times if _is_date_anchor(item)]
    if date_anchors:
        return max(date_anchors, key=lambda item: (item.confidence, -_duration_seconds(item)))
    return min(parsed_times, key=lambda item: (_duration_seconds(item), -item.confidence))


def _date_label_for(parsed: ParsedTime, message: str, now: datetime) -> str:
    text = parsed.original_text
    period = next((label for word, label in _PERIOD_LABELS.items() if word in message), "")

    if "大后天" in message:
        return f"大后天{period}" if period else "大后天"
    if "后天" in message:
        return f"后天{period}" if period else "后天"
    if "明天" in message:
        if period == "晚上":
            return "明晚"
        return f"明天{period}" if period else "明天"
    if "今天" in message:
        if period == "晚上":
            return "今晚"
        return f"今天{period}" if period else "今天"
    if text in _PERIOD_LABELS:
        return "今晚" if text == "晚上" else f"今天{_PERIOD_LABELS[text]}"
    if parsed.start.date() == now.date() and period:
        return "今晚" if period == "晚上" else f"今天{period}"
    return text or "相关日期"


def resolve_schedule_query_scope(
    message: str,
    *,
    now: datetime | None = None,
    require_query_cue: bool = False,
) -> ScheduleQueryScope | None:
    """Resolve the schedule range requested by a user message.

    `require_query_cue=True` is for keyword fallback intent detection, where a
    bare explicit date like "明天是我的生日" must not become a schedule query.
    When the LLM has already classified the message as SCHEDULE_QUERY, callers
    should leave it False so parsed time expressions can determine the scope.
    """
    if require_query_cue and not _has_schedule_query_cue(message):
        return None

    now = now or _local_now()

    if any(phrase in message for phrase in _CURRENT_STATE_SCOPE_PHRASES):
        return ScheduleQueryScope(
            query_type="current",
            target_date=now,
            date_label="今天",
            source="keyword",
        )

    if has_explicit_time(message):
        parsed_times = parse_time_expressions(message, now=now)
        parsed = _pick_schedule_time(parsed_times)
        if parsed:
            return ScheduleQueryScope(
                query_type="date",
                target_date=parsed.start,
                date_label=_date_label_for(parsed, message, now),
                explicit_time_text=parsed.original_text,
                source="time_parser",
            )

    for query_type in ("routine", "current"):
        keywords = _SCHEDULE_QUERY_KEYWORDS.get(query_type, [])
        if any(kw in message for kw in keywords):
            return ScheduleQueryScope(
                query_type=query_type,
                target_date=now if query_type == "current" else None,
                date_label="今天" if query_type in {"current", "routine"} else "相关日期",
                source="keyword",
            )

    return None


def detect_schedule_query(message: str) -> str | None:
    """检测作息查询意图。返回 'current'/'routine'/'date' 或 None。"""
    scope = resolve_schedule_query_scope(message, require_query_cue=True)
    return scope.query_type if scope else None


def get_upcoming_slots(
    schedule: list[dict], now: datetime | None = None, limit: int = 2,
) -> list[dict]:
    """当前时段之后接下来的几段安排。

    `get_current_status` 只回答"此刻在干什么"。但对话里更常出现的是未来时态 ——
    「等下一起看个电影吧」「明天有空吗」。只知道当下的话, AI 只能现编一个安排,
    而且编完不留痕: 到了那个时间点没人对照, 用户下次问起就前后矛盾了。

    Generative Agents (arXiv 2304.03442 §4.3) 把这个问题讲得很直白 —— 只给当下
    状态而不给计划, agent 会 "eat lunch at 12 pm, but then again at 12:30 pm
    and 1 pm": 瞬间合理, 长期崩坏。

    只取接下来 limit 段。给全天会让这段太长, 而且远端的安排对当前这轮对话没用。
    """
    now = now or _local_now()
    current_time = now.strftime("%H:%M")

    upcoming: list[dict] = []
    for slot in schedule:
        start = str(slot.get("start", ""))
        # 跨午夜的段 (23:00-07:00) 起点在今天但延伸到明天, 不算"接下来"
        if not start or start <= current_time:
            continue
        upcoming.append(slot)
        if len(upcoming) >= limit:
            break
    return upcoming


def format_upcoming(slots: list[dict]) -> str:
    """把接下来的安排渲染成一行。没有安排时返回空串, 由调用方决定是否略过整段。"""
    if not slots:
        return ""
    parts = []
    for slot in slots:
        event = slot.get("event") or slot.get("activity") or ""
        if not event:
            continue
        parts.append(f"{slot.get('start', '')} {event}")
    return "；".join(parts)


def format_schedule_context(status: dict) -> str:
    """格式化当前状态供Prompt注入。"""
    s = status.get("status", "idle")
    activity = (
        str(status.get("activity") or status.get("event") or "").strip()
        or _DEFAULT_ACTIVITY.get(s, "自由时间")
    )

    if s == "sleep":
        # 夜间推断出来的睡眠只有「睡觉」二字, 别渲染成「正在睡觉（睡觉）」
        detail = "" if activity == _DEFAULT_ACTIVITY["sleep"] else f"（{activity}）"
        return f"你现在正在睡觉{detail}，可能会延迟回复。"
    elif s == "very_busy":
        return f"你现在正在忙{activity}，这是最忙的时段，回复会比较慢。"
    elif s == "busy":
        return f"你现在正在{activity}，可能需要一会儿才能回复。"
    else:
        return f"你现在在{activity}，有空聊天。"


# --- 作息调整 ---

async def compute_adjustment_feasibility(
    agent_id: str,
    current_status: dict,
    intimacy_score: float = 0.0,
    mbti: dict | None = None,
    adjustment_minutes: int = 0,
) -> dict:
    """4F.1 计算作息调整可行性评分。

    base=50, 加减分:
    - 亲密度>80 → +20
    - P 程度>0.7 → +15
    - 当前sleep → -10
    - 调整幅度>60min → -30
    - 今日已调整≥2次 → -50

    评分区间: <30拒绝, 30-70部分接受, >70接受
    """
    score = 50

    if intimacy_score > 80:
        score += 20

    if mbti and mbti_signal(mbti, "P") > 0.7:
        score += 15

    if current_status.get("status") == "sleep":
        score -= 10

    if abs(adjustment_minutes) > 60:
        score -= 30

    # 今日调整次数
    redis = await get_redis()
    adj_count = int(await redis.get(_adj_key(agent_id)) or 0)
    if adj_count >= 2:
        score -= 50

    score = max(0, min(100, score))

    return {"score": score, "today_adjustments": adj_count}


async def log_schedule_adjustment(
    agent_id: str,
    current_status: dict,
    request: str,
    adjustment: str,
    *,
    feasibility_score: int | None = None,
) -> None:
    """记录一次真实的作息调整 (spec §3.4.2 第 4 步): Redis 当日计数 + DB 日志。

    仅在 LLM 确实产出了 adjustment (真的改了作息) 时调用; 婉拒/口头安抚不落表、
    不计数。Redis 当日计数供 compute_adjustment_feasibility 作"今日已调整次数"信号,
    让反复索求时 AI 的配合意愿自然下降 (但由大模型决定语气, 不再有硬编码判决)。
    """
    redis = await get_redis()
    key = _adj_key(agent_id)
    await redis.incr(key)
    await redis.expire(key, 86400)
    try:
        await db.scheduleadjustlog.create(data={
            "agentId": agent_id,
            "adjustType": "user_request",
            "oldValue": json.dumps(current_status, ensure_ascii=False),
            "newValue": f"{request} => {adjustment}",
            "reason": (
                f"feasibility={feasibility_score}"
                if feasibility_score is not None else "llm_generated"
            ),
        })
    except Exception as e:
        logger.warning(f"Failed to log schedule adjustment: {e}")


async def update_schedule_slot(
    agent_id: str,
    schedule: list[dict],
    current_status: dict,
    new_type: str = "leisure",
    new_activity: str = "和用户聊天",
) -> list[dict]:
    """更新当前时段并写回 Redis。按时间范围匹配当前时段。"""
    now = _local_now()
    now_str = now.strftime("%H:%M")
    updated = [s.copy() for s in schedule]
    for s in updated:
        if s.get("start", "") <= now_str < s.get("end", "24:00"):
            s["type"] = new_type
            s["activity"] = new_activity
            break
    await _cache_schedule(agent_id, now, updated)
    return updated


def format_full_schedule_for_query(
    schedule: list[dict],
    query_type: str,
    status: dict | None = None,
    date_label: str = "今天",
) -> str:
    """格式化完整日程供查询意图的 prompt 注入。"""
    lines = [
        f"{s['start']}-{s['end']} {s.get('event') or s.get('activity', '')}"
        for s in schedule
    ]
    full_text = "\n".join(lines)

    current = ""
    if query_type == "current" and status:
        current = f"\n当前状态：{format_schedule_context(status)}"

    if query_type == "current":
        instruction = "用户在问你现在在干什么。自然地描述你当前的状态和活动。"
    elif query_type == "routine":
        instruction = "用户在问你的日程安排。自然地描述你今天的计划，用你的性格说话。"
    else:
        instruction = f"用户在问你{date_label}的日程安排。只根据这份{date_label}作息表自然回答。"

    return f"以下是你{date_label}的作息：\n{full_text}{current}\n{instruction}"


# --- 每日作息回顾 ---


async def review_daily_schedule(
    agent_id: str, user_id: str, agent_name: str = "伙伴"
) -> list[str]:
    """Review yesterday once in the exact active owner/agent workspace.

    Daily summaries are optional derived memories. A failed attempt is recorded
    explicitly and cannot silently repeat partially committed memory writes.
    """
    import uuid
    from app.services.workspace.workspaces import get_active_workspace
    from app.services.memory.lifecycle.capacity import DAILY_SUMMARY_ROW_BUDGET

    workspace = await get_active_workspace(user_id=user_id, agent_id=agent_id)
    if (
        workspace is None
        or workspace.userId != user_id
        or workspace.agentId != agent_id
    ):
        return []
    day_end = _local_now().replace(hour=0, minute=0, second=0, microsecond=0)
    day_start = day_end - timedelta(days=1)
    local_date = day_start.replace(tzinfo=None)
    claimed = await db.query_raw(
        """INSERT INTO memory_daily_reviews(id,workspace_id,local_date,status)
           VALUES($1,$2,$3::date,'running') ON CONFLICT(workspace_id,local_date)
           DO NOTHING RETURNING id""",
        str(uuid.uuid4()),
        workspace.id,
        local_date.isoformat(),
    )
    if not claimed:
        logger.info("Daily review already attempted for this workspace/day")
        return []
    review_id = claimed[0]["id"]
    try:
        count = await db.aimemory.count(
            where={
                "userId": user_id,
                "workspaceId": workspace.id,
                "provenance": "daily_summary",
            }
        )
        budget = min(5, max(0, DAILY_SUMMARY_ROW_BUDGET - count))
        if budget == 0:
            await db.execute_raw(
                "UPDATE memory_daily_reviews SET status='capacity_skipped',updated_at=NOW() WHERE id=$1",
                review_id,
            )
            logger.warning(
                "Daily summary memory budget reached; factual chat memories are unaffected"
            )
            return []
        stored = await _review_daily_schedule(
            agent_id,
            user_id,
            agent_name,
            workspace.id,
            day_start,
            day_end,
            budget,
            review_id=review_id,
        )
        return stored
    except Exception:
        try:
            await db.execute_raw(
                "UPDATE memory_daily_reviews SET status='failed',updated_at=NOW() WHERE id=$1 AND status='running'",
                review_id,
            )
        except Exception:
            logger.exception("Could not persist failed daily review status")
        raise


async def _review_daily_schedule(
    agent_id: str,
    user_id: str,
    agent_name: str,
    workspace_id: str,
    day_start: datetime,
    day_end: datetime,
    budget: int,
    *,
    review_id: str,
) -> list[str]:
    from types import SimpleNamespace
    from app.services.schedule_domain.review_inputs import (
        collect_inputs,
        public_summary,
        render_games,
    )
    from app.services.schedule_domain.review_write import (
        prepare_candidates,
        commit_review,
    )
    from app.services.persona_grounding import grounding_context

    # Brief locks keep the snapshot coherent; no transaction spans model/network work.
    async with db.tx(timeout=timedelta(seconds=15)) as tx:
        await tx.execute_raw("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        inputs = await collect_inputs(
            tx,
            user_id=user_id,
            workspace_id=workspace_id,
            agent_id=agent_id,
            start=day_start,
            end=day_end,
            lock=True,
        )
    sampled_at = datetime.now(UTC).isoformat()
    payload = {
        "local_date": inputs["local_date"],
        "review_id": review_id,
        "inputs": inputs,
        "summary": public_summary(inputs, sampled_at),
        "calls": [],
    }
    prepared = []
    if inputs["schedule"]:
        agent = SimpleNamespace(**inputs["agent"])
        schedule = await _ground_schedule(agent, inputs["schedule"])
        schedule_text = "昨日作息计划（不代表活动已实际完成）：\n" + "\n".join(
            f"{s['start']}-{s['end']} {s.get('event') or s.get('activity', '')}"
            for s in schedule
        )
        adjustments_text = "计划调整记录（不代表执行结果）：\n" + "\n".join(
            f"- {r['adjust_type']}: {r['reason'] or ''}" for r in inputs["adjustments"]
        )
        proactive_text = (
            "AI生成的主动话术（不证明现实经历或用户已收到）：\n"
            + "\n".join(f"- {r['message']}" for r in inputs["proactive"])
        )
        if inputs["proactive_truncated"]:
            proactive_text += "\n（仅采样最近5条，并非全天全部记录）"
        summary_prompt = (await get_prompt_text("schedule.daily_summary")).format(
            name=inputs["agent"]["name"],
            schedule_text=schedule_text,
            adjustments_text=(
                adjustments_text if inputs["adjustments"] else "（昨日无计划调整记录）"
            ),
            proactive_text=(
                proactive_text if inputs["proactive"] else "（昨日无主动话术记录）"
            ),
            chat_summary_text="（未采集聊天回顾）",
            games_text=render_games(inputs["games"]),
        )
        summary_prompt += "\n\n" + await grounding_context(
            agent, prompt_reader=get_prompt_text
        )
        model = get_utility_model()
        model_name = getattr(model, "model_name", None) or getattr(model, "model", None)
        payload["model"] = model_name if isinstance(model_name, str) else None
        from app.services.llm.models import invoke_text

        summary_text = (await invoke_text(model, summary_prompt)).strip()
        payload["calls"].append(
            {"kind": "summary", "prompt": summary_prompt, "output": summary_text}
        )
        if summary_text:
            memories_prompt = (
                await get_prompt_text("schedule.daily_summary_memories")
            ).format(summary_text=summary_text)
            result = await invoke_json(model, memories_prompt)
            payload["calls"].append(
                {"kind": "classification", "prompt": memories_prompt, "output": result}
            )
            prepared = await prepare_candidates(
                result, user_id=user_id, workspace_id=workspace_id, budget=budget
            )
    return await commit_review(
        db,
        review_id=review_id,
        user_id=user_id,
        workspace_id=workspace_id,
        agent_id=agent_id,
        inputs=inputs,
        payload=payload,
        prepared=prepared,
    )
