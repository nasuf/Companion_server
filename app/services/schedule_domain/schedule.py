"""AI作息系统。

管理AI的生活画像、每日作息表、状态查询和作息调整。
生活画像由LLM生成，作息表基于模板+个性化。
"""

from __future__ import annotations

import json
import logging
import random
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from prisma import Json

from app.config import settings
from app.db import db
from app.redis_client import get_redis
from app.services.llm.models import get_utility_model, invoke_json, invoke_text
from app.services.prompting.store import get_prompt_text
from app.services.schedule_domain.time_service import classify_day_kind, is_holiday
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
) -> list[dict]:
    """生成每日作息表。基于生活画像+模板+个性化。节日强制LLM路径。

    Spec Part 1 §2.1 要求 prompt 注入:
        姓名 / 年龄 / 职业 / 性格(MBTI) / 生活画像 / 日期属性 [+ 用户记忆]
    age/occupation 如果调用方已知可直接传; 缺时从 DB 懒查 agent 记录.
    """
    date = date or _local_now()
    weekday = _WEEKDAY_CN[date.weekday()]

    # 检测节日 + 当日属性
    holiday = is_holiday(date.date())
    day_kind = classify_day_kind(date.date(), holiday)

    # Age / occupation 懒查: 调用方未传时从 DB 拉 agent
    if age is None or occupation is None:
        try:
            agent_row = await db.aiagent.find_unique(where={"id": agent_id})
            if agent_row is not None:
                age = age if age is not None else getattr(agent_row, "age", None)
                occupation = (
                    occupation
                    if occupation is not None
                    else getattr(agent_row, "occupation", None)
                )
        except Exception as e:
            logger.debug(f"Failed to lazy-lookup agent {agent_id}: {e}")

    personality_brief = _mbti_brief(mbti)

    base_fmt = {
        "name": name,
        "age": age if age is not None else "未知",
        "occupation": occupation or "普通人",
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
                memory_summary = await _get_user_memory_summary(user_id)
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
                await _cache_schedule(agent_id, date, schedule)
                logger.info(f"[SCHEDULE] {agent_id} {date.date()} gen via {prompt_key}")
                return schedule
        except Exception as e:
            logger.warning(f"LLM schedule generation failed, falling back to template: {e}")

    # 兜底: 无 life_overview 或 LLM 失败时用模板
    schedule = _personalize_template(mbti, date, holiday=holiday)
    await _cache_schedule(agent_id, date, schedule)
    return schedule


def _personalize_template(mbti: dict | None, date: datetime, *, holiday=None) -> list[dict]:
    """根据 MBTI 派生信号微调基准模板。法定节日将 work 替换为 leisure/rest。"""
    schedule = [slot.copy() for slot in _BASE_SCHEDULE_TEMPLATE]
    lively = mbti_signal(mbti, "E")
    planned = mbti_signal(mbti, "J")

    is_weekend = date.weekday() >= 5
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


async def _get_user_memory_summary(user_id: str | None, limit: int = 5) -> str:
    """查询用户L1核心记忆，返回简短摘要文本。"""
    if not user_id:
        return ""
    try:
        from app.services.memory.storage import repo as memory_repo
        memories = await memory_repo.find_many(
            source="user",
            where={"userId": user_id, "level": 1},
            order={"importance": "desc"},
            take=limit,
        )
        if not memories:
            return ""
        return "；".join(m.content for m in memories if m.content)
    except Exception as e:
        logger.warning(f"Failed to load user memories for schedule: {e}")
        return ""


async def _cache_schedule(agent_id: str, date: datetime, schedule: list[dict]) -> None:
    """缓存当日作息到Redis并持久化到DB。"""
    redis = await get_redis()
    await redis.set(_schedule_key(agent_id, date), json.dumps(schedule, ensure_ascii=False), ex=86400 * 2)

    # 持久化到 DB（用于历史查询）
    date_only = date.replace(hour=0, minute=0, second=0, microsecond=0)
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


async def get_cached_schedule(agent_id: str, date: datetime | None = None) -> list[dict] | None:
    """从Redis获取缓存的当日作息。"""
    date = date or _local_now()
    redis = await get_redis()
    data = await redis.get(_schedule_key(agent_id, date))
    if data:
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def get_current_status(schedule: list[dict], now: datetime | None = None) -> dict:
    """根据作息表查询当前时段状态。

    Spec Part 1 §2.1：LLM 直接输出 4 个状态之一（空闲/忙碌/很忙碌/睡眠）。
    为了兼容旧缓存数据（字段 activity/type 的 6 值枚举），读取时统一规范化。

    返回 {"event": str, "status": "idle"|"busy"|"very_busy"|"sleep"}
    """
    now = now or _local_now()
    current_time = now.strftime("%H:%M")

    for slot in schedule:
        start = slot.get("start", "00:00")
        end = slot.get("end", "23:59")

        # 处理跨午夜的时段（如23:00-07:00）
        if start > end:
            if current_time >= start or current_time < end:
                return _slot_to_status(slot)
        else:
            if start <= current_time < end:
                return _slot_to_status(slot)

    return {"event": "自由时间", "status": "idle"}


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


def detect_schedule_query(message: str) -> str | None:
    """检测作息查询意图。返回 'current'/'routine'/'date' 或 None。"""
    for intent, keywords in _SCHEDULE_QUERY_KEYWORDS.items():
        if any(kw in message for kw in keywords):
            return intent
    return None


def format_schedule_context(status: dict) -> str:
    """格式化当前状态供Prompt注入。"""
    activity = status.get("activity", "")
    s = status.get("status", "idle")

    if s == "sleep":
        return f"你现在正在睡觉（{activity}），可能会延迟回复。"
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


async def handle_schedule_adjustment(
    agent_id: str,
    request: str,
    current_status: dict,
    intimacy_score: float = 0.0,
    mbti: dict | None = None,
    adjustment_minutes: int = 0,
) -> dict:
    """处理作息调整请求 (spec §1.2 起 MBTI)。

    4F.1: 基于可行性评分决定接受/拒绝。
    返回 {"accepted": bool, "response": str, "score": int}
    """
    feasibility = await compute_adjustment_feasibility(
        agent_id=agent_id,
        current_status=current_status,
        intimacy_score=intimacy_score,
        mbti=mbti,
        adjustment_minutes=adjustment_minutes,
    )
    score = feasibility["score"]
    activity = current_status.get("activity", "")

    async def _record_adjustment(accepted: bool) -> None:
        """Redis计数 + DB持久化。"""
        redis = await get_redis()
        key = _adj_key(agent_id)
        await redis.incr(key)
        await redis.expire(key, 86400)
        try:
            await db.scheduleadjustlog.create(data={
                "agentId": agent_id,
                "adjustType": "user_request",
                "oldValue": json.dumps(current_status, ensure_ascii=False),
                "newValue": request,
                "reason": f"feasibility={score}, accepted={accepted}",
            })
        except Exception as e:
            logger.warning(f"Failed to log schedule adjustment: {e}")

    if score >= 70:
        # 接受
        await _record_adjustment(True)

        if current_status.get("status") == "sleep":
            response = f"好吧...本来在{activity}，那就再聊一会儿～"
        elif current_status.get("status") in ("busy", "very_busy"):
            response = f"刚好{activity}差不多了，可以聊一会儿～"
        else:
            response = ""
        return {"accepted": True, "response": response, "score": score}

    elif score >= 30:
        # 部分接受（50%概率）
        if random.random() < 0.5:
            await _record_adjustment(True)
            return {
                "accepted": True,
                "response": f"嗯...那稍微调整一下吧，不过不能太久哦",
                "score": score,
            }
        return {
            "accepted": False,
            "response": f"这个时间不太方便呢，要不换个时间？",
            "score": score,
        }

    else:
        # 拒绝
        if current_status.get("status") == "sleep":
            response = "不行啦，太晚了我真的好困...明天再聊好不好？"
        else:
            response = "今天已经调整过好几次了，这次真的不行啦"
        return {"accepted": False, "response": response, "score": score}


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
) -> str:
    """格式化完整日程供查询意图的 prompt 注入。"""
    lines = [
        f"{s['start']}-{s['end']} {s.get('event') or s.get('activity', '')}"
        for s in schedule
    ]
    full_text = "\n".join(lines)

    current = ""
    if status:
        current = f"\n当前状态：{format_schedule_context(status)}"

    if query_type == "current":
        instruction = "用户在问你现在在干什么。自然地描述你当前的状态和活动。"
    elif query_type == "routine":
        instruction = "用户在问你的日程安排。自然地描述你今天的计划，用你的性格说话。"
    else:
        instruction = "用户在问你的日程安排。根据你的作息表自然回答。"

    return f"以下是你今天的完整作息：\n{full_text}{current}\n{instruction}"


# --- 每日作息回顾 ---

async def review_daily_schedule(agent_id: str, user_id: str, agent_name: str = "伙伴") -> list[str]:
    """回顾当日作息，合并调整记录+主动日志+聊天摘要，生成AI自我记忆。"""
    from app.services.memory.storage.persistence import store_memory
    from app.services.proactive.history import get_proactive_history

    schedule = await get_cached_schedule(agent_id)
    if not schedule:
        return []

    schedule_text = "\n".join(
        f"{s['start']}-{s['end']} {s.get('event') or s.get('activity', '')}" for s in schedule
    )

    # 查询当日调整记录
    adjustments_text = ""
    try:
        today_start = _local_now().replace(hour=0, minute=0, second=0, microsecond=0)
        adjustments = await db.scheduleadjustlog.find_many(
            where={
                "agentId": agent_id,
                "createdAt": {"gte": today_start},
            },
        )
        if adjustments:
            lines = [f"- {a.adjustType}: {a.reason}" for a in adjustments]
            adjustments_text = "\n今日作息调整：\n" + "\n".join(lines)
    except Exception as e:
        logger.warning(f"Failed to load adjustments for review: {e}")

    # 查询当日主动消息日志
    proactive_text = ""
    try:
        logs = await get_proactive_history(agent_id, user_id, limit=5)
        if logs:
            lines = [f"- {p['content']}" for p in logs if p.get("content")]
            if lines:
                proactive_text = "\n今日主动消息：\n" + "\n".join(lines)
    except Exception as e:
        logger.warning(f"Failed to load proactive history for review: {e}")

    # TODO(workspace-isolation): 原本用 redis.scan(match='cache:sum:*') 取前 3 个
    # 注入聊天回顾, 但 cache:sum:* 无对应写入点, 是死路径 + 跨 workspace 泄漏风险
    # (SCAN 会匹配任意 agent/user 的 summary). 若未来恢复聊天摘要注入, 必须改为
    # cache:sum:{agent_id}:{user_id}:{...} 精确 key 格式.
    chat_summary_text = "（无聊天回顾）"

    # Spec Part 1 §2.2: Step 1 — 先生成 200 字自然语言总结。
    from app.services.llm.models import invoke_text
    summary_prompt = (await get_prompt_text("schedule.daily_summary")).format(
        name=agent_name,
        schedule_text=schedule_text,
        adjustments_text=adjustments_text or "（无调整）",
        proactive_text=proactive_text or "（无主动消息）",
        chat_summary_text=chat_summary_text,
    )
    try:
        summary_text = (await invoke_text(get_utility_model(), summary_prompt)).strip()
    except Exception as e:
        logger.warning(f"Daily summary (text) failed for agent {agent_id}: {e}")
        return []
    if not summary_text:
        logger.info(f"Daily summary empty for agent {agent_id}, skip memory extraction")
        return []

    # Spec Part 1 §2.3: Step 2 — 把总结拆分为记忆条目 + 五类分类 + 0-100 打分。
    memories_prompt = (await get_prompt_text("schedule.daily_summary_memories")).format(
        summary_text=summary_text,
    )
    try:
        result = await invoke_json(get_utility_model(), memories_prompt)
    except Exception as e:
        logger.warning(f"Daily summary memory classification failed for agent {agent_id}: {e}")
        return []

    if not isinstance(result, dict):
        logger.warning(f"Schedule review returned non-dict: {type(result)}")
        return []
    memories = result.get("memories", [])
    if not isinstance(memories, list):
        logger.warning(f"Schedule review 'memories' is not a list: {type(memories)}")
        return []

    # Spec Part 1 §2.3 layer mapping: score ≥85→L1, 50-84→L2, 10-49→L3, <10→drop
    def _score_to_level(score: float) -> tuple[int, float] | None:
        if score < 10:
            return None
        importance = min(1.0, max(0.1, score / 100.0))
        if score >= 85:
            return (1, importance)
        if score >= 50:
            return (2, importance)
        return (3, importance)

    stored = []
    for mem in memories[:10]:
        # Spec 格式：{"type":"类型","content":"记忆内容","score":0-100}
        if isinstance(mem, str):
            content, level, importance = mem, 3, 0.3
        elif isinstance(mem, dict):
            content = mem.get("content", "")
            score = mem.get("score")
            if isinstance(score, (int, float)):
                mapped = _score_to_level(float(score))
                if mapped is None:
                    continue
                level, importance = mapped
            else:
                # 兜底：旧格式 level + importance
                level = mem.get("level", 3)
                importance = min(1.0, mem.get("importance", 0.3))
        else:
            continue

        if not content:
            continue

        # Spec §2.3 五类记忆：身份/情绪/偏好边界/生活/思维
        raw_type = mem.get("type") if isinstance(mem, dict) else None
        main_cat = _MEM_TYPE_TO_MAIN.get(str(raw_type or "").strip(), "生活")

        mem_id = await store_memory(
            user_id=user_id,
            content=content,
            memory_type="life",
            main_category=main_cat,
            level=level,
            importance=importance,
            source="ai",
        )
        if mem_id:
            stored.append(mem_id)

    return stored


# Spec §2.3 五类记忆名称（来自「AI总结记忆分类及打分」prompt 输出）→ taxonomy main_category
_MEM_TYPE_TO_MAIN = {
    "身份": "身份",
    "身份记忆": "身份",
    "情绪": "情绪",
    "情绪记忆": "情绪",
    "偏好边界": "偏好",
    "偏好边界记忆": "偏好",
    "偏好": "偏好",
    "生活": "生活",
    "生活记忆": "生活",
    "思维": "思维",
    "思维记忆": "思维",
}
