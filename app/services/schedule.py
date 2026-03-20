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
from app.services.llm.models import get_utility_model, invoke_json
from app.services.trait_model import get_dim, get_seven_dim

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

_LIFE_OVERVIEW_PROMPT = """请根据以下信息，为一位AI朋友生成一份概括性的日常生活画像。这份画像将用于指导AI的每日作息生成，以及回答用户关于AI生活规律的问题。

【AI基本信息】
- 姓名：{name}
- 年龄：{age}
- 职业：{occupation}
- 居住地：{city}

【性格维度】（每个维度0-100）
- 活泼度：{lively}（高分者热情开朗，喜欢分享；低分者安静内敛）
- 理性度：{rational}（高分者逻辑清晰，习惯分析；低分者依赖直觉）
- 感性度：{emotional}（高分者共情能力强，善解人意；低分者冷静直接）
- 计划度：{planned}（高分者喜欢规划，有条理；低分者随性自由）
- 随性度：{spontaneous}（高分者拥抱变化，灵活应变；低分者按部就班）
- 脑洞度：{creative}（高分者思维天马行空；低分者脚踏实地）
- 幽默度：{humor}（高分者风趣幽默；低分者严肃认真）

请生成JSON，包含以下字段：
1. "description": 一段自然语言描述（约200字），概括AI的日常生活模式，包括工作日和周末的典型安排，以及可能的休假活动。描述要符合性格和职业，自然真实，就像AI在介绍自己的生活。
2. "weekday_schedule": 典型工作日时间线，数组，每个元素包含 start（HH:MM）、end（HH:MM）、activity（活动描述）、status（空闲/忙碌/很忙碌/睡眠）。时间段应覆盖全天。
3. "weekend_activities": 周末典型活动列表，数组，每个元素包含 activity（活动名称）、typical_time（常见时间段，如"下午"）、status（通常为空闲）。
4. "holiday_habits": 字符串，描述休假习惯。

要求：活动描述要具体，状态标注合理，整体要体现性格特点。只返回JSON，不要其他内容。"""

_DAILY_SCHEDULE_PROMPT = """根据以下AI角色的生活画像，生成今日作息表。

角色名：{name}
生活画像：{overview}
今日日期：{date}
星期：{weekday}

返回JSON数组，每个时段包含start/end/activity/type：
- type: routine(日常)/work(工作)/rest(休息)/leisure(休闲)/social(社交)/sleep(睡觉)
- 时间格式HH:MM
- 覆盖全天24小时
- 根据星期适当调整（周末可以晚起、多休闲）
- 加入1-2个个性化活动

返回JSON数组（不要其他内容）："""


async def generate_life_overview(
    name: str,
    seven_dim: dict,
    age: int = 22,
    occupation: str | None = None,
    city: str | None = None,
) -> dict:
    """生成AI角色的生活画像（文本+结构化数据）。

    返回 dict: {"description": str, "weekday_schedule": list, "weekend_activities": list, "holiday_habits": str}
    """
    prompt = _LIFE_OVERVIEW_PROMPT.format(
        name=name,
        age=age,
        occupation=occupation or "自由职业",
        city=city or "未设定",
        lively=seven_dim.get("活泼度", 50),
        rational=seven_dim.get("理性度", 50),
        emotional=seven_dim.get("感性度", 50),
        planned=seven_dim.get("计划度", 50),
        spontaneous=seven_dim.get("随性度", 50),
        creative=seven_dim.get("脑洞度", 50),
        humor=seven_dim.get("幽默度", 50),
    )

    model = get_utility_model()
    result = await invoke_json(model, prompt)

    # 确保返回的 dict 包含必需字段
    if not isinstance(result, dict) or "description" not in result:
        raise ValueError(f"Life overview generation returned invalid format: {type(result)}")

    return {
        "description": result.get("description", ""),
        "weekday_schedule": result.get("weekday_schedule", []),
        "weekend_activities": result.get("weekend_activities", []),
        "holiday_habits": result.get("holiday_habits", ""),
    }


async def generate_and_save_life_overview(agent: Any) -> dict:
    """从 agent 对象提取信息，生成并保存生活画像。返回 overview_data dict。"""
    seven_dim = get_seven_dim(agent)
    overview_data = await generate_life_overview(
        agent.name, seven_dim,
        age=agent.age or 22,
        occupation=agent.occupation,
        city=agent.city,
    )
    await save_life_overview(agent.id, overview_data)
    return overview_data


async def generate_daily_schedule(
    agent_id: str,
    name: str,
    seven_dim: dict,
    life_overview: str | None = None,
    date: datetime | None = None,
    user_id: str | None = None,
) -> list[dict]:
    """生成每日作息表。基于生活画像+模板+个性化。"""
    date = date or _local_now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekdays[date.weekday()]

    if life_overview:
        # LLM生成个性化作息（40%概率或周末）
        is_weekend = date.weekday() >= 5
        if is_weekend or random.random() < 0.4:
            try:
                prompt = _DAILY_SCHEDULE_PROMPT.format(
                    name=name,
                    overview=life_overview,
                    date=date.strftime("%Y-%m-%d"),
                    weekday=weekday,
                )
                # 10D.5: 融入用户记忆
                memory_summary = await _get_user_memory_summary(user_id) if user_id else ""
                if memory_summary:
                    prompt += f"\n用户近期情况：{memory_summary}\n可以在空闲时段融入与用户相关的个性化活动。"

                model = get_utility_model()
                schedule = await invoke_json(model, prompt)
                if isinstance(schedule, list) and len(schedule) >= 5:
                    await _cache_schedule(agent_id, date, schedule)
                    return schedule
            except Exception as e:
                logger.warning(f"Custom schedule generation failed: {e}")

    # 默认用模板（根据人格微调）
    schedule = _personalize_template(seven_dim, date)
    await _cache_schedule(agent_id, date, schedule)
    return schedule


def _personalize_template(seven_dim: dict, date: datetime) -> list[dict]:
    """根据七维人格微调基准模板。"""
    schedule = [slot.copy() for slot in _BASE_SCHEDULE_TEMPLATE]
    lively = get_dim(seven_dim, "活泼度")
    planned = get_dim(seven_dim, "计划度")

    is_weekend = date.weekday() >= 5

    for slot in schedule:
        # 活泼度高：更多社交活动
        if slot["type"] == "leisure" and lively >= 0.7:
            if random.random() < 0.4:
                slot["activity"] = random.choice(["和朋友聊天", "刷社交媒体", "出去逛逛"])
                slot["type"] = "social"

        # 计划度高：早起
        if slot["activity"] == "起床洗漱" and planned >= 0.7:
            slot["start"] = "06:30"

        # 周末晚起
        if is_weekend and slot["activity"] == "起床洗漱":
            slot["start"] = "09:00"
        if is_weekend and slot["type"] == "work":
            slot["activity"] = random.choice(["看书", "追剧", "逛街", "玩游戏", "画画"])
            slot["type"] = "leisure"

    return schedule


async def _get_user_memory_summary(user_id: str | None, limit: int = 5) -> str:
    """查询用户L1核心记忆，返回简短摘要文本。"""
    if not user_id:
        return ""
    try:
        from app.services.memory import memory_repo
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
    """缓存当日作息到Redis。"""
    redis = await get_redis()
    await redis.set(_schedule_key(agent_id, date), json.dumps(schedule, ensure_ascii=False), ex=86400 * 2)


async def save_life_overview(agent_id: str, overview_data: dict) -> None:
    """保存生活画像到DB和Redis。

    overview_data: {"description": str, "weekday_schedule": list, "weekend_activities": list, "holiday_habits": str}
    """
    description = overview_data.get("description", "")
    struct_data = {k: overview_data[k] for k in ("weekday_schedule", "weekend_activities", "holiday_habits") if k in overview_data}

    # 先写 DB，成功后才写 Redis（避免 split-brain）
    try:
        await db.aiagent.update(
            where={"id": agent_id},
            data={
                "lifeOverview": description,
                "lifeOverviewData": Json(struct_data),
            },
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

    返回 {"activity": str, "type": str, "status": "idle"|"busy"|"sleep"}
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

    return {"activity": "自由时间", "type": "leisure", "status": "idle"}


def _slot_to_status(slot: dict) -> dict:
    """将时段转为状态。

    4F.3: 区分 busy 和 very_busy:
    - work 类型的核心时段(9-12, 14-17) → very_busy
    - 其他 work/routine → busy
    """
    slot_type = slot.get("type", "leisure")
    if slot_type == "sleep":
        status = "sleep"
    elif slot_type == "work":
        start = slot.get("start", "00:00")
        if "09:00" <= start < "12:00" or "14:00" <= start < "17:00":
            status = "very_busy"
        else:
            status = "busy"
    elif slot_type == "routine":
        status = "busy"
    else:
        status = "idle"

    return {
        "activity": slot.get("activity", ""),
        "type": slot_type,
        "status": status,
    }


_STATUS_LABELS = {
    "idle": "空闲",
    "busy": "忙碌",
    "very_busy": "很忙",
    "sleep": "睡眠中",
}

_TYPE_LABELS = {
    "leisure": "休闲",
    "work": "工作",
    "routine": "日常",
    "sleep": "睡眠",
}


def status_label(status: str) -> str:
    """中文状态标签。"""
    return _STATUS_LABELS.get(status, status)


def type_label(slot_type: str) -> str:
    """中文类型标签。"""
    return _TYPE_LABELS.get(slot_type, slot_type)


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
    seven_dim: dict | None = None,
    adjustment_minutes: int = 0,
) -> dict:
    """4F.1 计算作息调整可行性评分。

    base=50, 加减分:
    - 亲密度>80 → +20
    - 随性度>0.7 → +15
    - 当前sleep → -10
    - 调整幅度>60min → -30
    - 今日已调整≥2次 → -50

    评分区间: <30拒绝, 30-70部分接受, >70接受
    """
    score = 50

    if intimacy_score > 80:
        score += 20

    if seven_dim and get_dim(seven_dim, "随性度") > 0.7:
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
    seven_dim: dict | None = None,
    adjustment_minutes: int = 0,
) -> dict:
    """处理作息调整请求。

    4F.1: 基于可行性评分决定接受/拒绝。
    返回 {"accepted": bool, "response": str, "score": int}
    """
    feasibility = await compute_adjustment_feasibility(
        agent_id=agent_id,
        current_status=current_status,
        intimacy_score=intimacy_score,
        seven_dim=seven_dim,
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


# --- 每日作息回顾 ---

_SCHEDULE_REVIEW_PROMPT = """你是{name}。回顾今天的经历，用第一人称写2-3条简短感想。

今日作息：
{schedule_text}
{adjustments_text}
{chat_summary_text}
要求：
- 用口语化第一人称
- 每条30-50字以内
- 关注感受和体验
- 如有作息调整，提及这些变化
- 如果和用户有聊天，提及互动感受

返回JSON：
{{"memories": ["感想1", "感想2"]}}"""


async def review_daily_schedule(agent_id: str, user_id: str, agent_name: str = "伙伴") -> list[str]:
    """回顾当日作息，合并调整记录，生成AI自我记忆。"""
    from app.services.memory.storage import store_memory

    schedule = await get_cached_schedule(agent_id)
    if not schedule:
        return []

    schedule_text = "\n".join(
        f"{s['start']}-{s['end']} {s['activity']}" for s in schedule
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

    # PRD §9.7: 合并当日聊天摘要
    chat_summary_text = ""
    try:
        redis = await get_redis()
        # 尝试获取最近的summarizer缓存
        keys = await redis.keys(f"summarizer:*")
        for key in keys[:3]:
            data = await redis.get(key)
            if data:
                cached = json.loads(data) if isinstance(data, str) else json.loads(data.decode())
                review = cached.get("review", "")
                if review:
                    chat_summary_text = f"\n今日聊天回顾：\n{review[:200]}"
                    break
    except Exception as e:
        logger.warning(f"Failed to load chat summary for review: {e}")

    prompt = _SCHEDULE_REVIEW_PROMPT.format(
        name=agent_name,
        schedule_text=schedule_text,
        adjustments_text=adjustments_text,
        chat_summary_text=chat_summary_text,
    )

    try:
        result = await invoke_json(get_utility_model(), prompt)
    except Exception as e:
        logger.warning(f"Schedule review failed for agent {agent_id}: {e}")
        return []

    memories = result.get("memories", [])
    stored = []
    for content in memories[:3]:
        if not content or not isinstance(content, str):
            continue
        mem_id = await store_memory(
            user_id=user_id,
            content=content,
            memory_type="life",
            level=3,
            importance=0.8,
            source="ai",
        )
        if mem_id:
            stored.append(mem_id)

    return stored
