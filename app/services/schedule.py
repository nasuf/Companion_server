"""AI作息系统。

管理AI的生活画像、每日作息表、状态查询和作息调整。
生活画像由LLM生成，作息表基于模板+个性化。
"""

from __future__ import annotations

import json
import logging
import random
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from app.config import settings
from app.db import db
from app.redis_client import get_redis
from app.services.llm.models import get_utility_model, invoke_json, invoke_text
from app.services.trait_model import get_dim

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

_LIFE_OVERVIEW_PROMPT = """根据以下AI角色信息，生成一段简短的生活画像描述（100-150字）。

角色名：{name}
年龄设定：{age}
性格特征：外向性{e:.1f} 宜人性{a:.1f} 开放性{o:.1f} 尽责性{c:.1f} 神经质{n:.1f}

要求：
- 描述这个角色的日常生活习惯、作息偏好
- 包含：起床时间偏好、工作/学习方式、兴趣爱好、社交习惯、睡觉时间
- 用第三人称"ta"
- 自然简洁，像朋友间介绍

生活画像："""

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
    personality: dict,
    age: int = 22,
) -> str:
    """生成AI角色的生活画像文本。"""
    e = personality.get("extraversion", 0.5)
    a = personality.get("agreeableness", 0.5)
    o = personality.get("openness", 0.5)
    c = personality.get("conscientiousness", 0.5)
    n = personality.get("neuroticism", 0.5)

    prompt = _LIFE_OVERVIEW_PROMPT.format(
        name=name, age=age, e=e, a=a, o=o, c=c, n=n,
    )

    model = get_utility_model()
    overview = await invoke_text(model, prompt)
    return overview.strip()


async def generate_daily_schedule(
    agent_id: str,
    name: str,
    personality: dict,
    life_overview: str | None = None,
    date: datetime | None = None,
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
                model = get_utility_model()
                schedule = await invoke_json(model, prompt)
                if isinstance(schedule, list) and len(schedule) >= 5:
                    await _cache_schedule(agent_id, date, schedule)
                    return schedule
            except Exception as e:
                logger.warning(f"Custom schedule generation failed: {e}")

    # 默认用模板（根据人格微调）
    schedule = _personalize_template(personality, date)
    await _cache_schedule(agent_id, date, schedule)
    return schedule


def _personalize_template(personality: dict, date: datetime) -> list[dict]:
    """根据人格微调基准模板。"""
    schedule = [slot.copy() for slot in _BASE_SCHEDULE_TEMPLATE]
    e = personality.get("extraversion", 0.5)
    c = personality.get("conscientiousness", 0.5)
    n = personality.get("neuroticism", 0.5)

    is_weekend = date.weekday() >= 5

    for slot in schedule:
        # 外向型：更多社交活动
        if slot["type"] == "leisure" and e >= 0.7:
            if random.random() < 0.4:
                slot["activity"] = random.choice(["和朋友聊天", "刷社交媒体", "出去逛逛"])
                slot["type"] = "social"

        # 尽责型：早起
        if slot["activity"] == "起床洗漱" and c >= 0.7:
            slot["start"] = "06:30"

        # 周末晚起
        if is_weekend and slot["activity"] == "起床洗漱":
            slot["start"] = "09:00"
        if is_weekend and slot["type"] == "work":
            slot["activity"] = random.choice(["看书", "追剧", "逛街", "玩游戏", "画画"])
            slot["type"] = "leisure"

    return schedule


async def _cache_schedule(agent_id: str, date: datetime, schedule: list[dict]) -> None:
    """缓存当日作息到Redis。"""
    redis = await get_redis()
    await redis.set(_schedule_key(agent_id, date), json.dumps(schedule, ensure_ascii=False), ex=86400 * 2)


async def save_life_overview(agent_id: str, overview: str) -> None:
    """保存生活画像到Redis。"""
    redis = await get_redis()
    await redis.set(f"life_overview:{agent_id}", overview, ex=86400 * 30)


async def get_life_overview(agent_id: str) -> str | None:
    """从Redis获取生活画像。"""
    redis = await get_redis()
    data = await redis.get(f"life_overview:{agent_id}")
    if isinstance(data, bytes):
        return data.decode()
    return data


def _schedule_key(agent_id: str, date: datetime) -> str:
    return f"schedule:{agent_id}:{date.strftime('%Y%m%d')}"


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
        if start in ("09:00", "10:00", "11:00", "14:00", "15:00", "16:00"):
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


# --- 作息查询意图识别 ---

_SCHEDULE_QUERY_KEYWORDS = {
    "current": ["在干嘛", "在做什么", "忙不忙", "有空吗", "做什么呢"],
    "routine": ["几点起", "几点睡", "作息", "日程", "时间表"],
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
    today = _local_now().strftime("%Y%m%d")
    adj_key = f"schedule_adj:{agent_id}:{today}"
    adj_count = int(await redis.get(adj_key) or 0)
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

    if score >= 70:
        # 接受
        redis = await get_redis()
        today = _local_now().strftime("%Y%m%d")
        adj_key = f"schedule_adj:{agent_id}:{today}"
        await redis.incr(adj_key)
        await redis.expire(adj_key, 86400)

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
            redis = await get_redis()
            today = _local_now().strftime("%Y%m%d")
            adj_key = f"schedule_adj:{agent_id}:{today}"
            await redis.incr(adj_key)
            await redis.expire(adj_key, 86400)
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

_SCHEDULE_REVIEW_PROMPT = """你是{name}。回顾今天的作息，用第一人称写1-2条简短感想。

今日作息：
{schedule_text}

要求：
- 用口语化第一人称
- 每条30字以内
- 关注感受和体验

返回JSON：
{{"memories": ["感想1", "感想2"]}}"""


async def review_daily_schedule(agent_id: str, user_id: str, agent_name: str = "伙伴") -> list[str]:
    """回顾当日作息，生成AI自我记忆。"""
    from app.services.memory.storage import store_memory

    schedule = await get_cached_schedule(agent_id)
    if not schedule:
        return []

    schedule_text = "\n".join(
        f"{s['start']}-{s['end']} {s['activity']}" for s in schedule
    )

    prompt = _SCHEDULE_REVIEW_PROMPT.format(
        name=agent_name, schedule_text=schedule_text,
    )

    try:
        result = await invoke_json(get_utility_model(), prompt)
    except Exception as e:
        logger.warning(f"Schedule review failed for agent {agent_id}: {e}")
        return []

    memories = result.get("memories", [])
    stored = []
    for content in memories[:2]:
        if not content or not isinstance(content, str):
            continue
        mem_id = await store_memory(
            user_id=user_id,
            content=content,
            memory_type="生活",
            level=3,
            importance=0.3,
        )
        if mem_id:
            stored.append(mem_id)

    return stored
