"""亲密度系统。

三维度计算：
- 互动粘性: 对话频率 + 消息量 + 连续活跃天数
- 自我暴露: 情感深度 + 个人话题比例
- 关系时长: 从创建至今的天数

成长亲密度(0-1000): 每日计算，权重0.3/0.3/0.4
话题亲密度(0-100): 每周计算，权重0.2/0.5/0.3
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta

from app.db import db
from app.redis_client import get_redis

logger = logging.getLogger(__name__)

# --- 亲密度等级 ---

INTIMACY_LEVELS = [
    (0, 100, "L1", "初识"),
    (100, 300, "L2", "熟悉"),
    (300, 600, "L3", "亲近"),
    (600, 850, "L4", "信任"),
    (850, 1001, "L5", "挚友"),
]

TOPIC_LEVELS = [
    (0, 20, "浅层", "公共话题"),
    (20, 50, "中层", "个人经历"),
    (50, 80, "深层", "情感与价值观"),
    (80, 101, "核心", "深层秘密与脆弱面"),
]


def get_intimacy_level(score: float) -> dict:
    """返回亲密度等级信息。"""
    for lo, hi, level, label in INTIMACY_LEVELS:
        if lo <= score < hi:
            return {"level": level, "label": label, "score": score}
    return {"level": "L5", "label": "挚友", "score": score}


def get_topic_depth(score: float) -> dict:
    """返回话题深度信息。"""
    for lo, hi, depth, label in TOPIC_LEVELS:
        if lo <= score < hi:
            return {"depth": depth, "label": label, "score": score}
    return {"depth": "核心", "label": "深层秘密与脆弱面", "score": score}


# --- Redis 缓存 ---

def _intimacy_key(agent_id: str, user_id: str) -> str:
    return f"intimacy:{agent_id}:{user_id}"


def _topic_intimacy_key(agent_id: str, user_id: str) -> str:
    return f"topic_intimacy:{agent_id}:{user_id}"


async def get_cached_intimacy(agent_id: str, user_id: str) -> dict | None:
    """获取缓存的亲密度数据。"""
    redis = await get_redis()
    import json
    data = await redis.get(_intimacy_key(agent_id, user_id))
    if data:
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


async def save_intimacy(agent_id: str, user_id: str, data: dict) -> None:
    """保存亲密度到Redis。"""
    import json
    redis = await get_redis()
    await redis.set(
        _intimacy_key(agent_id, user_id),
        json.dumps(data, ensure_ascii=False),
        ex=86400 * 7,
    )


async def save_topic_intimacy(agent_id: str, user_id: str, score: float) -> None:
    """保存话题亲密度到Redis。"""
    redis = await get_redis()
    await redis.set(_topic_intimacy_key(agent_id, user_id), str(score), ex=86400 * 14)


async def get_topic_intimacy(agent_id: str, user_id: str) -> float:
    """获取话题亲密度，默认0。"""
    redis = await get_redis()
    val = await redis.get(_topic_intimacy_key(agent_id, user_id))
    return float(val) if val else 0.0


# --- 维度计算 ---

async def _compute_interaction_stickiness(
    agent_id: str, user_id: str, days: int = 7,
) -> float:
    """计算互动粘性(0-1)。基于最近N天的消息量和活跃天数。"""
    since = datetime.now(UTC) - timedelta(days=days)

    # 获取Agent的对话
    conversations = await db.conversation.find_many(
        where={"userId": user_id, "agentId": agent_id},
    )
    conv_ids = [c.id for c in conversations]
    if not conv_ids:
        return 0.0

    # 统计近N天消息量
    msg_count = await db.message.count(
        where={
            "conversationId": {"in": conv_ids},
            "createdAt": {"gte": since},
        }
    )

    # 消息量归一化：0条=0, 50条=0.5, 100+=1.0
    msg_score = min(1.0, msg_count / 100)

    # 活跃天数：简化为消息量/天均
    daily_avg = msg_count / max(days, 1)
    activity_score = min(1.0, daily_avg / 10)

    return (msg_score * 0.6 + activity_score * 0.4)


async def _compute_self_disclosure(
    user_id: str, days: int = 7,
) -> float:
    """计算自我暴露程度(0-1)。基于情感类记忆的比例。"""
    since = datetime.now(UTC) - timedelta(days=days)

    total = await db.memory.count(
        where={
            "userId": user_id,
            "createdAt": {"gte": since},
            "isArchived": False,
        }
    )
    if total == 0:
        return 0.0

    emotional = await db.memory.count(
        where={
            "userId": user_id,
            "createdAt": {"gte": since},
            "isArchived": False,
            "type": {"in": ["情感", "感受", "关系", "价值观"]},
        }
    )

    return min(1.0, emotional / max(total * 0.5, 1))


def _compute_relationship_duration(created_at: datetime) -> float:
    """计算关系时长因子(0-1)。30天达到0.5，180天达到0.9。"""
    days = (datetime.now(UTC) - created_at).days
    # 对数增长，避免线性
    return min(1.0, math.log1p(days) / math.log1p(180))


# --- 成长亲密度（每日） ---

async def compute_growth_intimacy(agent_id: str, user_id: str) -> float:
    """计算成长亲密度(0-1000)。

    权重：互动粘性0.3 + 自我暴露0.3 + 关系时长0.4
    """
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent:
        return 0.0

    interaction = await _compute_interaction_stickiness(agent_id, user_id)
    disclosure = await _compute_self_disclosure(user_id)
    duration = _compute_relationship_duration(agent.createdAt)

    raw = interaction * 0.3 + disclosure * 0.3 + duration * 0.4
    score = raw * 1000

    # 保存
    data = {
        "growth": round(score, 1),
        "interaction": round(interaction, 3),
        "disclosure": round(disclosure, 3),
        "duration": round(duration, 3),
        "level": get_intimacy_level(score),
    }
    await save_intimacy(agent_id, user_id, data)

    return score


# --- 话题亲密度（每周） ---

async def compute_topic_intimacy(agent_id: str, user_id: str) -> float:
    """计算话题亲密度(0-100)。

    基于近30天记忆的话题深度。
    权重：话题多样性0.2 + 情感深度0.5 + 互动时长0.3
    """
    since = datetime.now(UTC) - timedelta(days=30)

    memories = await db.memory.find_many(
        where={
            "userId": user_id,
            "createdAt": {"gte": since},
            "isArchived": False,
        },
        order={"importance": "desc"},
        take=100,
    )

    if not memories:
        return 0.0

    # 话题多样性：记忆类型种类
    types = {m.type for m in memories if m.type}
    diversity = min(1.0, len(types) / 6)

    # 情感深度：高重要性记忆比例
    high_importance = sum(1 for m in memories if (m.importance or 0) >= 0.7)
    depth = min(1.0, high_importance / max(len(memories) * 0.3, 1))

    # 互动时长
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    duration = _compute_relationship_duration(agent.createdAt) if agent else 0.0

    raw = diversity * 0.2 + depth * 0.5 + duration * 0.3
    score = raw * 100

    await save_topic_intimacy(agent_id, user_id, score)
    return score


# --- API ---

async def get_intimacy_data(agent_id: str, user_id: str) -> dict:
    """获取完整亲密度数据。"""
    cached = await get_cached_intimacy(agent_id, user_id)
    topic = await get_topic_intimacy(agent_id, user_id)

    growth = cached.get("growth", 0) if cached else 0
    level = get_intimacy_level(growth)
    topic_depth = get_topic_depth(topic)

    return {
        "growth_intimacy": growth,
        "topic_intimacy": topic,
        "level": level,
        "topic_depth": topic_depth,
        "details": cached,
    }
