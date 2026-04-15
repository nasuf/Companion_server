"""亲密度系统。

三维度计算：
- 互动粘性: 对话频率 + 消息量 + 连续活跃天数
- 自我暴露: 情感深度 + 个人话题比例
- 关系时长: 从创建至今的天数

成长亲密度(0-1000): 每日计算，权重0.3/0.3/0.4
话题亲密度(0-100): 每周计算，权重0.2/0.5/0.3
"""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime, timedelta

from app.db import db
from app.redis_client import get_redis
from app.services.workspace.workspaces import resolve_workspace_id

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

# PRD §4.6.2.1: 亲密度阶段标签 (基于topic_intimacy 0-100)
RELATIONSHIP_STAGES = [
    (0, 31, "普通朋友"),
    (31, 61, "好朋友"),
    (61, 86, "挚友"),
    (86, 101, "灵魂伴侣"),
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


def get_relationship_stage(score: float) -> str:
    """根据话题亲密度(0-100)返回PRD关系阶段标签。"""
    for lo, hi, stage in RELATIONSHIP_STAGES:
        if lo <= score < hi:
            return stage
    return "灵魂伴侣"


# --- Redis 缓存 ---

def _intimacy_key(agent_id: str, user_id: str) -> str:
    return f"intimacy:{agent_id}:{user_id}"


def _topic_intimacy_key(agent_id: str, user_id: str) -> str:
    return f"topic_intimacy:{agent_id}:{user_id}"


async def get_cached_intimacy(agent_id: str, user_id: str) -> dict | None:
    """获取缓存的亲密度数据。"""
    redis = await get_redis()
    data = await redis.get(_intimacy_key(agent_id, user_id))
    if data:
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


async def save_intimacy(agent_id: str, user_id: str, data: dict) -> None:
    """保存亲密度到Redis。"""
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
    agent_id: str, user_id: str, days: int = 30,
) -> float:
    """G1 互动粘性(0-1)。PRD §8.3.1: x=近30天平均每日对话轮数, G1=min(1000, 200*log10(x+1))/1000"""
    since = datetime.now(UTC) - timedelta(days=days)

    conversations = await db.conversation.find_many(
        where={"userId": user_id, "agentId": agent_id},
    )
    conv_ids = [c.id for c in conversations]
    if not conv_ids:
        return 0.0

    msg_count = await db.message.count(
        where={
            "conversationId": {"in": conv_ids},
            "createdAt": {"gte": since},
        }
    )

    # PRD §8.3.1: x = 近30天平均每日对话轮数
    daily_avg = msg_count / days
    return min(1000, 200 * math.log10(daily_avg + 1)) / 1000


async def _compute_self_disclosure(
    user_id: str,
    workspace_id: str | None = None,
    *,
    cutoff: datetime | None = None,
) -> float:
    """G2 自我暴露(0-1)。基于L1+L2记忆重要度总和的对数公式。

    min(1000, 200*log10(importance_sum+1)) / 1000

    Per spec §3.4.1:
      - 每日值（成长亲密度）: 截止到当日结束时累计
      - 每周值（话题亲密度）: 截止到当周周日结束时累计

    `cutoff` 给调用方一个显式截止点。None 表示"现在"，等同于历史行为。
    importance 在写入时确定后不会随 level 衰减，所以 cutoff 在表上对应
    的就是 createdAt ≤ cutoff 的子集。
    """
    workspace_id = workspace_id or await resolve_workspace_id(user_id=user_id)
    if cutoff is None:
        result = await db.query_raw(
            """
            SELECT COALESCE(SUM(importance), 0) as total_importance
            FROM memories_user
            WHERE user_id = $1 AND workspace_id = $2
              AND level IN (1, 2) AND is_archived = false
            """,
            user_id,
            workspace_id,
        )
    else:
        result = await db.query_raw(
            """
            SELECT COALESCE(SUM(importance), 0) as total_importance
            FROM memories_user
            WHERE user_id = $1 AND workspace_id = $2
              AND level IN (1, 2) AND is_archived = false
              AND created_at <= $3
            """,
            user_id,
            workspace_id,
            cutoff,
        )

    importance_sum = float(result[0]["total_importance"]) if result else 0.0

    return min(1000, 200 * math.log10(importance_sum + 1)) / 1000


def _compute_relationship_duration(created_at: datetime) -> float:
    """G3 关系时长(0-1)。Sigmoid曲线: min(1000, 1000/(1+e^(-0.1*(days-30)))) / 1000

    t=0→18, t=30→500, t=60→952, t=90→998
    """
    days = (datetime.now(UTC) - created_at).days
    return min(1000, 1000 / (1 + math.exp(-0.1 * (days - 30)))) / 1000


# --- 成长亲密度（每日） ---

async def compute_growth_intimacy(
    agent_id: str, user_id: str, created_at: datetime | None = None,
) -> float:
    """计算成长亲密度(0-1000)。

    spec §3.4.5: 每日凌晨基于"前一日结束时"的快照计算。
    权重：互动粘性 0.3 + 自我暴露 0.3 + 关系时长 0.4
    """
    workspace_id = await resolve_workspace_id(user_id=user_id, agent_id=agent_id)
    if not created_at:
        agent = await db.aiagent.find_unique(where={"id": agent_id})
        if not agent:
            return 0.0
        created_at = agent.createdAt

    # spec §3.4.1 daily snapshot: 截止到前一日结束时（= 今日 00:00 UTC）
    now = datetime.now(UTC)
    daily_cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)

    interaction = await _compute_interaction_stickiness(agent_id, user_id)
    disclosure = await _compute_self_disclosure(
        user_id, workspace_id=workspace_id, cutoff=daily_cutoff,
    )
    duration = _compute_relationship_duration(created_at)
    # Note: G1/G2/G3 all return 0-1 range via /1000 normalization

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

async def compute_topic_intimacy(
    agent_id: str, user_id: str, created_at: datetime | None = None,
) -> float:
    """计算话题亲密度(0-100)。

    spec §3.4.5: 与成长亲密度共用同一组三维度（互动粘性 / 自我暴露 / 关系
    时长），仅权重和归一化不同：
      话题亲密度 = round(T1 * 0.2 + T2 * 0.5 + T3 * 0.3)
      其中 T1/T2/T3 = round(G1/G2/G3 * 100)
    """
    workspace_id = await resolve_workspace_id(user_id=user_id, agent_id=agent_id)
    if not created_at:
        agent = await db.aiagent.find_unique(where={"id": agent_id})
        if not agent:
            return 0.0
        created_at = agent.createdAt

    # spec §3.4.1 weekly snapshot: 截止到当周周日结束时
    # job 在周日凌晨跑，"当周周日结束" = job 触发当下，cutoff = now
    g1 = await _compute_interaction_stickiness(agent_id, user_id)
    g2 = await _compute_self_disclosure(
        user_id, workspace_id=workspace_id, cutoff=datetime.now(UTC),
    )
    g3 = _compute_relationship_duration(created_at)

    # T = round(G * 100), 范围 0-100
    t1 = round(g1 * 100)
    t2 = round(g2 * 100)
    t3 = round(g3 * 100)

    # spec 权重: 互动粘性 20% + 自我暴露 50% + 关系时长 30%
    score = round(t1 * 0.2 + t2 * 0.5 + t3 * 0.3)

    await save_topic_intimacy(agent_id, user_id, score)
    return float(score)


# --- API ---

async def get_intimacy_data(agent_id: str, user_id: str) -> dict:
    """获取完整亲密度数据。"""
    cached = await get_cached_intimacy(agent_id, user_id)
    topic = await get_topic_intimacy(agent_id, user_id)

    growth = cached.get("growth", 0) if cached else 0
    level = cached.get("level") if cached else get_intimacy_level(growth)
    topic_depth = get_topic_depth(topic)

    return {
        "growth_intimacy": growth,
        "topic_intimacy": topic,
        "level": level,
        "topic_depth": topic_depth,
    }
