"""User portrait generation service.

Generates and updates 200-300 character user portraits from L1/L2 memories.
Runs weekly as a scheduled job.
"""

import logging
from datetime import UTC, datetime, timedelta

from app.db import db
from app.services.llm.models import get_utility_model, invoke_text

logger = logging.getLogger(__name__)

PORTRAIT_GENERATION_PROMPT = """你是一个用户画像生成系统。请根据以下用户记忆，生成一份200-300字的用户画像。

用户记忆：
{memories}

画像结构要求（共5段）：
1. 基本身份（姓名、年龄、性别、职业等已知信息）
2. 主要偏好与禁忌（从L1、L2偏好记忆提取）
3. 生活状态与重要事件
4. 性格特征与交流风格
5. 情感倾向与关注点

规则：
- 只使用记忆中明确提到的信息，不要推测
- 未知信息用"未知"标注
- 语言简洁客观，不加评价
- 总字数200-300字"""

PORTRAIT_UPDATE_PROMPT = """你是一个用户画像更新系统。请根据上一版画像和本周新增变化，生成更新后的画像。

上一版画像：
{previous_portrait}

本周变化摘要：
{weekly_changes}

规则：
- 保留未变化的信息
- 更新有变化的部分
- 新增新发现的信息
- 删除被用户否定的旧信息
- 总字数200-300字
- 保持5段结构"""


async def check_portrait_preconditions(user_id: str, agent_id: str) -> bool:
    """检查首次画像生成前置条件。

    条件:
    - 注册≥24h
    - L2记忆≥20条
    - L1记忆≥5条
    """
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent:
        return False

    hours_since_creation = (datetime.now(UTC) - agent.createdAt.replace(
        tzinfo=UTC if agent.createdAt.tzinfo is None else agent.createdAt.tzinfo
    )).total_seconds() / 3600
    if hours_since_creation < 24:
        logger.info(f"Portrait precondition: agent {agent_id} created <24h ago")
        return False

    l2_count = await db.memory.count(
        where={"userId": user_id, "level": 2, "isArchived": False},
    )
    if l2_count < 20:
        logger.info(f"Portrait precondition: only {l2_count} L2 memories (need 20)")
        return False

    l1_count = await db.memory.count(
        where={"userId": user_id, "level": 1, "isArchived": False},
    )
    if l1_count < 5:
        logger.info(f"Portrait precondition: only {l1_count} L1 memories (need 5)")
        return False

    return True


async def generate_portrait(user_id: str, agent_id: str) -> str | None:
    """Generate a user portrait from L1/L2 memories."""
    # Check preconditions for first-time generation
    existing = await db.userportrait.find_first(
        where={"userId": user_id, "agentId": agent_id},
    )
    if not existing:
        if not await check_portrait_preconditions(user_id, agent_id):
            return None

    memories = await db.memory.find_many(
        where={
            "userId": user_id,
            "level": {"in": [1, 2]},
            "isArchived": False,
        },
        order={"importance": "desc"},
        take=30,
    )

    if not memories:
        logger.info(f"No L1/L2 memories for user {user_id}, skipping portrait")
        return None

    memories_text = "\n".join(
        f"- [L{m.level}] {m.summary or m.content}" for m in memories
    )

    prompt = PORTRAIT_GENERATION_PROMPT.format(memories=memories_text)

    try:
        portrait = await invoke_text(get_utility_model(), prompt)
    except Exception as e:
        logger.error(f"Portrait generation failed: {e}")
        return None

    # Store portrait
    existing = await db.userportrait.find_first(
        where={"userId": user_id, "agentId": agent_id},
        order={"version": "desc"},
    )
    version = (existing.version + 1) if existing else 1

    await db.userportrait.create(
        data={
            "user": {"connect": {"id": user_id}},
            "agentId": agent_id,
            "version": version,
            "content": portrait,
        }
    )

    logger.info(f"Generated portrait v{version} for user {user_id}")
    return portrait


async def update_portrait_weekly(user_id: str, agent_id: str) -> str | None:
    """Update user portrait based on weekly memory changes."""
    # Get previous portrait
    previous = await db.userportrait.find_first(
        where={"userId": user_id, "agentId": agent_id},
        order={"version": "desc"},
    )

    if not previous:
        return await generate_portrait(user_id, agent_id)

    # Get this week's changelog
    one_week_ago = datetime.now(UTC) - timedelta(days=7)

    changelogs = await db.memorychangelog.find_many(
        where={
            "userId": user_id,
            "createdAt": {"gte": one_week_ago},
        },
        order={"createdAt": "desc"},
        take=50,
    )

    if not changelogs:
        logger.info(f"No changes this week for user {user_id}, keeping portrait")
        return previous.content

    changes_text = "\n".join(
        f"- [{cl.operation}] {cl.newValue or cl.oldValue or ''}"
        for cl in changelogs
    )

    prompt = PORTRAIT_UPDATE_PROMPT.format(
        previous_portrait=previous.content,
        weekly_changes=changes_text,
    )

    try:
        portrait = await invoke_text(get_utility_model(), prompt)
    except Exception as e:
        logger.error(f"Portrait update failed: {e}")
        return previous.content

    await db.userportrait.create(
        data={
            "user": {"connect": {"id": user_id}},
            "agentId": agent_id,
            "version": previous.version + 1,
            "content": portrait,
        }
    )

    logger.info(f"Updated portrait v{previous.version + 1} for user {user_id}")
    return portrait


async def get_latest_portrait(user_id: str, agent_id: str) -> str | None:
    """Get the latest portrait for a user-agent pair."""
    portrait = await db.userportrait.find_first(
        where={"userId": user_id, "agentId": agent_id},
        order={"version": "desc"},
    )
    return portrait.content if portrait else None
