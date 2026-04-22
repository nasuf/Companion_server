"""主动分享服务。

AI主动发起消息：事件路由 + 频率控制(≤3/日) + 记忆触发。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from app.db import db
from app.redis_client import get_redis
from app.services.llm.models import get_utility_model, invoke_text
from app.services.memory.retrieval.legacy import retrieve_memories, format_memories_for_prompt
from app.services.relationship.emotion import get_ai_emotion
from app.services.prompting.defaults import PROACTIVE_PROMPT
from app.services.prompting.store import get_prompt_text
from app.services.workspace.workspaces import resolve_workspace_id

logger = logging.getLogger(__name__)

MAX_DAILY_PROACTIVE = 3
MAX_2DAY_PROACTIVE = 4  # n≤4: 2天内最多4次主动(含首次)


def _proactive_count_key(agent_id: str, user_id: str) -> str:
    return f"proactive_count:{agent_id}:{user_id}:{datetime.now(UTC).strftime('%Y%m%d')}"


def _proactive_2day_count_key(agent_id: str, user_id: str) -> str:
    return f"proactive_2day:{agent_id}:{user_id}"


async def can_send_proactive(agent_id: str, user_id: str) -> bool:
    """检查今日是否还能发送主动消息。"""
    redis = await get_redis()
    count = await redis.get(_proactive_count_key(agent_id, user_id))
    return int(count or 0) < MAX_DAILY_PROACTIVE


async def can_send_proactive_2day(agent_id: str, user_id: str) -> bool:
    """检查2天滑动窗口内是否还能发送主动消息（≤4次含首次）。"""
    redis = await get_redis()
    count = await redis.get(_proactive_2day_count_key(agent_id, user_id))
    return int(count or 0) < MAX_2DAY_PROACTIVE


async def increment_proactive_count(agent_id: str, user_id: str) -> None:
    """增加今日主动消息计数。"""
    redis = await get_redis()
    key = _proactive_count_key(agent_id, user_id)
    await redis.incr(key)
    await redis.expire(key, 86400)


async def increment_proactive_2day_count(agent_id: str, user_id: str) -> None:
    """增加2天滑动窗口主动消息计数。"""
    redis = await get_redis()
    key = _proactive_2day_count_key(agent_id, user_id)
    await redis.incr(key)
    await redis.expire(key, 172800)  # 48小时


async def generate_proactive_message(
    user_id: str,
    agent_id: str,
) -> str | None:
    """生成主动消息，或None（无内容/超限）。"""
    # 频率控制
    if not await can_send_proactive(agent_id, user_id):
        logger.info(f"Proactive limit reached for agent {agent_id}")
        return None

    try:
        # 获取 agent 信息
        agent = await db.aiagent.find_unique(where={"id": agent_id})
        if not agent:
            return None

        # 获取记忆（跳过语义检索，只用最近+重要记忆）
        memories = await retrieve_memories(
            "",
            user_id,
            semantic_k=0,
            recent_k=5,
            important_k=3,
        )
        memory_strings = format_memories_for_prompt(memories)

        # Proactive 读缓存 PAD（不触发 emotion.ai_pad LLM）
        emotion = await get_ai_emotion(agent_id)
        pleasure = emotion.get("pleasure", 0.0)
        mood = "不错" if pleasure > 0.2 else ("有点低落" if pleasure < -0.2 else "平静")

        prompt = (await get_prompt_text("proactive.message")).format(
            ai_name=agent.name,
            mood=mood,
            pleasure=emotion.get("pleasure", 0),
            arousal=emotion.get("arousal", 0),
            dominance=emotion.get("dominance", 0),
            memories="\n".join(f"- {m}" for m in memory_strings) or "暂无记忆。",
        )

        model = get_utility_model()
        response = await invoke_text(model, prompt)
        response = response.strip()

        if response == "SKIP" or len(response) < 5:
            return None

        # 记录日志
        await increment_proactive_count(agent_id, user_id)
        workspace_id = await resolve_workspace_id(user_id=user_id, agent_id=agent_id)
        await db.proactivechatlog.create(
            data={
                "agent": {"connect": {"id": agent_id}},
                "userId": user_id,
                "workspaceId": workspace_id,
                "message": response,
                "eventType": "scheduled",
            }
        )

        return response

    except Exception as e:
        logger.error(f"Proactive message generation failed: {e}")
        return None


async def get_proactive_history(
    agent_id: str,
    user_id: str,
    limit: int = 10,
    workspace_id: str | None = None,
) -> list[dict]:
    """获取主动消息历史。"""
    effective_workspace_id = workspace_id or await resolve_workspace_id(user_id=user_id, agent_id=agent_id)
    where = {"workspaceId": effective_workspace_id} if effective_workspace_id else {"agentId": agent_id, "userId": user_id}
    logs = await db.proactivechatlog.find_many(
        where=where,
        order={"createdAt": "desc"},
        take=limit,
    )
    return [
        {"content": log.message, "trigger_type": log.eventType, "created_at": str(log.createdAt)}
        for log in logs
    ]
