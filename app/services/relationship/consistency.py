"""长对话一致性服务。

维护50轮上下文窗口 + 话题栈，检测人格漂移。
纯计算+Redis，无LLM调用。
"""

from __future__ import annotations

import json
import logging

from app.redis_client import get_redis

logger = logging.getLogger(__name__)

CONTEXT_WINDOW_SIZE = 50


async def get_context_window(conversation_id: str) -> list[dict]:
    """获取最近50轮对话上下文（从Redis）。"""
    redis = await get_redis()
    key = f"context_window:{conversation_id}"
    data = await redis.lrange(key, 0, CONTEXT_WINDOW_SIZE - 1)
    result = []
    for item in data:
        try:
            result.append(json.loads(item))
        except (json.JSONDecodeError, TypeError):
            continue
    return result


async def push_to_context_window(conversation_id: str, role: str, content: str) -> None:
    """将消息推入上下文窗口。"""
    redis = await get_redis()
    key = f"context_window:{conversation_id}"
    entry = json.dumps({"role": role, "content": content[:500]}, ensure_ascii=False)
    await redis.lpush(key, entry)
    await redis.ltrim(key, 0, CONTEXT_WINDOW_SIZE - 1)
    await redis.expire(key, 86400 * 7)  # 7天过期


def detect_personality_drift(
    responses: list[str],
    personality: dict,
) -> dict:
    """检测人格漂移（基于回复特征分析）。

    返回 {"drifted": bool, "details": str}
    """
    if len(responses) < 5:
        return {"drifted": False, "details": ""}

    e = personality.get("extraversion", 0.5)
    a = personality.get("agreeableness", 0.5)

    # 分析最近回复的特征
    recent = responses[-10:]
    avg_length = sum(len(r) for r in recent) / len(recent)

    # 检测1: 外向性漂移（话多/话少）
    if e >= 0.7 and avg_length < 15:
        return {"drifted": True, "details": "外向性格但回复过短，可能人格漂移"}
    if e <= 0.3 and avg_length > 100:
        return {"drifted": True, "details": "内向性格但回复过长，可能人格漂移"}

    # 检测2: 情绪用词统计
    exclamation_count = sum(r.count("！") + r.count("!") for r in recent)
    exclamation_rate = exclamation_count / len(recent)

    if e >= 0.7 and exclamation_rate < 0.1:
        return {"drifted": True, "details": "外向性格但缺少感叹号"}
    if e <= 0.3 and exclamation_rate > 1.5:
        return {"drifted": True, "details": "内向性格但感叹号过多"}

    return {"drifted": False, "details": ""}


def build_consistency_context(
    context_window: list[dict],
    drift_info: dict,
) -> str | None:
    """构建一致性上下文供Prompt注入。"""
    parts = []

    if drift_info.get("drifted"):
        parts.append(f"注意：{drift_info['details']}，请回到你的性格设定。")

    if not parts:
        return None

    return "\n".join(parts)
