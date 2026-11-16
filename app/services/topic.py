"""话题管理服务。

6类话题分类 + Redis话题栈 + 疲劳检测。
纯计算与 Redis 状态；写入用短数据库锁校验会话生命周期，无 LLM 调用。
"""

from __future__ import annotations

import json
import logging

from app.redis_client import get_redis
from app.services.runtime import user_erasure

logger = logging.getLogger(__name__)

# 话题分类
TOPIC_CATEGORIES = ["生活", "兴趣", "情感", "工作", "思想", "社交"]

# 话题关键词映射（轻量级规则分类）
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "生活": ["吃", "饭", "睡", "觉", "天气", "出门", "回家", "做饭", "运动", "健身", "逛街", "买"],
    "兴趣": ["电影", "音乐", "游戏", "书", "旅游", "画", "摄影", "运动", "综艺", "动漫", "追剧"],
    "情感": ["喜欢", "爱", "讨厌", "难过", "开心", "伤心", "焦虑", "压力", "想", "恋爱", "分手"],
    "工作": ["上班", "工作", "公司", "项目", "老板", "同事", "加班", "面试", "升职", "工资"],
    "思想": ["觉得", "认为", "思考", "意义", "价值", "选择", "人生", "未来", "梦想", "目标"],
    "社交": ["朋友", "聚会", "约", "见面", "家人", "父母", "同学", "闺蜜", "兄弟"],
}


def classify_topic(message: str) -> str:
    """基于关键词将消息分类到话题类别。"""
    scores: dict[str, int] = {cat: 0 for cat in TOPIC_CATEGORIES}
    for category, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw in message:
                scores[category] += 1

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "生活"  # 默认生活


# 与重逢感知「大间隔」档 (prompt_builder._REENGAGE_LONG_S) 对齐: 隔 ≥3h 回来,
# 上一个话题的"持续 N 轮"语义已失效.
TOPIC_RESET_GAP_SECONDS = 3 * 3600


async def push_topic(
    conversation_id: str,
    message: str,
    gap_seconds: float | None = None,
) -> dict:
    """分类并推入话题栈，返回当前话题信息。

    gap_seconds ≥ TOPIC_RESET_GAP_SECONDS 时先清空话题栈再入栈：用户隔了
    几小时回来，不清空会让主 prompt 拖着聊天间隔前的旧话题（turns 累计），
    放大"无缝续聊"的错觉；清空后当前消息作为全新话题 turns=1。
    """
    async with user_erasure.conversation_write_scope(conversation_id):
        redis = await get_redis()
        key = f"topics:{conversation_id}"
        if gap_seconds is not None and gap_seconds >= TOPIC_RESET_GAP_SECONDS:
            await redis.delete(key)
        category = classify_topic(message)

        # 检查栈顶是否同一话题类别
        current = await redis.lindex(key, 0)
        if current:
            try:
                top = json.loads(current)
            except (json.JSONDecodeError, TypeError):
                top = None
            if top and top.get("category") == category:
                # 同类话题，增加轮数
                top["turns"] = top.get("turns", 1) + 1
                await redis.lset(key, 0, json.dumps(top, ensure_ascii=False))
                return top

        # 新话题入栈
        topic_name = message[:20]
        entry = {"topic": topic_name, "turns": 1, "category": category}
        await redis.lpush(key, json.dumps(entry, ensure_ascii=False))
        await redis.ltrim(key, 0, 9)  # 保留最近10个
        await redis.expire(key, 86400)  # 24小时过期

        return entry


def detect_topic_fatigue(topic_info: dict, recent_messages: list[str] | None = None) -> bool:
    """检测话题疲劳。

    2I.5: 两个条件取OR:
    - 连续5轮消息长度<10字
    - 连续3轮消息长度<2字
    """
    recent_messages = recent_messages or []

    # 条件1: 连续5轮<10字
    if len(recent_messages) >= 5:
        last5 = recent_messages[-5:]
        if all(len(m) < 10 for m in last5):
            return True

    # 条件2: 连续3轮<2字
    if len(recent_messages) >= 3:
        last3 = recent_messages[-3:]
        if all(len(m) < 2 for m in last3):
            return True

    return False


# format_topic_context 已删除: orchestrator 直传 push_topic 的 dict 给
# prompt_builder, 由 chat.topic_context_section 模板渲染 (registry 管理).
