"""话题管理服务。

6类话题分类 + Redis话题栈 + 疲劳检测。
纯计算+Redis，无LLM调用（热路径安全）。
"""

from __future__ import annotations

import json
import logging

from app.redis_client import get_redis

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


async def get_topic_stack(conversation_id: str) -> list[dict]:
    """获取当前话题栈（从Redis）。"""
    redis = await get_redis()
    key = f"topics:{conversation_id}"
    data = await redis.lrange(key, 0, 9)  # 最近10个话题
    result = []
    for item in data:
        try:
            result.append(json.loads(item))
        except (json.JSONDecodeError, TypeError):
            continue
    return result


async def push_topic(conversation_id: str, message: str) -> dict:
    """分类并推入话题栈，返回当前话题信息。"""
    redis = await get_redis()
    key = f"topics:{conversation_id}"
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


def detect_topic_fatigue(topic_info: dict, recent_responses: list[str] | None = None) -> bool:
    """检测话题疲劳：连续3轮且回复过短。"""
    if topic_info.get("turns", 0) < 3:
        return False
    if recent_responses:
        short_count = sum(1 for r in recent_responses[-3:] if len(r) < 10)
        return short_count >= 2
    return False


def format_topic_context(current_topic: dict) -> str | None:
    """格式化当前话题上下文供Prompt注入。"""
    if not current_topic:
        return None
    return f"当前话题：{current_topic['category']}（已持续{current_topic['turns']}轮）"
