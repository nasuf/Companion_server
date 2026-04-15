"""话题管理服务。

6类话题分类 + Redis话题栈 + 疲劳检测 + 话题推荐。
纯计算+Redis，无LLM调用（热路径安全）。
"""

from __future__ import annotations

import json
import logging
import random

from app.redis_client import get_redis
from app.services.mbti import signal as mbti_signal

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


def format_topic_context(current_topic: dict) -> str | None:
    """格式化当前话题上下文供Prompt注入。"""
    if not current_topic:
        return None
    return f"当前话题：{current_topic['category']}（已持续{current_topic['turns']}轮）"


# --- 2I.6 话题推荐 ---

# 话题库：按亲密度等级和人格倾向分类
TOPIC_POOL: dict[str, list[str]] = {
    "生活_casual": [
        "最近有看什么好看的剧吗", "今天天气真不错", "你中午吃了什么",
        "周末有什么安排", "最近有没有发现什么好吃的店",
    ],
    "兴趣_casual": [
        "你最近在听什么歌", "有没有推荐的电影", "你玩游戏吗",
        "最近有什么好书推荐", "你喜欢什么运动",
    ],
    "生活_deep": [
        "你觉得理想的生活是什么样的", "如果可以去任何地方旅行你想去哪",
        "你有什么一直想做但还没做的事", "你觉得什么时候最放松",
    ],
    "情感_deep": [
        "你觉得什么是好的友谊", "你最近有什么烦恼吗",
        "有没有什么事让你特别感动", "你觉得孤独是什么感觉",
    ],
    "思想_deep": [
        "你觉得人生中最重要的是什么", "你对未来有什么期待",
        "你觉得成功的定义是什么", "你相信命运吗",
    ],
    "幽默": [
        "你知道什么冷笑话吗", "如果你是一种动物你觉得自己是什么",
        "你觉得世界上最无用的发明是什么", "你有什么奇怪的小习惯",
    ],
    "脑洞": [
        "如果你能拥有一种超能力你想要什么", "你觉得外星人存在吗",
        "如果可以穿越到任何时代你想去哪里", "你觉得AI会有感情吗",
    ],
}

# MBTI 字母 → 话题偏好权重 (spec §1.2: 直接 MBTI, 不经派生信号)
# 注: "humor" 作为 (E + N)/2 的复合在循环里另算。
_MBTI_TOPIC_WEIGHTS: dict[str, dict[str, float]] = {
    "E": {"生活_casual": 0.3, "兴趣_casual": 0.3, "幽默": 0.4},
    "T": {"思想_deep": 0.5, "生活_deep": 0.3, "兴趣_casual": 0.2},
    "F": {"情感_deep": 0.5, "生活_deep": 0.3, "生活_casual": 0.2},
    "N": {"脑洞": 0.5, "思想_deep": 0.3, "幽默": 0.2},
}
_HUMOR_TOPIC_WEIGHTS: dict[str, float] = {
    "幽默": 0.5, "生活_casual": 0.3, "兴趣_casual": 0.2,
}

# 亲密度等级 → 可用话题深度
_INTIMACY_ALLOWED: dict[str, list[str]] = {
    "L1": ["生活_casual", "兴趣_casual"],
    "L2": ["生活_casual", "兴趣_casual", "幽默", "脑洞"],
    "L3": ["生活_casual", "兴趣_casual", "幽默", "脑洞", "生活_deep"],
    "L4": ["生活_casual", "兴趣_casual", "幽默", "脑洞", "生活_deep", "情感_deep"],
    "L5": ["生活_casual", "兴趣_casual", "幽默", "脑洞", "生活_deep", "情感_deep", "思想_deep"],
}


def select_new_topic(
    mbti: dict | None = None,
    intimacy_level: str = "L1",
    recent_topics: list[str] | None = None,
) -> str:
    """2I.6 基于人格偏好+亲密度等级推荐新话题。spec §1.2 使用 MBTI。

    Args:
        mbti: 当前 effective MBTI dict（None 时所有类别等权）
        intimacy_level: 亲密度等级 L1-L5
        recent_topics: 最近使用过的话题（用于去重）

    Returns:
        推荐话题文本
    """
    recent_topics = recent_topics or []

    # 1. 确定可用话题池
    allowed_categories = _INTIMACY_ALLOWED.get(intimacy_level, _INTIMACY_ALLOWED["L1"])

    # 2. 计算各话题类别权重
    category_weights: dict[str, float] = {cat: 1.0 for cat in allowed_categories}

    if mbti:
        for letter, topic_prefs in _MBTI_TOPIC_WEIGHTS.items():
            sig_val = mbti_signal(mbti, letter)
            for cat, weight in topic_prefs.items():
                if cat in category_weights:
                    category_weights[cat] += sig_val * weight
        # humor 是 (E + N)/2 的复合, 单独算
        humor = (mbti_signal(mbti, "E") + mbti_signal(mbti, "N")) / 2
        for cat, weight in _HUMOR_TOPIC_WEIGHTS.items():
            if cat in category_weights:
                category_weights[cat] += humor * weight

    # 3. 按权重选择类别
    categories = list(category_weights.keys())
    weights = [category_weights[c] for c in categories]
    total = sum(weights)
    weights = [w / total for w in weights]

    chosen_category = random.choices(categories, weights=weights, k=1)[0]

    # 4. 从类别中选话题，避免最近用过的
    pool = TOPIC_POOL.get(chosen_category, TOPIC_POOL["生活_casual"])
    available = [t for t in pool if t not in recent_topics]
    if not available:
        available = pool

    return random.choice(available)
