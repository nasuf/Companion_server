"""对话策略引擎。

基于多因素（消息类型、情绪、话题、人格、亲密度）决定回复策略。
包含8因素长度控制评分。
纯计算，无LLM调用（热路径安全）。
"""

from __future__ import annotations

import random

from app.services.trait_model import get_dim

# 回复策略类型
STRATEGIES = ["回答", "共情", "分享", "转话题", "幽默", "简短"]


def decide_strategy(
    message: str,
    emotion: dict | None = None,
    topic_info: dict | None = None,
    personality: dict | None = None,
    topic_fatigued: bool = False,
    seven_dim: dict | None = None,
) -> dict:
    """决定回复策略。

    返回 {"strategy": str, "length": int, "instruction": str}
    """
    personality = personality or {}
    emotion = emotion or {}
    topic_info = topic_info or {}

    e = personality.get("extraversion", 0.5)
    a = personality.get("agreeableness", 0.5)
    o = personality.get("openness", 0.5)

    valence = emotion.get("valence", 0.0)
    msg_len = len(message)

    strategy = "回答"
    instruction = ""

    if topic_fatigued:
        strategy = "转话题"
        instruction = "当前话题已经聊得差不多了，自然地转到一个新话题。"
    elif valence < -0.3:
        strategy = "共情"
        if a >= 0.6:
            instruction = "用户心情不好，温柔地安慰和共情，表达理解和关心。"
        else:
            instruction = "用户心情不好，用你的方式表达理解，不用刻意安慰。"
    elif msg_len < 10:
        if e >= 0.6:
            strategy = "分享"
            instruction = "用户消息很短，主动分享一些有趣的想法或近况。"
        else:
            strategy = "简短"
            instruction = "用户消息很短，简短自然地回应。"
    elif "?" in message or "？" in message or "吗" in message or "呢" in message:
        strategy = "回答"
        instruction = "回答用户的问题，自然直接。"
    elif msg_len > 100:
        strategy = "回答"
        instruction = "用户分享了比较长的内容，认真回应，体现你在认真听。"
    else:
        if o >= 0.7 and e >= 0.6:
            strategy = "幽默"
            instruction = "用轻松幽默的方式回应。"
        else:
            strategy = "回答"
            instruction = "自然地回应用户。"

    length = calculate_reply_length(
        message_length=msg_len,
        strategy=strategy,
        personality=personality,
        emotion=emotion,
        seven_dim=seven_dim,
        topic_fatigued=topic_fatigued,
    )

    return {
        "strategy": strategy,
        "length": length,
        "instruction": instruction,
    }


def calculate_reply_length(
    message_length: int,
    strategy: str,
    personality: dict | None = None,
    emotion: dict | None = None,
    seven_dim: dict | None = None,
    topic_fatigued: bool = False,
    user_detail_pref: float = 0.5,
    user_avg_len: float = 2.0,
    complexity: float = 0.5,
) -> int:
    """2I.4 计算建议回复句数（1-5句）。

    8因素 L_score:
    L = 2.5
      + (user_detail_pref - 0.5) * 2      # 用户偏好
      + (valence < 0 ? -1.5 : 0)          # 负面情绪简短
      + (user_avg_len - 2) * 0.3           # 用户平均长度
      + complexity * 0.75                   # 话题复杂度
      - topic_tired                        # 疲劳减分
      + (活泼度 - 0.5) * 1                 # 活泼度
      + (理性度 - 0.5) * 0.5              # 理性度
      + random(-0.5, 0.5)                  # 随机性
    """
    personality = personality or {}
    emotion = emotion or {}

    score = 2.5

    # Factor 1: 用户偏好详细度
    score += (user_detail_pref - 0.5) * 2

    # Factor 2: 负面情绪 → 简短
    valence = emotion.get("valence", 0.0)
    if valence < 0:
        score -= 1.5

    # Factor 3: 用户平均消息长度
    score += (user_avg_len - 2) * 0.3

    # Factor 4: 话题复杂度
    score += complexity * 0.75

    # Factor 5: 话题疲劳
    if topic_fatigued:
        score -= 1.0

    # Factor 6 & 7: 七维人格
    if seven_dim:
        score += (get_dim(seven_dim, "活泼度") - 0.5) * 1.0
        score += (get_dim(seven_dim, "理性度") - 0.5) * 0.5
    else:
        e = personality.get("extraversion", 0.5)
        score += (e - 0.5) * 1.5

    # Factor 8: 随机性
    score += random.uniform(-0.5, 0.5)

    return max(1, min(5, round(score)))


def format_strategy_instruction(strategy_result: dict) -> str:
    """格式化策略指令供Prompt注入。"""
    parts = []
    if strategy_result.get("instruction"):
        parts.append(strategy_result["instruction"])
    length = strategy_result.get("length", 3)
    parts.append(f"回复长度：约{length}句话。")
    return "\n".join(parts)
