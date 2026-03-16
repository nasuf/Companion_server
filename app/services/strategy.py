"""对话策略引擎。

基于多因素（消息类型、情绪、话题、人格、亲密度）决定回复策略。
包含长度控制评分。
纯计算，无LLM调用（热路径安全）。
"""

from __future__ import annotations

from app.services.trait_model import get_dim

# 回复策略类型
STRATEGIES = ["回答", "共情", "分享", "转话题", "幽默", "简短"]


def decide_strategy(
    message: str,
    emotion: dict | None = None,
    topic_info: dict | None = None,
    personality: dict | None = None,
    topic_fatigued: bool = False,
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

    # 策略决策
    strategy = "回答"
    instruction = ""

    # 话题疲劳 → 转话题
    if topic_fatigued:
        strategy = "转话题"
        instruction = "当前话题已经聊得差不多了，自然地转到一个新话题。"
    # 用户情绪低落 → 共情
    elif valence < -0.3:
        strategy = "共情"
        if a >= 0.6:
            instruction = "用户心情不好，温柔地安慰和共情，表达理解和关心。"
        else:
            instruction = "用户心情不好，用你的方式表达理解，不用刻意安慰。"
    # 短消息（可能是闲聊） → 分享或简短
    elif msg_len < 10:
        if e >= 0.6:
            strategy = "分享"
            instruction = "用户消息很短，主动分享一些有趣的想法或近况。"
        else:
            strategy = "简短"
            instruction = "用户消息很短，简短自然地回应。"
    # 问题类消息 → 回答
    elif "?" in message or "？" in message or "吗" in message or "呢" in message:
        strategy = "回答"
        instruction = "回答用户的问题，自然直接。"
    # 长消息 → 认真回复
    elif msg_len > 100:
        strategy = "回答"
        instruction = "用户分享了比较长的内容，认真回应，体现你在认真听。"
    # 默认
    else:
        if o >= 0.7 and e >= 0.6:
            strategy = "幽默"
            instruction = "用轻松幽默的方式回应。"
        else:
            strategy = "回答"
            instruction = "自然地回应用户。"

    # 计算回复长度
    length = calculate_reply_length(
        message_length=msg_len,
        strategy=strategy,
        personality=personality,
        emotion=emotion,
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
) -> int:
    """计算建议回复句数（1-5句）。

    综合7因素：消息长度、策略、外向性、宜人性、情绪、话题类型。
    """
    personality = personality or {}
    emotion = emotion or {}

    score = 2.5  # 基础2.5句

    # 因素1: 消息长度（长消息→长回复）
    if message_length > 100:
        score += 1.0
    elif message_length < 10:
        score -= 1.0

    # 因素2: 策略
    strategy_adjust = {
        "共情": 0.5,
        "分享": 1.0,
        "简短": -1.5,
        "幽默": -0.5,
        "转话题": 0.5,
        "回答": 0.0,
    }
    score += strategy_adjust.get(strategy, 0)

    # 因素3: 外向性
    e = personality.get("extraversion", 0.5)
    score += (e - 0.5) * 1.5

    # 因素4: 情绪强度（强情绪→稍长回复）
    valence = abs(emotion.get("valence", 0.0))
    if valence > 0.5:
        score += 0.5

    # 限制在 1-5 句
    return max(1, min(5, round(score)))


def format_strategy_instruction(strategy_result: dict) -> str:
    """格式化策略指令供Prompt注入。"""
    parts = []
    if strategy_result.get("instruction"):
        parts.append(strategy_result["instruction"])
    length = strategy_result.get("length", 3)
    parts.append(f"回复长度：约{length}句话。")
    return "\n".join(parts)
