"""回复时机模拟服务。

计算回复延迟时间，模拟真人打字节奏。
纯计算，无LLM调用。
"""

from __future__ import annotations

import random

from app.services.trait_model import get_dim


def calculate_reply_delay(
    message_length: int,
    personality: dict,
    response_length: int = 50,
    seven_dim: dict | None = None,
) -> float:
    """计算回复延迟（秒）。

    基于消息长度、人格、回复长度计算 0.5-5s 延迟。
    2I.1: thinking_factor 改用 理性度×2.0
    """
    base = 0.5 + random.random() * 0.5
    length_factor = min(1.5, message_length / 100)

    # 2I.1: 思考因子改用理性度×2.0
    if seven_dim:
        thinking_factor = get_dim(seven_dim, "理性度") * 2.0
    else:
        conscientiousness = personality.get("conscientiousness", 0.5)
        thinking_factor = conscientiousness * 1.0

    typing_factor = min(1.0, response_length / 100)
    delay = base + length_factor + thinking_factor + typing_factor

    return max(0.5, min(5.0, delay))


def should_skip_reply(personality: dict, seven_dim: dict | None = None) -> bool:
    """判断是否"已读未回"。

    2I.2: 随性度×0.2（高随性 → 更高概率已读未回）
    """
    if seven_dim:
        skip_probability = get_dim(seven_dim, "随性度") * 0.2
    else:
        spontaneity = personality.get("extraversion", 0.5)
        skip_probability = (1.0 - spontaneity) * 0.15
    return random.random() < skip_probability


def should_delay_roundtrip(seven_dim: dict | None = None) -> float | None:
    """2I.3 回马枪延迟：随性度≥0.7时10%概率，返回延迟秒数(30min-2hr)，否则None。"""
    if not seven_dim:
        return None
    spontaneous = get_dim(seven_dim, "随性度")
    if spontaneous >= 0.7 and random.random() < 0.1:
        return random.uniform(1800, 7200)  # 30min - 2hr
    return None


def calculate_typing_duration(response_length: int) -> float:
    """计算模拟打字动画时长（秒）。"""
    chars_per_second = 5 + random.random() * 3
    duration = response_length / chars_per_second
    return max(0.5, min(4.0, duration))


def calculate_status_delay(status: str) -> float:
    """根据AI当前状态计算额外延迟（秒）。

    2I.7: 概率分层
    - idle: 70% 即回(0-1s), 30% 稍延(1-3s)
    - busy: 60% 短延(30-60s), 30% 中延(60-180s), 10% 长延(180-300s)
    - sleep: 40% 浅睡(300-1800s), 30% 中睡(1800-7200s), 20% 深睡(7200-10800s), 10% 熟睡(10800-14400s)
    """
    r = random.random()
    if status == "sleep":
        if r < 0.4:
            return random.uniform(300, 1800)
        elif r < 0.7:
            return random.uniform(1800, 7200)
        elif r < 0.9:
            return random.uniform(7200, 10800)
        else:
            return random.uniform(10800, 14400)
    elif status == "very_busy":
        # 4F.3: 非常忙碌（工作高峰）→ 更长延迟
        if r < 0.5:
            return random.uniform(60, 180)
        elif r < 0.8:
            return random.uniform(180, 300)
        else:
            return random.uniform(300, 600)
    elif status == "busy":
        if r < 0.6:
            return random.uniform(30, 60)
        elif r < 0.9:
            return random.uniform(60, 180)
        else:
            return random.uniform(180, 300)
    else:  # idle
        if r < 0.7:
            return random.uniform(0, 1)
        else:
            return random.uniform(1, 3)


def compute_message_interval_delay(
    last_message_age_seconds: float,
    ai_emotion: dict | None = None,
    current_status: str = "idle",
) -> float:
    """4F.2 消息间隔感知延迟。

    - <30min → 1-5s (正常聊天)
    - ≥30min → 查情绪(高兴→90%即回) → 查作息状态
    """
    if last_message_age_seconds < 1800:  # <30min
        return random.uniform(1, 5)

    # ≥30min gap: check emotion
    if ai_emotion:
        valence = ai_emotion.get("valence", 0.0)
        if valence > 0.3 and random.random() < 0.9:
            return random.uniform(1, 3)  # happy → quick reply

    # Fall back to status-based delay
    return calculate_status_delay(current_status)
