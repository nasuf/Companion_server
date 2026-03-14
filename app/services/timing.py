"""回复时机模拟服务。

计算回复延迟时间，模拟真人打字节奏。
纯计算，无LLM调用。
"""

from __future__ import annotations

import random


def calculate_reply_delay(
    message_length: int,
    personality: dict,
    response_length: int = 50,
) -> float:
    """计算回复延迟（秒）。

    基于消息长度、人格、回复长度计算 0.5-5s 延迟。
    """
    # 基础延迟 0.5-1s
    base = 0.5 + random.random() * 0.5

    # 消息长度因子：长消息需要更多"阅读时间"
    length_factor = min(1.5, message_length / 100)

    # 思考因子：基于人格
    conscientiousness = personality.get("conscientiousness", 0.5)
    thinking_factor = conscientiousness * 1.0  # 认真的人"思考"更久

    # 回复长度因子：长回复需要更多"打字时间"
    typing_factor = min(1.0, response_length / 100)

    delay = base + length_factor + thinking_factor + typing_factor

    # 限制在 0.5-5s
    return max(0.5, min(5.0, delay))


def should_skip_reply(personality: dict) -> bool:
    """判断是否"已读未回"（概率性跳过回复）。

    基于自发性(extraversion)的概率。
    """
    spontaneity = personality.get("extraversion", 0.5)
    # 低外向性 → 更高概率已读未回
    skip_probability = (1.0 - spontaneity) * 0.15
    return random.random() < skip_probability


def calculate_typing_duration(response_length: int) -> float:
    """计算模拟打字动画时长（秒）。"""
    # 大约每秒5-8个字
    chars_per_second = 5 + random.random() * 3
    duration = response_length / chars_per_second
    # 限制在 0.5-4s
    return max(0.5, min(4.0, duration))
