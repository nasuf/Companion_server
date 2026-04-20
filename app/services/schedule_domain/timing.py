"""回复时机模拟服务。

计算回复延迟时间，模拟真人打字节奏。
纯计算，无LLM调用。
"""

from __future__ import annotations

import random

from app.services.mbti import signal as mbti_signal


def calculate_reply_delay(
    message_length: int,
    response_length: int = 50,
    mbti: dict | None = None,
) -> float:
    """计算回复延迟（秒）。

    基于消息长度、MBTI 派生信号、回复长度计算 0.5-5s 延迟。
    spec §1.2 起 thinking_factor 改用 MBTI 的 T 程度×2.0（T 越高思考越慢）。
    """
    base = 0.5 + random.random() * 0.5
    length_factor = min(1.5, message_length / 100)
    thinking_factor = mbti_signal(mbti, "T") * 2.0
    typing_factor = min(1.0, response_length / 100)
    delay = base + length_factor + thinking_factor + typing_factor
    return max(0.5, min(5.0, delay))


def calculate_typing_duration(response_length: int) -> float:
    """计算模拟打字动画时长（秒）。"""
    chars_per_second = 5 + random.random() * 3
    duration = response_length / chars_per_second
    return max(0.5, min(4.0, duration))


def calculate_status_delay(status: str) -> float:
    """根据AI当前状态计算额外延迟（秒）。spec §6.2。

    - idle:      70% 0-3s,   30% 4-6s
    - busy:      60% 3-10s,  40% 10-20s
    - very_busy: 50% 3-20s,  30% 20-30s,   20% 30-60s
    - sleep:     10% 10-30s, 30% 30-120s,  40% 60-300s (1-5min), 20% 300-3600s (5-60min)
    """
    r = random.random()
    if status == "sleep":
        if r < 0.1:
            return random.uniform(10, 30)
        if r < 0.4:
            return random.uniform(30, 120)
        if r < 0.8:
            return random.uniform(60, 300)
        return random.uniform(300, 3600)
    if status == "very_busy":
        if r < 0.5:
            return random.uniform(3, 20)
        if r < 0.8:
            return random.uniform(20, 30)
        return random.uniform(30, 60)
    if status == "busy":
        if r < 0.6:
            return random.uniform(3, 10)
        return random.uniform(10, 20)
    # idle
    if r < 0.7:
        return random.uniform(0, 3)
    return random.uniform(4, 6)


def explain_delay_reason(reason: str, activity: str | None = None, status: str | None = None) -> str:
    """Human-readable delay reason summary for prompt injection."""
    if reason == "conversation_mode":
        return "你们刚刚一直在连续聊天，所以通常会回得更快。"
    if reason == "high_emotion":
        return "用户情绪较强烈，你会倾向于更快接住对方的情绪。"
    if reason == "schedule_sleep":
        return f"收到消息时你正在{activity or '睡觉'}，因此没有立刻看到消息。"
    if reason == "schedule_very_busy":
        return f"收到消息时你正在{activity or '忙事情'}，而且是当天最忙的时段。"
    if reason == "schedule_busy":
        return f"收到消息时你正在{activity or '处理日常安排'}，所以回复被拖后了。"
    return f"收到消息时你在{activity or '安排自己的事情'}，回复节奏受当时状态影响。"
