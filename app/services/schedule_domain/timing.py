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
    thinking_factor = mbti_signal(mbti, "rational") * 2.0
    typing_factor = min(1.0, response_length / 100)
    delay = base + length_factor + thinking_factor + typing_factor
    return max(0.5, min(5.0, delay))


def calculate_typing_duration(response_length: int) -> float:
    """计算模拟打字动画时长（秒）。"""
    chars_per_second = 5 + random.random() * 3
    duration = response_length / chars_per_second
    return max(0.5, min(4.0, duration))


def calculate_status_delay(status: str) -> float:
    """根据AI当前状态计算额外延迟（秒）。PRD §6.1.2.2。

    - idle: 70% 0-3s, 30% 3-10s
    - busy: 60% 0-60s, 30% 60-180s, 10% 180-300s
    - very_busy: 50% 0-180s, 30% 180-300s, 20% 300-600s
    - sleep: 40% 0-1800s, 30% 1800-3600s, 20% 3600-7200s, 10% 7200-14400s
    """
    r = random.random()
    if status == "sleep":
        if r < 0.4:
            return random.uniform(0, 1800)
        elif r < 0.7:
            return random.uniform(1800, 3600)
        elif r < 0.9:
            return random.uniform(3600, 7200)
        else:
            return random.uniform(7200, 14400)
    elif status == "very_busy":
        if r < 0.5:
            return random.uniform(0, 180)
        elif r < 0.8:
            return random.uniform(180, 300)
        else:
            return random.uniform(300, 600)
    elif status == "busy":
        if r < 0.6:
            return random.uniform(0, 60)
        elif r < 0.9:
            return random.uniform(60, 180)
        else:
            return random.uniform(180, 300)
    else:  # idle
        if r < 0.7:
            return random.uniform(0, 3)
        else:
            return random.uniform(3, 10)


def compute_message_interval_delay(
    last_message_age_seconds: float,
    ai_emotion: dict | None = None,
    current_status: str = "idle",
) -> float:
    """消息间隔感知延迟（PRD §6.2.1.2）。

    - <30min（交流状态）→ 1-5s
    - ≥30min + 高情绪(arousal>0.6 & pleasure<0.4) → 90% 0-5s, 10% 60-180s
    - ≥30min + 高兴(pleasure>0.3) → 90% 1-3s
    - 其他 → 按作息状态延迟
    """
    if last_message_age_seconds < 1800:  # 交流状态
        return random.uniform(1, 5)

    # 高情绪状态: arousal>0.6 且 pleasure<0.4
    if ai_emotion:
        arousal = ai_emotion.get("arousal", 0.0)
        pleasure = ai_emotion.get("pleasure", 0.0)
        if arousal > 0.6 and pleasure < 0.4:
            if random.random() < 0.9:
                return random.uniform(0, 5)
            return random.uniform(60, 180)

    # 高兴快速回复
    if ai_emotion:
        if ai_emotion.get("pleasure", 0.0) > 0.3 and random.random() < 0.9:
            return random.uniform(1, 3)

    return calculate_status_delay(current_status)


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
