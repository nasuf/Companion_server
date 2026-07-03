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


# reason → (registry prompt key, {activity} 兜底默认值). 文案由 registry 管理
# (reply.delay_reason_*), 停用某条 → 该场景延迟原因说明从最终输入中彻底移除.
_DELAY_REASON_PROMPT_KEYS: dict[str, tuple[str, str]] = {
    "conversation_mode": ("reply.delay_reason_conversation_mode", ""),
    "high_emotion": ("reply.delay_reason_high_emotion", ""),
    "schedule_sleep": ("reply.delay_reason_sleep", "睡觉"),
    "schedule_very_busy": ("reply.delay_reason_very_busy", "忙事情"),
    "schedule_busy": ("reply.delay_reason_busy", "处理日常安排"),
}
_DELAY_REASON_DEFAULT_KEY = ("reply.delay_reason_default", "安排自己的事情")


async def explain_delay_reason(reason: str, activity: str | None = None, status: str | None = None) -> str:
    """Human-readable delay reason summary for prompt injection.

    文案取自 prompting registry; 停用时返回空串 (delay section 模板经
    render_template 会把空行剔除). registry 读取失败退回代码默认文案.
    """
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP
    from app.services.prompting.store import PromptDisabledError, get_prompt_text

    prompt_key, default_activity = _DELAY_REASON_PROMPT_KEYS.get(reason, _DELAY_REASON_DEFAULT_KEY)
    params = {"activity": activity or default_activity}
    try:
        tpl = await get_prompt_text(prompt_key)
    except PromptDisabledError:
        return ""
    except Exception:
        tpl = PROMPT_DEFINITION_MAP[prompt_key].default_text
    try:
        return tpl.format(**params)
    except Exception:
        return str(tpl)
