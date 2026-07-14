"""所有回复类指令的固定前置 (通用回复规则 + 反幻觉硬约束).

产品诉求: AI 发出的**每一条**用户可见消息 (含主动消息/短路回复/边界回复/
音乐/礼物文案) 都应遵守同一套核心回复规则与反幻觉底线, 而不是只有主回复
大 prompt 有. 实现上在 store.get_prompt_text 的单一出口对
REPLY_PROMPT_KEYS 内的模板注入前置:

    【通用回复规则】chat.response_instruction (渲染 {max_per}/{total} 常量)
    chat.anti_hallucination_hard_rule
    ------ 原模板 ------

设计要点:
- 两个前置模板各自尊重 admin 启用开关: 停用哪个哪个消失, 全停用 = 无前置.
- [EMO:标签/强度] 标记指令**不在**前置里 (已拆去 chat.reply_emotion_marker,
  仅主回复管线拼装) — 只有主回复路径会剥标记, 混入前置会漏给用户.
- 前置文本所有回复类调用字节级相同 → provider prefix cache 可跨调用命中,
  额外 token 大部分按缓存价计费.
- 分类原则: 只收"输出直接成为用户可见 AI 消息"的模板. 分类器/JSON 抽取/
  内部摘要/section 片段一律排除 (前置会污染结构化输出).
"""

from __future__ import annotations

import logging

from app.services.prompting.utils import safe_format
from app.services.prompts.system_prompts import MAX_PER_REPLY, MAX_TOTAL_CHARS

logger = logging.getLogger(__name__)

_PREFIX_HEADER = "【通用回复规则】"

# 前置自身的两个模板 key — 绝不能出现在 REPLY_PROMPT_KEYS (防递归).
PREFIX_SOURCE_KEYS: tuple[str, str] = (
    "chat.response_instruction",
    "chat.anti_hallucination_hard_rule",
)

# 输出直接成为用户可见 AI 消息的模板. 新增回复类模板时在此登记
# (有守卫测试校验 key 都在 registry 且分类器 key 不在).
REPLY_PROMPT_KEYS: frozenset[str] = frozenset({
    # 日常交流 tier 回复
    "memory.weak_reply",
    "memory.medium_reply",
    "memory.strong_reply",
    "memory.l3_reply",
    # 意图短路回复
    "intent.end_reply",
    "intent.schedule_query_reply",
    "intent.schedule_adjust_reply",
    "intent.schedule_missing_context",
    "intent.current_state_reply",
    "intent.deletion_confirm",
    "intent.deletion_reply",
    "intent.record_confirm_reply",
    "intent.record_ask_time",
    # 危机干预回复
    "intent.crisis_reply",
    "intent.crisis_followup_reply",
    # 矛盾处理 (追问/回复)
    "memory.contradiction_inquiry",
    "memory.contradiction_reply",
    # 边界系统回复
    "boundary.light_attack_reply",
    "boundary.medium_attack_reply",
    "boundary.severe_attack_reply",
    "boundary.medium_patience_reply",
    "boundary.low_patience_reply",
    "boundary.final_warning",
    "boundary.blacklist_reply",
    "boundary.apology_reply",
    # 异步回复
    "reply.delay_explanation",
    # 主动消息 (含特殊日期/提醒/开场)
    "proactive.silence_plain",
    "proactive.silence_ai_memory",
    "proactive.silence_user_memory",
    "proactive.silence_schedule",
    "proactive.memory_ai",
    "proactive.memory_user",
    "proactive.scheduled_scene",
    "proactive.decay_final",
    "proactive.first_greeting",
    "proactive.special_holiday",
    "proactive.special_birthday",
    "proactive.special_reminder",
    "proactive.special_combined",
    "proactive.reminder_message",
    # 音乐共听对话消息
    "music.proactive_recommend",
    "music.accept_invite",
    "music.busy_reject",
    "music.sleep_reject",
    "music.user_exit",
    "music.busy_exit",
    "music.user_pause_exit",
    "music.user_absent_exit",
    "music.switch_track",
    "music.track_changed_manual",
    "music.track_changed_auto",
    "music.agent_join_after_busy",
    "music.agent_late_missed",
    # 线下互动用户可见文案
    "offline.gift_first_address_request",
    "offline.gift_sent_message",
    "offline.gift_delivered_message",
    "offline.gift_thanks_reply",
    "offline.activity_invite_message",
})


async def build_reply_prefix() -> str:
    """构建固定前置文本. 两个来源模板各自尊重启用开关;
    全部不可用时返回空串 (调用方直接用原模板)."""
    from app.services.prompting.store import (
        PromptDisabledError,
        get_prompt_text,
    )

    parts: list[str] = []
    for key in PREFIX_SOURCE_KEYS:
        try:
            text = await get_prompt_text(key)
        except PromptDisabledError:
            continue
        except Exception as e:  # noqa: BLE001 — 前置失败不能挡住原模板
            logger.warning(f"[REPLY-PREFIX] load {key} failed: {e}")
            continue
        if key == "chat.response_instruction":
            # 渲染装饰性占位符为常量; n 仅兼容旧覆盖模板.
            rendered = safe_format(str(text), {
                "n": 2, "max_per": MAX_PER_REPLY, "total": MAX_TOTAL_CHARS,
            })
            parts.append(f"{_PREFIX_HEADER}\n{rendered}")
        else:
            parts.append(str(text))
    return "\n\n".join(parts)
