"""
Prompt Builder Service

Builds the multi-layer prompt stack for the AI companion agent.
Uses seven-dim personality (0-100) to build role-play personality descriptions.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from app.services.memory.retrieval.context_selector import ClassifiedMemory
from app.services.prompting.store import get_prompt_text
from app.services.style import generate_style_instruction
from app.services.mbti import format_mbti_for_prompt, get_mbti
from app.services.prompts.system_prompts import (
    MAX_PER_REPLY as _MAX_PER_REPLY,
    MAX_TOTAL_CHARS as _MAX_TOTAL_CHARS,
    CHAT_HISTORY_TOKEN_BUDGET,
)


# ---------------------------------------------------------------------------
# 七维人格 → 自然语言人格描述 (PRD §1.4.3 Prompt模板)
# ---------------------------------------------------------------------------

# MBTI 4 个维度 → (高分描述, 低分描述, 中段描述)
_MBTI_DIM_DESCRIPTIONS: dict[str, tuple[str, str, tuple[str, str, str]]] = {
    "EI": ("E", "I", (
        "外向开朗，喜欢与人互动，从社交中获取能量",
        "内向克制，享受独处，社交后需要独自恢复",
        "介于两者之间，根据场合调整",
    )),
    "NS": ("N", "S", (
        "直觉型，思维抽象、跳跃，喜欢探讨可能性",
        "感觉型，关注现实细节、当下事实",
        "兼具直觉与现实感",
    )),
    "TF": ("T", "F", (
        "思考型，逻辑清晰，按事实和原则做判断",
        "情感型，共情能力高，重视和谐与他人感受",
        "能平衡理性与感性",
    )),
    "JP": ("J", "P", (
        "判断型，喜欢规划、有条理、追求确定性",
        "知觉型，灵活随性、接受变化、活在当下",
        "有计划也能接受变动",
    )),
}


def _format_mbti_detail(mbti: dict) -> str:
    """4 维度数值 + 描述，用于 prompt 详细注入。"""
    lines: list[str] = []
    for i, (key, (hi_letter, lo_letter, (hi_desc, lo_desc, mid_desc))) in enumerate(
        _MBTI_DIM_DESCRIPTIONS.items(), 1,
    ):
        value = mbti.get(key, 50)
        normalized = value / 100
        if normalized >= 0.7:
            desc = hi_desc
            letter = hi_letter
        elif normalized <= 0.3:
            desc = lo_desc
            letter = lo_letter
        else:
            desc = mid_desc
            letter = f"{hi_letter}/{lo_letter}"
        lines.append(f"{i}. {key} [{letter}]：{value} — {desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _section(title: str, body: str) -> str:
    """Return a clearly-labelled prompt section."""
    return f"## {title}\n{body}"


def _has_prompt_body(body: str | None) -> bool:
    """Treat empty/admin-placeholder prompt text as absent."""
    if not body:
        return False
    stripped = body.strip()
    if not stripped:
        return False
    return bool(stripped.strip("。．.；;：:-—_ \n\t"))


def _optional_section(title: str, body: str | None) -> str | None:
    if not _has_prompt_body(body):
        return None
    return _section(title, str(body).strip())


async def _build_personality_section(agent: Any) -> str:
    """Build the personality section using MBTI (spec §1.2)."""
    name = getattr(agent, "name", None) or "伙伴"

    values = getattr(agent, "values", None)
    gender = "female"
    if isinstance(values, dict):
        gender = values.get("gender", "female")
    gender_text = "女生" if gender == "female" else "男生"

    mbti = get_mbti(agent)
    mbti_line = format_mbti_for_prompt(mbti)
    style = generate_style_instruction(mbti)

    detail = _format_mbti_detail(mbti) if mbti else "（性格未生成，将使用默认中性表达）"

    # Phase 6: 删 personality_rules 拼接. 实证内容跟 SYSTEM_BASE / RESPONSE_INSTRUCTION
    # 4 句全重叠 ("不要正式 / 不要客服 / 不要堆砌语气词 / 保持性格"). 删除节省 ~50
    # tokens 静态段, 减少噪声.
    body = (
        f"你的名字叫{name}，是一个{gender_text}。\n"
        f"你的性格画像：{mbti_line or '中性'}\n\n"
        f"四个维度详情：\n{detail}\n\n"
        f"你的说话风格：\n{style}"
    )
    return _section("你的身份", body)


async def _build_emotion_section(
    user_emotion: dict | None = None,
    intimacy_stage: str | None = None,
) -> str | None:
    """Phase 2.3 step 1: 删除 raw PAD vector 注入.

    历史方案: 注入 "用户PAD向量：(0.50, 0.30, 0.50)" + 条件性"请注意关心用户的感受".
    问题: LLM 看不懂抽象数值, 30 tokens/msg 浪费; 提示语只在 pleasure<-0.3 时
    fire 偏离了"PAD 影响语气" 的初衷.

    现方案: 仅注入 intimacy_stage (LLM 对自然语言 stage label 敏感度好). PAD
    数值仍在算法层用 (delay 计算 / 主动消息时机), 但不再注入 prompt.

    后续 Phase 2.3 step 2 可加自然语言 PAD 描述 (e.g. "用户略显低落但平静"),
    需在生产观察"删 raw PAD 后回复风格是否退化"再决定.
    """
    if not intimacy_stage:
        return None

    parts: list[str] = [f"你们目前的关系是{intimacy_stage}。"]

    return _section("当前情绪", "\n".join(parts))


    # (core_memory permanent injection removed — spec §3 uses retrieval only)


async def _build_memory_section(
    memories: list[ClassifiedMemory] | None,
    *,
    include_empty_anchor: bool = True,
) -> str | None:
    """按 owner 分两段渲染. 见 ClassifiedMemory.source 分组原因.

    即便 memories 为空 (弱路径不调记忆 / 强中路径召回为空) 也注入空 section,
    给 chat.consistency_rules 的反幻觉规则一个可靠锚点 — LLM 看到 "(本次没有
    联想到任何相关记忆)" 就明白搜过了没有, 可以柔和拒绝用户的预设性问句, 而不是
    顺承编造. 详见 CLAUDE.md 偏离表对应章节.
    """
    if not memories:
        if not include_empty_anchor:
            return None
        return _section(
            "你记得的事情",
            "(本次没有联想到任何与当前话题相关的记忆)",
        )

    def _days_since(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                return None
        if isinstance(value, datetime) and value.tzinfo:
            return (datetime.now(timezone.utc) - value).days
        if isinstance(value, datetime):
            return (datetime.now() - value).days
        return None

    def _format_memory(m: ClassifiedMemory) -> str:
        tags: list[str] = []
        if m.importance >= 0.85:
            tags.append("重要")
        if m.mention_count >= 3:
            tags.append("多次提及")
        days = _days_since(m.last_accessed_at or m.created_at)
        if days is not None and days < 30:
            tags.append("近期提到")
        score = m.display_score or m.score
        if score >= 0.75:
            tags.append("和当前话题高度相关")
        if not tags:
            return m.text
        return f"({' · '.join(tags)}) {m.text}"

    user_texts = [_format_memory(m) for m in memories if m.source != "ai"]
    ai_texts = [_format_memory(m) for m in memories if m.source == "ai"]

    def _numbered(label: str, items: list[str]) -> str:
        body = "\n".join(f"{i}. {t}" for i, t in enumerate(items, 1))
        return f"{label}\n{body}"

    parts: list[str] = []
    if user_texts:
        parts.append(_numbered("【用户告诉过你的事情】", user_texts))
    if ai_texts:
        parts.append(_numbered("【你自己的相关经历 / 人设】", ai_texts))

    if not parts:
        return None

    body = (
        "以下是与当前话题相关的事实, 已按归属分组. 回答时必须与这些保持一致, "
        "不得编造矛盾信息, 也不要把对方的记忆误当成自己的、或反之。"
        "括号里的标记只供你判断轻重缓急, 回复时不要复述这些标记。\n\n"
        + "\n\n".join(parts)
    )
    return _section("你记得的事情", body)


def _build_delay_context_section(delay_context: str | None) -> str | None:
    """Build the delayed-reply explanation section."""
    if not delay_context:
        return None
    return _section("回复时机说明", delay_context)


def _build_portrait_section(portrait: str | None) -> str | None:
    """Build the user portrait section."""
    if not portrait:
        return None
    return _section("用户画像", portrait)


# Phase 6: 删除 _build_relational_context_section + _build_graph_context_section.
# 实证依据:
# - relational_context: 注入"先接情绪/不要长解释" 等泛指令, 跟 SYSTEM_BASE
#   "像真人朋友" + ANTI_HALLUCINATION + RESPONSE_INSTRUCTION 重叠. 现代 LLM 看
#   SYSTEM_BASE 自然会做, 重复反而稀释信号.
# - graph_context: 注入"用户感兴趣 X / 经常提 Y / 高频分类 Z" 抽象列表.
#   信息已在 memory section 以具体形式出现, 抽象列表诱导 LLM 编造
#   ("用户感兴趣编程" → 给编程建议而非基于具体记忆). ~150-200 tokens/请求浪费.
# 调用方 build_system_prompt 也同时删除入参 + section append.


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def build_system_prompt(
    agent: Any,
    memories: list[ClassifiedMemory] | None = None,
    delay_context: str | None = None,
    portrait: str | None = None,
    topic_context: str | None = None,
    user_emotion: dict | None = None,
    patience_instruction: str | None = None,
    reply_count: int = 2,
    reply_total: int = _MAX_TOTAL_CHARS,
    intimacy_stage: str | None = None,
    time_context: str | None = None,
    time_memories: list[str] | None = None,
    l3_memories: list[str] | None = None,
    ai_status: dict | None = None,
    memory_relevance: str = "medium",
    # Phase 6: relational_context / graph_context 已删除 (实证冗余/幻觉源).
    # 保留 **kwargs 兜底以防 caller 还在传 — 调用方代码同步清理后可删 kwargs.
    **_deprecated_kwargs,
) -> str:
    """Build the full system prompt from the prompt stack.

    Section 排序按 dashscope context cache 友好原则: **稳定前缀在前, 变化字段后置**.
    阿里云 prefix-based cache 命中要前缀字节级完全相同 — 把 stable (核心规则 / 反幻觉
    / 身份+性格 / 对话一致性) 集中放头部, 让首次后的请求都能命中前缀 cache, 实测
    cached_input_tokens 单价是 input 的 ~40%, 单条 ~60% input cost 省下来 + TTFB
    减 100-300ms (服务端 KV cache 复用).

    变化字段 (情绪/画像/记忆/时间/L3/状态/回复要求 n=随机) 全部排到稳定段之后.
    cache miss 从这里开始, 但稳定段 ~1500 tokens 已经命中, 收益占比 80%+.
    """
    # Parallel — 4 independent prompt reads (each turn, hot path).
    system_base, consistency_rules, response_instruction, anti_hallucination = await asyncio.gather(
        get_prompt_text("chat.system_base"),
        get_prompt_text("chat.consistency_rules"),
        get_prompt_text("chat.response_instruction"),
        get_prompt_text("chat.anti_hallucination_hard_rule"),
    )

    # ═══ STABLE PREFIX (cache 命中区) ════════════════════════════════════
    # 同 agent 跨请求字节级一致, dashscope prefix cache 应命中.
    sections: list[str] = [_section("核心规则", system_base)]
    anti_hallucination_section = _optional_section("反幻觉硬约束", anti_hallucination)
    if anti_hallucination_section:
        sections.append(anti_hallucination_section)
    sections.append(await _build_personality_section(agent))   # per-agent 稳定
    consistency_section = _optional_section("对话一致性", consistency_rules)
    if consistency_section:
        sections.append(consistency_section)

    # ═══ VARIABLE SUFFIX (每请求变化, cache miss 起点) ═══════════════════

    emo = await _build_emotion_section(user_emotion, intimacy_stage)
    if emo:
        sections.append(emo)

    port = _build_portrait_section(portrait)
    if port:
        sections.append(port)

    delay = _build_delay_context_section(delay_context)
    if delay:
        sections.append(delay)

    # Phase 6: 删 relational_context 注入 (实证冗余 SYSTEM_BASE)

    mem = await _build_memory_section(
        memories,
        include_empty_anchor=(
            memory_relevance != "weak" and anti_hallucination_section is not None
        ),
    )
    if mem:
        sections.append(mem)

    # Phase 6: 删 graph_context 注入 (信息冗余 memory section, 抽象列表诱导编造)

    if topic_context:
        sections.append(_section("话题上下文", topic_context))

    # 时间上下文: 仅注入日期/星期/节假日, 不注入 AI 当前活动 (schedule_context).
    # spec §4 日常交流 步骤 4.3 / 5B.3 的"汇总参考信息"明确不包含 AI 当前作息;
    # 只有 §3.4.3 询问当前状态 才需要, 那是另一条 short-circuit 路径
    # (intent_handlers.handle_current_state → current_state_reply prompt).
    # 这里若注入 schedule_context 会让 §4 主回复跟 §3.4.3 的输出主题撞车
    # (例: 用户问"有意思。你现在在干嘛", 主意图 §3.4.3 回"我在沙发看剧",
    # 子意图"日常交流" §4 也回"我在沙发看老剧" — 重复). 实测 langsmith trace
    # 已确认这个失效, 见 commit 3d0417d 上下文.
    # NOTE: 回复时机说明已通过 reply_context.delay_seconds / received_at 路径
    # 单独注入 (delay_context_section), 不依赖 schedule_context.
    if time_context:
        sections.append(_section("时间", time_context))

    # 时间相关记忆
    if time_memories:
        numbered = "\n".join(f"- {m}" for m in time_memories)
        sections.append(_section("相关时间记忆", f"用户提到的时间对应的记忆：\n{numbered}"))

    # Spec §3.2 step 3: L3 distant memories (awakened only when relevant)
    if l3_memories:
        l3_block = "\n".join(f"- {m}" for m in l3_memories)
        sections.append(_section(
            "久远记忆（L3）",
            "以下是你很久以前的模糊记忆，用户正在回忆相关内容。"
            "回忆时语气自然，可以说\"我好像记得...\"或\"那好像是...\"：\n" + l3_block
        ))

    # 5B.4: 耐心区间语气描述
    if patience_instruction:
        sections.append(_section("情绪状态提醒", patience_instruction))

    # AI 自洽性约束 (§4 主回复路径). 告诉 LLM 当前状态 + 禁止主动展开,
    # 防止 ≥1min 延迟主回复路径下 LLM 编造跟实际状态矛盾的活动. 详见
    # CHAT_AI_STATE_CONSTRAINT_PROMPT 注释 (defaults.py).
    if ai_status:
        activity = str(ai_status.get("activity", "")).strip()
        status_label = str(ai_status.get("status", "idle")).strip()
        if activity:
            tpl = await get_prompt_text("chat.ai_state_constraint")
            sections.append(_section(
                "你的隐性状态约束",
                tpl.format(activity=activity, status=status_label),
            ))

    # 回复要求 (n=random 1-3 每轮变, 不可 cache, 排末尾)
    sections.append(
        _section(
            "回复要求",
            response_instruction.format(n=reply_count, total=reply_total, max_per=_MAX_PER_REPLY),
        )
    )

    return "\n\n".join(sections)


def build_chat_messages(
    system_prompt: str,
    messages: list[dict],
    token_budget: int = CHAT_HISTORY_TOKEN_BUDGET,
) -> list[dict]:
    """Return a list of role/content dicts ready for LLM consumption.

    Uses a token budget instead of a fixed message count:
    - Walks backwards from the latest message, adding complete messages
      until the budget is exhausted.
    - Short exchanges (嗯/好/哈哈) → more rounds of context.
    - Long messages (深度倾诉) → fewer rounds but full content.
    """
    from app.services.memory.retrieval.context_selector import estimate_tokens

    selected: list[dict] = []
    used_tokens = 0

    for msg in reversed(messages):
        content = msg.get("content", "")
        tokens = estimate_tokens(content)
        if used_tokens + tokens > token_budget and selected:
            break  # budget exhausted (always include at least the latest message)
        selected.append({"role": msg["role"], "content": content})
        used_tokens += tokens

    selected.reverse()

    result: list[dict] = [{"role": "system", "content": system_prompt}]
    result.extend(selected)
    return result
