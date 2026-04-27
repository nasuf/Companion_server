"""
Prompt Builder Service

Builds the multi-layer prompt stack for the AI companion agent.
Uses seven-dim personality (0-100) to build role-play personality descriptions.
"""

from __future__ import annotations

from typing import Any

from app.services.memory.retrieval.context_selector import ClassifiedMemory
from app.services.prompting.store import get_prompt_text
from app.services.style import generate_style_instruction
from app.services.mbti import format_mbti_for_prompt, get_mbti, signal as mbti_signal
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

    personality_rules = await get_prompt_text("chat.personality_rules")
    body = (
        f"你的名字叫{name}，是一个{gender_text}。\n"
        f"你的性格画像：{mbti_line or '中性'}\n\n"
        f"四个维度详情：\n{detail}\n\n"
        f"你的说话风格：\n{style}\n\n"
        f"{personality_rules}"
    )
    return _section("你的身份", body)


async def _build_emotion_section(
    user_emotion: dict | None = None,
    intimacy_stage: str | None = None,
) -> str | None:
    """Spec §4 汇总参考信息：用户PAD值 + 关系阶段（不含 AI PAD — spec 把 AI PAD
    限定在 §5.3/§5.4/§6.2 装饰/时机决策）。

    PAD 数值本身已是足够暗示, 早期版本曾追加 chat.emotion_instruction "让情绪
    影响语气" 的明示指令, 后审计为工程冗余删除——现代 LLM 看到 PAD 值会自然反应.
    """
    if not user_emotion and not intimacy_stage:
        return None

    parts: list[str] = []

    if user_emotion:
        u_pleasure = user_emotion.get("pleasure", 0.0)
        u_arousal = user_emotion.get("arousal", 0.0)
        u_dominance = user_emotion.get("dominance", 0.0)
        parts.append(f"用户PAD向量：({u_pleasure:.2f}, {u_arousal:.2f}, {u_dominance:.2f})")
        if u_pleasure < -0.3:
            parts.append("请注意关心用户的感受。")

    if intimacy_stage:
        parts.append(f"你们目前的关系是{intimacy_stage}。")

    return _section("当前情绪", "\n".join(parts))


    # (core_memory permanent injection removed — spec §3 uses retrieval only)


async def _build_memory_section(memories: list | None) -> str | None:
    """Build the memory section with relevance classification.

    Accepts list[ClassifiedMemory] or list[str] (backward compat).
    """
    if not memories:
        return None

    # Separate by classification
    strong: list[str] = []
    medium: list[str] = []
    plain: list[str] = []

    for m in memories:
        if isinstance(m, ClassifiedMemory):
            if m.relevance == "strong":
                strong.append(m.text)
            else:
                medium.append(m.text)
        elif isinstance(m, str):
            plain.append(m)

    # Spec §3.2: all retrieved memories are factual context, no strong/medium split.
    all_texts: list[str] = []
    for m in memories:
        if isinstance(m, ClassifiedMemory):
            all_texts.append(m.text)
        elif isinstance(m, str):
            all_texts.append(m)

    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(all_texts, 1))
    # 早期版本拼了 chat.memory_instruction "(记忆上下文预算: 约 N tokens...)"
    # 元注释, 审计后删除——LLM 看到 token 数字也不改变行为, 是无效占位.
    body = (
        "以下是你记忆中与当前话题相关的事实。回答用户时必须与这些记忆保持一致，不得编造与之矛盾的信息。\n\n"
        f"{numbered}"
    )
    return _section("你记得的事情", body)


def _build_delay_context_section(delay_context: str | None) -> str | None:
    """Build the delayed-reply explanation section."""
    if not delay_context:
        return None
    return _section("回复时机说明", delay_context)


def _build_relational_context_section(relational_context: str | None) -> str | None:
    """Build the relationship-sensitive response guidance section."""
    if not relational_context:
        return None
    return _section("关系回应重点", relational_context)


def _build_portrait_section(portrait: str | None) -> str | None:
    """Build the user portrait section."""
    if not portrait:
        return None
    return _section("用户画像", portrait)


def _build_graph_context_section(graph_context: dict | None) -> str | None:
    """Build the graph/relationship context section."""
    if not graph_context:
        return None

    lines: list[str] = []

    topics = graph_context.get("topics")
    if topics:
        lines.append("用户感兴趣的话题：")
        for t in topics:
            lines.append(f"  - {t}")

    entities = graph_context.get("entities")
    if entities:
        if lines:
            lines.append("")
        lines.append("用户经常提到的：")
        for e in entities:
            lines.append(f"  - {e}")

    categories = graph_context.get("categories")
    if categories:
        if lines:
            lines.append("")
        lines.append("高频记忆分类：")
        for category in categories:
            lines.append(f"  - {category}")

    if not lines:
        return None

    return _section("关系上下文", "\n".join(lines))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def build_system_prompt(
    agent: Any,
    memories: list[str] | None = None,
    delay_context: str | None = None,
    relational_context: str | None = None,
    graph_context: dict | None = None,
    portrait: str | None = None,
    topic_context: str | None = None,
    user_emotion: dict | None = None,
    schedule_context: str | None = None,
    patience_instruction: str | None = None,
    reply_count: int = 2,
    reply_total: int = _MAX_TOTAL_CHARS,
    intimacy_stage: str | None = None,
    time_context: str | None = None,
    time_memories: list[str] | None = None,
    l3_memories: list[str] | None = None,
) -> str:
    """Build the full system prompt from the prompt stack."""
    system_base = await get_prompt_text("chat.system_base")
    consistency_rules = await get_prompt_text("chat.consistency_rules")
    response_instruction = await get_prompt_text("chat.response_instruction")

    sections: list[str] = [_section("核心规则", system_base)]

    sections.append(await _build_personality_section(agent))

    # Spec §3: 记忆全部通过检索注入,不再有"永驻核心记忆"。
    # Agent 基本身份(名字/性格)已在上方 personality section 中。

    emo = await _build_emotion_section(user_emotion, intimacy_stage)
    if emo:
        sections.append(emo)

    port = _build_portrait_section(portrait)
    if port:
        sections.append(port)

    delay = _build_delay_context_section(delay_context)
    if delay:
        sections.append(delay)

    relational = _build_relational_context_section(relational_context)
    if relational:
        sections.append(relational)

    mem = await _build_memory_section(memories)
    if mem:
        sections.append(mem)

    graph = _build_graph_context_section(graph_context)
    if graph:
        sections.append(graph)

    if topic_context:
        sections.append(_section("话题上下文", topic_context))

    # 时间上下文：当前状态 + 时间信息 + 节假日
    status_parts = []
    if schedule_context:
        status_parts.append(schedule_context)
    if time_context:
        status_parts.append(time_context)
    if status_parts:
        sections.append(_section("当前状态", "\n".join(status_parts)))

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

    sections.append(_section("对话一致性", consistency_rules))
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
