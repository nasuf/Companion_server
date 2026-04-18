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
    MEMORY_TOKEN_BUDGET,
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


_EMOTION_LABEL_CN = {
    "joy": "开心", "sadness": "难过", "anger": "生气", "fear": "害怕",
    "surprise": "惊讶", "disgust": "反感", "neutral": "平静",
    "anxiety": "焦虑", "disappointment": "失望", "relief": "欣慰",
    "gratitude": "感激", "playful": "调皮",
}


async def _build_emotion_section(
    emotion: dict | None,
    user_emotion: dict | None = None,
    intimacy_stage: str | None = None,
) -> str | None:
    """Build the emotion section from PAD dicts (AI + user)."""
    if not emotion:
        return None

    pleasure = emotion.get("pleasure", 0.0)
    arousal = emotion.get("arousal", 0.0)
    dominance = emotion.get("dominance", 0.0)

    mood_parts: list[str] = []
    if pleasure > 0.3:
        mood_parts.append("心情不错")
    elif pleasure < -0.3:
        mood_parts.append("有点低落")

    if arousal > 0.7:
        mood_parts.append("比较兴奋")
    elif arousal < 0.3:
        mood_parts.append("比较平静")

    if dominance > 0.7:
        mood_parts.append("感觉自信")
    elif dominance < 0.3:
        mood_parts.append("感觉有些被动")

    mood_text = "，".join(mood_parts) if mood_parts else "心情平静"

    # PRD §4.6.2.1: 注入 AI primary_emotion
    ai_primary = emotion.get("primary_emotion")

    body = f"你现在的情绪：{mood_text}\n"
    if ai_primary:
        body += f"主要情绪：{ai_primary}\n"
    body += f"(PAD: {pleasure:.1f}, {arousal:.1f}, {dominance:.1f})\n"

    if user_emotion:
        primary = user_emotion.get("primary_emotion", "neutral")
        confidence = user_emotion.get("confidence", 0.0)
        u_pleasure = user_emotion.get("pleasure", 0.0)
        u_arousal = user_emotion.get("arousal", 0.0)
        u_dominance = user_emotion.get("dominance", 0.0)

        if confidence >= 0.5 and primary != "neutral":
            emotion_cn = _EMOTION_LABEL_CN.get(primary, primary)
            body += f"用户当前情绪：{emotion_cn}（置信度{confidence:.1f}）\n"
            body += f"用户PAD向量：({u_pleasure:.2f}, {u_arousal:.2f}, {u_dominance:.2f})\n"

            if u_pleasure < -0.3:
                body += "请注意关心用户的感受。\n"

    # PRD §4.6.2.1: 注入亲密度阶段
    if intimacy_stage:
        body += f"\n你们目前的关系是{intimacy_stage}。\n"

    emotion_instruction = await get_prompt_text("chat.emotion_instruction")
    body += f"\n{emotion_instruction}"
    return _section("当前情绪", body)


def _build_summarizer_section(summaries: dict | None) -> str | None:
    """Build summarizer sections (review / distillation / state)."""
    if not summaries:
        return None

    parts: list[str] = []

    review = summaries.get("review")
    if review:
        parts.append(f"### 对话回顾\n{review}")

    distillation = summaries.get("distillation")
    if distillation:
        parts.append(f"### 记忆要点\n{distillation}")

    state = summaries.get("state")
    if state:
        parts.append(f"### 当前状态\n{state}")

    if not parts:
        return None

    return _section("上下文摘要", "\n\n".join(parts))


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
    memory_instruction = await get_prompt_text("chat.memory_instruction")
    body = (
        "以下是你记忆中与当前话题相关的事实。回答用户时必须与这些记忆保持一致，不得编造与之矛盾的信息。\n\n"
        f"{numbered}\n\n"
        f"{memory_instruction.format(budget=MEMORY_TOKEN_BUDGET)}"
    )
    return _section("你记得的事情", body)


def _build_working_memory_section(working_facts: list[str] | None) -> str | None:
    """Build the hot-path working memory section."""
    if not working_facts:
        return None

    numbered = "\n".join(f"- {fact}" for fact in working_facts)
    body = (
        "以下是当前会话中刚确认的高价值事实。"
        "优先用于承接刚刚提到的重要信息，但不要机械复述：\n"
        f"{numbered}"
    )
    return _section("当前会话事实", body)


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
    working_facts: list[str] | None = None,
    delay_context: str | None = None,
    relational_context: str | None = None,
    emotion: dict | None = None,
    graph_context: dict | None = None,
    summaries: dict | None = None,
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

    emo = await _build_emotion_section(emotion, user_emotion, intimacy_stage)
    if emo:
        sections.append(emo)

    summ = _build_summarizer_section(summaries)
    if summ:
        sections.append(summ)

    port = _build_portrait_section(portrait)
    if port:
        sections.append(port)

    working = _build_working_memory_section(working_facts)
    if working:
        sections.append(working)

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
