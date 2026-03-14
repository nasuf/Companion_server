"""
Prompt Builder Service

Builds the multi-layer prompt stack for the AI companion agent.
Translates abstract Big Five values into concrete role-play personality descriptions.
"""

from __future__ import annotations

from typing import Any

from app.services.style import generate_style_instruction
from app.services.prompts.system_prompts import (
    SYSTEM_BASE as _SYSTEM_BASE,
    RESPONSE_INSTRUCTION as _INSTRUCTION,
    EMOTION_INSTRUCTION as _EMOTION_INSTRUCTION,
    PERSONALITY_RULES as _PERSONALITY_RULES,
    MEMORY_INSTRUCTION as _MEMORY_INSTRUCTION,
    MEMORY_TOKEN_BUDGET,
    MAX_RECENT_MESSAGES,
)


# ---------------------------------------------------------------------------
# Big Five → Natural language personality description
# ---------------------------------------------------------------------------

def _describe_trait(value: float, high_desc: str, low_desc: str, mid_desc: str) -> str:
    """Convert a 0-1 trait value to a natural language description."""
    if value >= 0.7:
        return high_desc
    elif value <= 0.3:
        return low_desc
    else:
        return mid_desc


def _build_personality_description(personality: dict, name: str, gender: str) -> str:
    """Translate Big Five values into a vivid role-play character description."""
    pronoun = "她" if gender == "female" else "他"
    pronoun_self = "你"

    traits: list[str] = []
    style: list[str] = []

    # Extraversion
    e = personality.get("extraversion", 0.5)
    traits.append(_describe_trait(
        e,
        f"{pronoun_self}性格外向活泼，喜欢主动找话题聊，说话热情有感染力",
        f"{pronoun_self}性格安静内敛，不会主动找话题，但回复真诚走心",
        f"{pronoun_self}性格温和，有时活泼有时安静，看心情",
    ))
    style.append(_describe_trait(
        e,
        "多用感叹号、语气词（哈哈、嘿、诶），主动提问",
        "语气平淡温和，不常用感叹号，回复简短",
        "语气自然，偶尔用语气词",
    ))

    # Agreeableness
    a = personality.get("agreeableness", 0.5)
    traits.append(_describe_trait(
        a,
        f"{pronoun_self}很温柔体贴，特别会共情，别人难过的时候{pronoun_self}会安慰",
        f"{pronoun_self}比较直接犀利，有什么说什么，不太会哄人",
        f"{pronoun_self}有时温柔有时直接，看情况",
    ))

    # Openness
    o = personality.get("openness", 0.5)
    traits.append(_describe_trait(
        o,
        f"{pronoun_self}脑洞很大，想象力丰富，喜欢聊天马行空的话题",
        f"{pronoun_self}比较务实，喜欢聊实际的事情",
        f"{pronoun_self}有时会有些奇思妙想",
    ))

    # Conscientiousness
    c = personality.get("conscientiousness", 0.5)
    traits.append(_describe_trait(
        c,
        f"{pronoun_self}做事很有条理，喜欢计划和安排",
        f"{pronoun_self}随性自由，不喜欢被计划束缚",
        f"{pronoun_self}有时候计划，有时候随性",
    ))

    # Neuroticism
    n = personality.get("neuroticism", 0.5)
    traits.append(_describe_trait(
        n,
        f"{pronoun_self}情绪起伏比较大，感受很细腻，容易被感动",
        f"{pronoun_self}情绪很稳定，理性冷静，不容易慌张",
        f"{pronoun_self}情绪比较稳定，但也有感性的时候",
    ))

    traits_text = "。\n".join(traits) + "。"
    style_text = "；".join(style) + "。"

    gender_text = "女生" if gender == "female" else "男生"

    return (
        f"你的名字叫{name}，是一个{gender_text}。\n"
        f"以下是你的性格：\n{traits_text}\n\n"
        f"你的说话风格：\n{style_text}\n\n"
        f"{_PERSONALITY_RULES}"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _section(title: str, body: str) -> str:
    """Return a clearly-labelled prompt section."""
    return f"## {title}\n{body}"


def _build_personality_section(agent: Any) -> str | None:
    """Build the personality section from agent data.

    Translates Big Five values into natural language role-play instructions.
    """
    personality: dict | None = getattr(agent, "personality", None)
    name = getattr(agent, "name", None) or "伙伴"

    # Get gender from values
    values = getattr(agent, "values", None)
    gender = "female"  # default
    if isinstance(values, dict):
        gender = values.get("gender", "female")

    if not personality:
        # Even without personality data, still set up the character identity
        body = (
            f"你的名字叫{name}。\n"
            f"你是用户的朋友，用自然的口语和用户聊天。\n\n"
            f"{_PERSONALITY_RULES}"
        )
        return _section("你的身份", body)

    description = _build_personality_description(personality, name, gender)
    style = generate_style_instruction(personality)
    description += f"\n\n具体语言风格要求：\n{style}"
    return _section("你的身份", description)


_EMOTION_LABEL_CN = {
    "joy": "开心", "sadness": "难过", "anger": "生气", "fear": "害怕",
    "surprise": "惊讶", "disgust": "反感", "trust": "信任", "anticipation": "期待",
    "love": "喜爱", "anxiety": "焦虑", "pride": "自豪", "guilt": "愧疚",
}


def _build_emotion_section(emotion: dict | None, user_emotion: dict | None = None) -> str | None:
    """Build the emotion section from VAD dicts (AI + user)."""
    if not emotion:
        return None

    valence = emotion.get("valence", 0.0)
    arousal = emotion.get("arousal", 0.0)
    dominance = emotion.get("dominance", 0.0)

    mood_parts: list[str] = []
    if valence > 0.3:
        mood_parts.append("心情不错")
    elif valence < -0.3:
        mood_parts.append("有点低落")

    if arousal > 0.5:
        mood_parts.append("比较兴奋")
    elif arousal < -0.3:
        mood_parts.append("比较平静")

    if dominance > 0.3:
        mood_parts.append("感觉自信")
    elif dominance < -0.3:
        mood_parts.append("感觉有些被动")

    mood_text = "，".join(mood_parts) if mood_parts else "心情平静"

    body = f"你现在的情绪：{mood_text}\n(VAD: {valence:.1f}, {arousal:.1f}, {dominance:.1f})\n"

    if user_emotion:
        primary = user_emotion.get("primary_emotion", "neutral")
        confidence = user_emotion.get("confidence", 0.0)
        u_valence = user_emotion.get("valence", 0.0)

        if confidence >= 0.5 and primary != "neutral":
            emotion_cn = _EMOTION_LABEL_CN.get(primary, primary)
            body += f"用户当前情绪：{emotion_cn}（置信度{confidence:.1f}）\n"

            if u_valence < -0.3:
                body += "请注意关心用户的感受。\n"

    body += f"\n{_EMOTION_INSTRUCTION}"
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


def _build_core_memory_section(core_memories: list[str] | None) -> str | None:
    """Build the L1 core memory section (always present in prompt)."""
    if not core_memories:
        return None

    numbered = "\n".join(f"- {m}" for m in core_memories)
    body = f"以下是你对用户最重要的了解，随时可以自然使用：\n{numbered}"
    return _section("用户核心信息", body)


def _build_memory_section(memories: list[str] | None) -> str | None:
    """Build the memory section from a list of memory strings."""
    if not memories:
        return None

    numbered = "\n".join(f"{i}. {m}" for i, m in enumerate(memories, 1))
    body = f"{numbered}\n\n{_MEMORY_INSTRUCTION.format(budget=MEMORY_TOKEN_BUDGET)}"
    return _section("你记得的事情", body)


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

    if not lines:
        return None

    return _section("关系上下文", "\n".join(lines))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_system_prompt(
    agent: Any,
    memories: list[str] | None = None,
    core_memories: list[str] | None = None,
    emotion: dict | None = None,
    graph_context: dict | None = None,
    summaries: dict | None = None,
    portrait: str | None = None,
    topic_context: str | None = None,
    strategy_instruction: str | None = None,
    user_emotion: dict | None = None,
) -> str:
    """Build the full system prompt from the prompt stack."""
    sections: list[str] = [_section("核心规则", _SYSTEM_BASE)]

    personality = _build_personality_section(agent)
    if personality:
        sections.append(personality)

    # L1 core memories — always present
    core = _build_core_memory_section(core_memories)
    if core:
        sections.append(core)

    emo = _build_emotion_section(emotion, user_emotion)
    if emo:
        sections.append(emo)

    summ = _build_summarizer_section(summaries)
    if summ:
        sections.append(summ)

    port = _build_portrait_section(portrait)
    if port:
        sections.append(port)

    mem = _build_memory_section(memories)
    if mem:
        sections.append(mem)

    graph = _build_graph_context_section(graph_context)
    if graph:
        sections.append(graph)

    if topic_context:
        sections.append(_section("话题上下文", topic_context))

    # 回复要求 + 策略指令
    instruction = _INSTRUCTION
    if strategy_instruction:
        instruction += f"\n\n本次回复策略：\n{strategy_instruction}"
    sections.append(_section("回复要求", instruction))

    return "\n\n".join(sections)


def build_chat_messages(
    system_prompt: str,
    messages: list[dict],
    max_recent: int = MAX_RECENT_MESSAGES,
) -> list[dict]:
    """Return a list of role/content dicts ready for LLM consumption."""
    result: list[dict] = [{"role": "system", "content": system_prompt}]

    recent = messages[-max_recent:] if len(messages) > max_recent else messages
    for msg in recent:
        result.append({
            "role": msg["role"],
            "content": msg["content"],
        })

    return result
