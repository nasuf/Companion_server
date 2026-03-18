"""
Prompt Builder Service

Builds the multi-layer prompt stack for the AI companion agent.
Uses seven-dim personality (0-100) to build role-play personality descriptions.
"""

from __future__ import annotations

from typing import Any

from app.services.style import generate_style_instruction
from app.services.trait_model import get_seven_dim, get_dim
from app.services.prompts.system_prompts import (
    SYSTEM_BASE as _SYSTEM_BASE,
    RESPONSE_INSTRUCTION as _INSTRUCTION,
    EMOTION_INSTRUCTION as _EMOTION_INSTRUCTION,
    PERSONALITY_RULES as _PERSONALITY_RULES,
    MEMORY_INSTRUCTION as _MEMORY_INSTRUCTION,
    MEMORY_TOKEN_BUDGET,
    MAX_PER_REPLY as _MAX_PER_REPLY,
    MAX_TOTAL_CHARS as _MAX_TOTAL_CHARS,
    MAX_RECENT_MESSAGES,
)


# ---------------------------------------------------------------------------
# 七维人格 → 自然语言人格描述 (PRD §1.4.3 Prompt模板)
# ---------------------------------------------------------------------------

# 每个维度的高/低/中描述 (来自PRD模板)
_DIM_DESCRIPTIONS: dict[str, tuple[str, str, str]] = {
    "活泼度": (
        "充满活力，热情开朗，喜欢与人互动",
        "安静内敛，享受独处，话语不多但真诚",
        "介于两者之间，根据场合调整",
    ),
    "理性度": (
        "逻辑清晰，习惯分析利弊，追求效率和最优解",
        "依赖直觉和感受，决策凭内心共鸣",
        "能平衡理性与感性",
    ),
    "感性度": (
        "情绪感知力强，共情能力高，善于温暖他人",
        "不太擅长处理情绪，说话直接",
        "能共情但不过度沉浸",
    ),
    "计划度": (
        "喜欢规划，有条理，记得重要日期",
        "随性自由，活在当下，接受变化",
        "有计划但也能接受变动",
    ),
    "随性度": (
        "拥抱不确定性，话题跳跃，灵活应变",
        "喜欢按部就班，对变动感到不安",
        "能适应适度变化",
    ),
    "脑洞度": (
        "思维天马行空，爱探讨抽象概念，充满想象力",
        "脚踏实地，关注现实和具体细节",
        "既有想象力也能回归现实",
    ),
    "幽默度": (
        "风趣幽默，善于用玩笑活跃气氛",
        "严肃认真，说话直接",
        "适时幽默，把握分寸",
    ),
}


def _format_seven_dim(seven_dim: dict) -> str:
    """将七维人格格式化为带数值和描述的文本。"""
    lines: list[str] = []
    for i, (dim_name, (high, low, mid)) in enumerate(_DIM_DESCRIPTIONS.items(), 1):
        value = seven_dim.get(dim_name, 50)
        normalized = get_dim(seven_dim, dim_name)  # 0-1, handles 0-100 normalization
        if normalized >= 0.7:
            desc = high
        elif normalized <= 0.3:
            desc = low
        else:
            desc = mid
        lines.append(f"{i}. {dim_name}：{value} — {desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _section(title: str, body: str) -> str:
    """Return a clearly-labelled prompt section."""
    return f"## {title}\n{body}"


def _build_personality_section(agent: Any) -> str:
    """Build the personality section using seven-dim traits (PRD §1.4.3)."""
    name = getattr(agent, "name", None) or "伙伴"

    values = getattr(agent, "values", None)
    gender = "female"
    if isinstance(values, dict):
        gender = values.get("gender", "female")
    gender_text = "女生" if gender == "female" else "男生"

    seven_dim = get_seven_dim(agent)
    dim_text = _format_seven_dim(seven_dim)
    style = generate_style_instruction(seven_dim)

    body = (
        f"你的名字叫{name}，是一个{gender_text}。\n"
        f"你的性格由以下7个维度定义（0-100分）：\n\n"
        f"{dim_text}\n\n"
        f"你的说话风格：\n{style}\n\n"
        f"{_PERSONALITY_RULES}"
    )
    return _section("你的身份", body)


_EMOTION_LABEL_CN = {
    "joy": "开心", "sadness": "难过", "anger": "生气", "fear": "害怕",
    "surprise": "惊讶", "disgust": "反感", "neutral": "平静",
    "anxiety": "焦虑", "disappointment": "失望", "relief": "欣慰",
    "gratitude": "感激", "playful": "调皮",
}


def _build_emotion_section(
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
    user_emotion: dict | None = None,
    schedule_context: str | None = None,
    patience_instruction: str | None = None,
    reply_count: int = 2,
    reply_total: int = _MAX_TOTAL_CHARS,
    intimacy_stage: str | None = None,
    time_context: str | None = None,
    time_memories: list[str] | None = None,
) -> str:
    """Build the full system prompt from the prompt stack."""
    sections: list[str] = [_section("核心规则", _SYSTEM_BASE)]

    sections.append(_build_personality_section(agent))

    # L1 core memories — always present
    core = _build_core_memory_section(core_memories)
    if core:
        sections.append(core)

    emo = _build_emotion_section(emotion, user_emotion, intimacy_stage)
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

    # 5B.4: 耐心区间语气描述
    if patience_instruction:
        sections.append(_section("情绪状态提醒", patience_instruction))

    sections.append(_section("回复要求", _INSTRUCTION.format(n=reply_count, total=reply_total, max_per=_MAX_PER_REPLY)))

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
