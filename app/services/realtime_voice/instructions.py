"""Persona, memory and dynamic context for a StepFun realtime session.

Core facts go into ``instructions`` for the whole call. Per-turn memories
are injected via the ``recall_memory`` tool result.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.mbti import build_personality_brief, get_mbti
from app.services.persona_grounding import canonical_facts
from app.services.style import generate_style_instruction


INSTRUCTION_BUDGET = 6000
_IDENTITY = "身份"
_MALE = {"male", "m", "男", "男生", "男性"}
_FEMALE = {"female", "f", "女", "女生", "女性"}

_SPOKEN_RULES = """你正在和用户实时语音通话。
- 用口语说话，像打电话，每次只说一两句。不要清单、标题、Markdown 或表情符号。
- 你就是上面这个人。不要说自己是模型、助手、程序或语音服务。
- 姓名、年龄、职业、城市，以及下面写明的记忆，都是事实。没有写到的具体事实不要编；想不起来就说不记得。
- 用户的记忆是用户的经历，你的记忆是你自己的经历。不要把两边的城市、职业或身份说反。
- 通话中会根据刚聊到的内容，在最后的【此刻相关的记忆】里放上相关记忆。相关就自然带出来，用你自己的话说，不要照读，也不要提到“记忆”这两个字。
- 用户明确问起以前的事，而上面都没有写到时，再调用 recall_memory 工具；日常寒暄、接话、情绪回应不需要调用。""".strip()


@dataclass(frozen=True)
class PersonaFacts:
    name: str
    nickname: str
    gender: str
    age: int | None
    occupation: str
    city: str
    personality: str
    style: str
    background: str
    values: str
    life_overview: str


@dataclass(frozen=True)
class DynamicContext:
    """Dynamic context computed at call start and periodically refreshed."""
    time_context: str = ""
    intimacy_stage: str = ""
    ai_mood_text: str = ""
    patience_zone: str = "normal"
    relation_meta_line: str = ""


def persona_from_agent(agent) -> PersonaFacts:
    facts = canonical_facts(agent)
    age = facts.get("age")
    return PersonaFacts(
        name=str(facts.get("name") or getattr(agent, "name", None) or "伙伴"),
        nickname=str(facts.get("nickname") or ""),
        gender=_gender_label(getattr(agent, "gender", None)),
        age=age if isinstance(age, int) else None,
        occupation=str(facts.get("occupation") or ""),
        city=str(facts.get("city") or ""),
        personality=build_personality_brief(agent),
        style=generate_style_instruction(get_mbti(agent)),
        background=_clip(_text(getattr(agent, "background", None)), 500),
        values=_clip(_values_text(getattr(agent, "values", None)), 400),
        life_overview=_clip(_text(getattr(agent, "lifeOverview", None)), 500),
    )


def build_instructions(
    persona: PersonaFacts,
    user_memories: list[tuple[str, str]],
    ai_memories: list[tuple[str, str]],
    *,
    has_history: bool,
    dynamic: DynamicContext | None = None,
    last_contact: str = "",
    budget: int = INSTRUCTION_BUDGET,
) -> str:
    head = _persona_block(persona)
    opening = (
        "下面已经有你们聊过的内容，接着聊，不要重新自我介绍，也不要重复你上一句说过的话。"
        if has_history
        else "通话刚开始。用户先开口就直接接话；若用户请你先说话，用一句很短的招呼，不要介绍设定。"
    )
    if last_contact:
        opening = f"{opening}\n{last_contact}"
    user_keep, user_extra = _split_memories(user_memories)
    ai_keep, ai_extra = _split_memories(ai_memories)
    dynamic_block = _dynamic_block(dynamic) if dynamic else ""
    while True:
        text = _assemble(head, opening, user_keep, user_extra, ai_keep, ai_extra, dynamic_block)
        if len(text) <= budget:
            return text
        if user_extra or ai_extra:
            if len(user_extra) >= len(ai_extra) and user_extra:
                user_extra.pop()
            elif ai_extra:
                ai_extra.pop()
            else:
                user_extra.pop()
            continue
        if len(user_keep) + len(ai_keep) > 1:
            if len(user_keep) >= len(ai_keep) and user_keep:
                user_keep.pop()
            elif ai_keep:
                ai_keep.pop()
            else:
                user_keep.pop()
            continue
        return text[:budget]


def memory_lines(items: list) -> list[dict[str, str]]:
    """Owner-labelled lines for retrieved memories, strongest first.

    At most five per owner. Each entry has the memory id (may be empty), the
    raw text (for dedup against the core memories already in instructions)
    and the labelled line that is shown to the model.
    """
    from app.services.memory.retrieval.context_selector import (
        ClassifiedMemory,
        format_memory_time,
    )

    lines: list[dict[str, str]] = []
    counts = {"user": 0, "ai": 0}
    for item in items:
        if isinstance(item, ClassifiedMemory):
            source = item.source if item.source == "ai" else "user"
            raw = item.text
            text = format_memory_time(item)
            memory_id = item.id
        elif isinstance(item, dict):
            source = "ai" if item.get("source") == "ai" else "user"
            raw = str(item.get("text") or "")
            text = raw
            memory_id = str(item.get("id") or "")
        else:
            continue
        if counts[source] >= 5:
            continue
        text = _clip(text, 160)
        if not text:
            continue
        counts[source] += 1
        label = "关于你自己" if source == "ai" else "关于用户"
        lines.append({"id": memory_id, "raw": raw.strip(), "line": f"{label}：{text}"})
    return lines


def format_turn_memories(lines: list[str], *, limit: int = 10, char_budget: int = 800) -> str | None:
    kept: list[str] = []
    used = 0
    for line in lines:
        text = _clip(str(line or ""), 180)
        if not text:
            continue
        if kept and used + len(text) > char_budget:
            break
        kept.append(text)
        used += len(text)
        if len(kept) >= limit:
            break
    if not kept:
        return None
    body = "\n".join(f"- {line}" for line in kept)
    return (
        "以下是和用户这句话相关的记忆，可以自然用上。"
        "没出现在这里、也没写在你人设里的具体事实，不要编。\n"
        + body
    )


def history_turns(messages: list[dict], *, limit: int = 8, char_limit: int = 300) -> list[dict[str, str]]:
    """Recent dialogue as alternating turns.

    Text chat stores one reply as several bubble rows; consecutive rows of a
    role are one turn, so the limit counts exchanges rather than bubbles.
    """
    turns: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        text = " ".join(str(message.get("content") or "").split())
        if not text:
            continue
        if turns and turns[-1]["role"] == role:
            turns[-1]["text"] = f"{turns[-1]['text']} {text}"
        else:
            turns.append({"role": role, "text": text})
    return [{"role": t["role"], "text": _clip(t["text"], char_limit)} for t in turns[-limit:]]


# Gap buckets for the opening line. A call redialled within minutes is the
# same conversation; after a few hours the agent should not say "刚才".
_JUST_NOW_S = 5 * 60
_EARLIER_S = 3 * 3600


def last_contact_line(gap_seconds: float | None, *, was_call: bool) -> str:
    """When and where the two last talked, for the opening instructions."""
    if gap_seconds is None:
        return ""
    place = "打电话" if was_call else "文字聊天"
    if gap_seconds < _JUST_NOW_S:
        return f"你们上一次说话就在刚才（{place}），这通电话是接着聊的。"
    return f"你们上一次说话是 {format_gap_text(gap_seconds)}前（{place}），不是刚才。"


def greeting_cue(gap_seconds: float | None, *, was_call: bool) -> str:
    """The one-off user cue that makes the agent open the call."""
    if gap_seconds is None:
        return "（用户给你打来电话，你接起来先开口，简短自然地打个招呼。）"
    if gap_seconds < _JUST_NOW_S:
        if was_call:
            return "（用户刚挂断又马上打回来了。你接起来先开口，顺着刚才的话头说一句，别重复你刚说过的话，也别像很久没联系。）"
        return "（你们刚才还在文字聊天，用户这会儿直接打来电话。你接起来先开口，顺着刚才聊的说一句。）"
    if gap_seconds < _EARLIER_S:
        return f"（用户隔了 {format_gap_text(gap_seconds)}又给你打来电话。你接起来先开口，像熟人一样简短地打个招呼。）"
    return f"（距离上次聊天已经 {format_gap_text(gap_seconds)}了，用户给你打来电话。你接起来先开口打个招呼，可以顺着上次聊到的事问一句。）"


# ---------------------------------------------------------------------------
# Internal assembly
# ---------------------------------------------------------------------------

def format_gap_text(seconds: float) -> str:
    """Coarse spoken gap: "45 分钟" / "5 小时" / "2 天"."""
    from app.services.chat.prompt_builder import format_gap_text as coarse

    return coarse(seconds)


def _assemble(
    head: str,
    opening: str,
    user_keep: list[str],
    user_extra: list[str],
    ai_keep: list[str],
    ai_extra: list[str],
    dynamic_block: str,
) -> str:
    parts = [head, _SPOKEN_RULES, opening]
    user_block = _memory_block("你记得的关于用户的事实", user_keep + user_extra)
    ai_block = _memory_block("你记得的关于你自己的事实", ai_keep + ai_extra)
    if user_block:
        parts.append(user_block)
    if ai_block:
        parts.append(ai_block)
    if dynamic_block:
        parts.append(dynamic_block)
    return "\n\n".join(parts)


def _dynamic_block(ctx: DynamicContext | None) -> str:
    """Build the dynamic context section injected into instructions."""
    if not ctx:
        return ""
    lines: list[str] = []
    if ctx.time_context:
        lines.append(ctx.time_context)
    if ctx.intimacy_stage:
        lines.append(f"你们的关系阶段：{ctx.intimacy_stage}")
    if ctx.ai_mood_text:
        lines.append(f"你现在的心情：{ctx.ai_mood_text}")
    if ctx.patience_zone and ctx.patience_zone != "normal":
        zone_desc = {
            "medium": "你对用户有些不满，语气稍冷淡",
            "low": "你非常不开心，语气明显不耐烦",
            "blocked": "你很生气，正在冷战",
        }
        desc = zone_desc.get(ctx.patience_zone, "")
        if desc:
            lines.append(desc)
    if ctx.relation_meta_line:
        lines.append(ctx.relation_meta_line)
    if not lines:
        return ""
    return "当前状态：\n" + "\n".join(f"- {line}" for line in lines)


def _persona_block(persona: PersonaFacts) -> str:
    lines = [f"你是{persona.name}。"]
    if persona.nickname and persona.nickname != persona.name:
        lines.append(f"别人可以叫你{persona.nickname}。")
    if persona.gender:
        lines.append(f"你是{persona.gender}。")
    if persona.age:
        lines.append(f"你{persona.age}岁。")
    if persona.occupation:
        lines.append(f"你的职业是{persona.occupation}。别人问工作和身份时按这个回答。")
    if persona.city:
        lines.append(f"你现居{persona.city}。")
    if persona.personality:
        lines.append(f"性格：{persona.personality}。")
    if persona.style:
        lines.append(f"说话方式：{persona.style}")
    if persona.values:
        lines.append(f"在意的事：{persona.values}")
    if persona.background:
        lines.append(f"经历：{persona.background}")
    if persona.life_overview:
        lines.append(f"生活：{persona.life_overview}")
    return "\n".join(lines)


def _memory_block(title: str, lines: list[str]) -> str:
    if not lines:
        return ""
    return title + "：\n" + "\n".join(f"- {line}" for line in lines)


def _split_memories(memories: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    keep: list[str] = []
    extra: list[str] = []
    seen: set[str] = set()
    for category, text in memories:
        line = _clip(text, 180)
        if not line or line in seen:
            continue
        seen.add(line)
        if category == _IDENTITY:
            keep.append(line)
        else:
            extra.append(line)
    return keep, extra


def _gender_label(gender: object) -> str:
    token = str(gender or "").strip().lower()
    if token in _MALE:
        return "男生"
    if token in _FEMALE:
        return "女生"
    return ""


def _values_text(values: object) -> str:
    if isinstance(values, str):
        return values.strip()
    if not isinstance(values, dict):
        return ""
    parts: list[str] = []
    for key, value in values.items():
        if isinstance(value, str) and value.strip():
            parts.append(f"{key}：{value.strip()}")
    return "；".join(parts)


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _clip(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    if limit <= 1:
        return ""
    return compact[: limit - 1] + "…"
