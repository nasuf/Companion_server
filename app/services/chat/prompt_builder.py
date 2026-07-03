"""
Prompt Builder Service

Builds the multi-layer prompt stack for the AI companion agent.
Uses seven-dim personality (0-100) to build role-play personality descriptions.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.services.memory.retrieval.context_selector import ClassifiedMemory
from app.services.prompting.store import (
    PromptDisabledError,
    get_prompt_text_for_context,
    get_prompt_text_or_default,
)
from app.services.prompting.trace_components import record_prompt_render
from app.services.prompting.utils import render_template
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

@dataclass(frozen=True)
class _PromptBody:
    body: str
    prompt_key: str | None = None


def _has_prompt_body(body: str | None) -> bool:
    """Treat empty/admin-placeholder prompt text as absent."""
    if not body:
        return False
    stripped = body.strip()
    if not stripped:
        return False
    return bool(stripped.strip("。．.；;：:-—_ \n\t"))


def _append_section(
    sections: list[str],
    components: list[dict[str, Any]],
    title: str,
    body: str,
    *,
    prompt_key: str | None = None,
) -> None:
    prefix_len = 2 if sections else 0  # "\n\n" inserted by final join before this section
    section_header = f"## {title}\n"
    start = sum(len(part) for part in sections) + max(0, len(sections) - 1) * 2
    body_start = start + prefix_len + len(section_header)
    section = section_header + body
    sections.append(section)
    if prompt_key:
        components.append({
            "prompt_key": prompt_key,
            "start": body_start,
            "end": body_start + len(body),
            "editable": True,
        })


def _record_skipped_section(diagnostics: dict[str, Any] | None, title: str) -> None:
    if diagnostics is None:
        return
    skipped = diagnostics.setdefault("empty_prompt_sections_removed", [])
    if isinstance(skipped, list):
        skipped.append(title)


async def _get_optional_prompt(
    key: str,
    *,
    agent_id: str | None = None,
    user_id: str | None = None,
) -> str | None:
    """Fetch a section template; admin 停用 → None (该段从最终输入中彻底移除)."""
    try:
        return await get_prompt_text_for_context(key, agent_id=agent_id, user_id=user_id)
    except PromptDisabledError:
        return None


def _render_section(template: str, params: dict[str, Any]) -> str:
    """Section 模板统一走 SafeDict 安全渲染.

    admin 可在线编辑这些模板; 裸 str.format 遇到编辑时新加的未知占位符
    (如 {备注}) 会 KeyError 打崩整条聊天热路径. SafeDict 把未知占位符渲染
    为 "(无)", 保存侧的括号配对校验 (_template_fields) 拦截语法错误.
    """
    return render_template(template, params)


async def _build_personality_section(agent: Any) -> _PromptBody | None:
    """Build the personality section using MBTI (spec §1.2).

    段模板 chat.personality_section 由 registry 管理; MBTI 详情与风格片段为
    动态注入 ({mbti_detail}/{style_rules}); 说话风格的通用规则
    (chat.style_base_rule / chat.style_closing_rule) 同样走 registry,
    停用时从风格指令中彻底移除.
    """
    tpl, base_rule, closing_rule = await asyncio.gather(
        _get_optional_prompt("chat.personality_section"),
        _get_optional_prompt("chat.style_base_rule"),
        _get_optional_prompt("chat.style_closing_rule"),
    )
    if tpl is None:
        return None

    name = getattr(agent, "name", None) or "伙伴"
    age = getattr(agent, "age", None)

    values = getattr(agent, "values", None)
    gender = "female"
    if isinstance(values, dict):
        gender = values.get("gender", "female")
    gender_text = "女生" if gender == "female" else "男生"

    mbti = get_mbti(agent)
    mbti_line = format_mbti_for_prompt(mbti)
    style = generate_style_instruction(mbti, base_rule=base_rule, closing_rule=closing_rule)

    detail = _format_mbti_detail(mbti) if mbti else "（性格未生成，将使用默认中性表达）"

    # Phase 6: 删 personality_rules 拼接. 实证内容跟 SYSTEM_BASE / RESPONSE_INSTRUCTION
    # 4 句全重叠 ("不要正式 / 不要客服 / 不要堆砌语气词 / 保持性格"). 删除节省 ~50
    # tokens 静态段, 减少噪声.
    age_text = f"你的年龄是{age}岁。" if isinstance(age, int) and age > 0 else ""

    body = _render_section(tpl, {
        "name": name,
        "gender_text": gender_text,
        "age_text": age_text,
        "mbti_line": mbti_line or "中性",
        "mbti_detail": detail,
        "style_rules": style,
    })
    if not _has_prompt_body(body):
        return None
    return _PromptBody(body, "chat.personality_section")


async def _build_emotion_section(
    user_emotion: dict | None = None,
    intimacy_stage: str | None = None,
) -> _PromptBody | None:
    """Only inject intimacy stage; runtime emotion vectors have been removed."""
    if not intimacy_stage:
        return None

    tpl = await _get_optional_prompt("chat.relationship_stage_section")
    if tpl is None:
        return None
    parts: list[str] = [
        _render_section(tpl, {"intimacy_stage": intimacy_stage})
    ]

    return _PromptBody("\n".join(parts), "chat.relationship_stage_section")


    # (core_memory permanent injection removed — spec §3 uses retrieval only)


async def _build_memory_section(
    memories: list[ClassifiedMemory] | None,
    *,
    include_empty_anchor: bool = True,
) -> _PromptBody | None:
    """按 owner 分两段渲染. 见 ClassifiedMemory.source 分组原因.

    即便 memories 为空 (弱路径不调记忆 / 强中路径召回为空) 也注入空 section,
    给 chat.consistency_rules 的反幻觉规则一个可靠锚点 — LLM 看到 "(本次没有
    联想到任何相关记忆)" 就明白搜过了没有, 可以柔和拒绝用户的预设性问句, 而不是
    顺承编造. 详见 CLAUDE.md 偏离表对应章节.
    """
    if not memories:
        if not include_empty_anchor:
            return None
        anchor = await _get_optional_prompt("chat.memory_empty_anchor")
        if anchor is None:
            return None
        return _PromptBody(str(anchor), "chat.memory_empty_anchor")

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

    user_memory_items = [
        (m, _format_memory(m)) for m in memories if m.source != "ai"
    ]
    user_texts = [text for _, text in user_memory_items]
    ai_texts = [_format_memory(m) for m in memories if m.source == "ai"]
    named_relation_texts = [
        text for m, text in user_memory_items
        if any(
            reason.startswith("保护槽:关系命名")
            for reason in (m.rank_reasons or [])
        )
    ]
    literal_task_texts = [
        text for m, text in user_memory_items
        if text not in named_relation_texts and any(
            reason.startswith("保护槽:字面命中")
            or reason.startswith("保护槽:当前问题事实")
            for reason in (m.rank_reasons or [])
        )
    ]
    task_user_texts = named_relation_texts + literal_task_texts
    safety_user_texts = [
        text for m, text in user_memory_items
        if any(
            reason.startswith("保护槽:安全情绪") or reason == "安全/情绪相关"
            for reason in (m.rank_reasons or [])
        ) and text not in task_user_texts
    ]
    user_profile_context_texts = [
        text for m, text in user_memory_items
        if (
            "AI资料查询:用户同类资料" in (m.rank_reasons or [])
            and text not in task_user_texts
            and text not in safety_user_texts
        )
    ]
    other_user_texts = [
        text for text in user_texts
        if (
            text not in task_user_texts
            and text not in safety_user_texts
            and text not in user_profile_context_texts
        )
    ]

    def _numbered(label: str, items: list[str]) -> str:
        body = "\n".join(f"{i}. {t}" for i, t in enumerate(items, 1))
        return f"{label}\n{body}"

    # 分组标签是结构性 glue (chat.memory_label_*): 停用/清空标签绝不能连带
    # 丢弃已检索出的记忆本身 (尤其【安全 / 情绪背景】组), 因此走 or_default —
    # 停用退回代码默认标签文案, 记忆数据照常注入. 分组顺序与 label_keys 严格对应.
    label_keys_and_texts = [
        ("chat.memory_label_named_relation", named_relation_texts),
        ("chat.memory_label_literal_task", literal_task_texts),
        ("chat.memory_label_safety", safety_user_texts),
        ("chat.memory_label_profile_context", user_profile_context_texts),
        ("chat.memory_label_other", other_user_texts),
        ("chat.memory_label_ai_self", ai_texts),
    ]
    tpl, *labels = await asyncio.gather(
        _get_optional_prompt("chat.memory_section_body"),
        *(get_prompt_text_or_default(key) for key, _ in label_keys_and_texts),
    )
    if tpl is None:
        return None

    parts: list[str] = []
    for (_, texts), label in zip(label_keys_and_texts, labels):
        if texts:
            parts.append(_numbered(str(label), texts))

    if not parts:
        return None

    body = _render_section(tpl, {"memory_groups": "\n\n".join(parts)})
    return _PromptBody(body, "chat.memory_section_body")


async def _build_delay_context_section(
    delay_context: dict[str, Any] | None,
) -> _PromptBody | None:
    """Build the delayed-reply explanation section (spec §6, <1min 延迟).

    只接受结构化 dict (received_at/activity/status/delay_seconds/delay_reason),
    段模板 chat.delay_context_section 由 registry 管理. 不接受预拼 str —
    那会让该段脱离 registry/trace 编辑, 违背"所有 section 走 registry"不变量.
    """
    if not delay_context:
        return None
    tpl = await _get_optional_prompt("chat.delay_context_section")
    if tpl is None:
        return None
    # delay_reason 被停用时为空串, render_template 会剔除该空行.
    body = render_template(tpl, dict(delay_context), optional_keys=["delay_reason"])
    return _PromptBody(body, "chat.delay_context_section")


async def _build_portrait_section(portrait: str | None) -> _PromptBody | None:
    """Build the user portrait section (模板包装 registry 管理)."""
    if not portrait:
        return None
    tpl = await _get_optional_prompt("chat.portrait_section")
    if tpl is None:
        return None
    return _PromptBody(_render_section(tpl, {"portrait": portrait}), "chat.portrait_section")


async def _build_topic_context_section(
    topic_context: dict[str, Any] | None,
) -> _PromptBody | None:
    """Build the topic context section (防话题跳跃).

    只接受 push_topic 返回的 dict ({category, turns}), 由
    chat.topic_context_section 模板渲染 (registry 管理, trace 内可编辑).
    """
    if not topic_context:
        return None
    tpl = await _get_optional_prompt("chat.topic_context_section")
    if tpl is None:
        return None
    body = _render_section(tpl, {
        "topic_category": topic_context.get("category", ""),
        "topic_turns": topic_context.get("turns", 1),
    })
    return _PromptBody(body, "chat.topic_context_section")


async def _build_time_context_section(time_context: str | None) -> _PromptBody | None:
    """Build the time section (时间日期系统数据文本 + registry 包装模板)."""
    if not time_context:
        return None
    tpl = await _get_optional_prompt("chat.time_context_section")
    if tpl is None:
        return None
    return _PromptBody(_render_section(tpl, {"time_context": time_context}), "chat.time_context_section")


async def _build_music_context_section(music_context: str | None) -> _PromptBody | None:
    """Build the co-listening section (music.co_listening_context 渲染结果 + 包装模板)."""
    if not music_context:
        return None
    tpl = await _get_optional_prompt("chat.music_context_section")
    if tpl is None:
        return None
    return _PromptBody(_render_section(tpl, {"music_context": music_context}), "chat.music_context_section")


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
    delay_context: dict[str, Any] | None = None,
    portrait: str | None = None,
    topic_context: dict[str, Any] | None = None,
    music_context: str | None = None,
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
    diagnostics: dict[str, Any] | None = None,
    canary_user_id: str | None = None,
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
    # admin 停用任一模板 → 返回 None → 对应 section 从最终输入中彻底移除.
    agent_id = str(getattr(agent, "id", "") or "") or None
    system_base, consistency_rules, response_instruction, anti_hallucination = await asyncio.gather(
        _get_optional_prompt("chat.system_base", agent_id=agent_id, user_id=canary_user_id),
        _get_optional_prompt("chat.consistency_rules", agent_id=agent_id, user_id=canary_user_id),
        _get_optional_prompt("chat.response_instruction", agent_id=agent_id, user_id=canary_user_id),
        _get_optional_prompt(
            "chat.anti_hallucination_hard_rule",
            agent_id=agent_id,
            user_id=canary_user_id,
        ),
    )

    # ═══ STABLE PREFIX (cache 命中区) ════════════════════════════════════
    # 同 agent 跨请求字节级一致, dashscope prefix cache 应命中.
    sections: list[str] = []
    components: list[dict[str, Any]] = []
    if system_base is not None:
        _append_section(
            sections, components, "核心规则", str(system_base),
            prompt_key="chat.system_base",
        )
    else:
        _record_skipped_section(diagnostics, "核心规则")
    anti_hallucination_body = str(anti_hallucination).strip() if anti_hallucination is not None else ""
    anti_hallucination_section = anti_hallucination_body if _has_prompt_body(anti_hallucination_body) else None
    if anti_hallucination_section:
        _append_section(
            sections, components, "反幻觉硬约束", anti_hallucination_section,
            prompt_key="chat.anti_hallucination_hard_rule",
        )
    else:
        _record_skipped_section(diagnostics, "反幻觉硬约束")
    personality = await _build_personality_section(agent)   # per-agent 稳定
    if personality:
        _append_section(
            sections, components, "你的身份", personality.body,
            prompt_key=personality.prompt_key,
        )
    else:
        _record_skipped_section(diagnostics, "你的身份")
    consistency_body = str(consistency_rules).strip() if consistency_rules is not None else ""
    consistency_section = consistency_body if _has_prompt_body(consistency_body) else None
    if consistency_section:
        _append_section(
            sections, components, "对话一致性", consistency_section,
            prompt_key="chat.consistency_rules",
        )
    else:
        _record_skipped_section(diagnostics, "对话一致性")

    # ═══ VARIABLE SUFFIX (每请求变化, cache miss 起点) ═══════════════════

    emo = await _build_emotion_section(user_emotion, intimacy_stage)
    if emo:
        _append_section(
            sections, components, "当前情绪", emo.body,
            prompt_key=emo.prompt_key,
        )
    else:
        _record_skipped_section(diagnostics, "当前情绪")

    port = await _build_portrait_section(portrait)
    if port:
        _append_section(
            sections, components, "用户画像", port.body,
            prompt_key=port.prompt_key,
        )
    else:
        _record_skipped_section(diagnostics, "用户画像")

    delay = await _build_delay_context_section(delay_context)
    if delay:
        _append_section(
            sections, components, "回复时机说明", delay.body,
            prompt_key=delay.prompt_key,
        )
    else:
        _record_skipped_section(diagnostics, "回复时机说明")

    # Phase 6: 删 relational_context 注入 (实证冗余 SYSTEM_BASE)

    mem = await _build_memory_section(
        memories,
        include_empty_anchor=(
            memory_relevance != "weak" and anti_hallucination_section is not None
        ),
    )
    if mem:
        _append_section(
            sections, components, "你记得的事情", mem.body,
            prompt_key=mem.prompt_key,
        )
    else:
        _record_skipped_section(diagnostics, "你记得的事情")

    # Phase 6: 删 graph_context 注入 (信息冗余 memory section, 抽象列表诱导编造)

    topic = await _build_topic_context_section(topic_context)
    if topic:
        _append_section(
            sections, components, "话题上下文", topic.body,
            prompt_key=topic.prompt_key,
        )
    else:
        _record_skipped_section(diagnostics, "话题上下文")

    music = await _build_music_context_section(music_context)
    if music:
        _append_section(
            sections, components, "一起听音乐", music.body,
            prompt_key=music.prompt_key,
        )
    else:
        _record_skipped_section(diagnostics, "一起听音乐")

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
    time_section = await _build_time_context_section(time_context)
    if time_section:
        _append_section(
            sections, components, "时间", time_section.body,
            prompt_key=time_section.prompt_key,
        )
    else:
        _record_skipped_section(diagnostics, "时间")

    # 时间相关记忆
    time_mem_tpl = (
        await _get_optional_prompt("chat.time_memories_section") if time_memories else None
    )
    if time_memories and time_mem_tpl is not None:
        numbered = "\n".join(f"- {m}" for m in time_memories)
        _append_section(
            sections, components, "相关时间记忆",
            _render_section(time_mem_tpl, {"time_memories": numbered}),
            prompt_key="chat.time_memories_section",
        )
    else:
        _record_skipped_section(diagnostics, "相关时间记忆")

    # Spec §3.2 step 3: L3 distant memories (awakened only when relevant)
    l3_tpl = await _get_optional_prompt("chat.l3_memory_section") if l3_memories else None
    if l3_memories and l3_tpl is not None:
        l3_block = "\n".join(f"- {m}" for m in l3_memories)
        _append_section(
            sections, components, "久远记忆（L3）",
            _render_section(l3_tpl, {"l3_memories": l3_block}),
            prompt_key="chat.l3_memory_section",
        )
    else:
        _record_skipped_section(diagnostics, "久远记忆（L3）")

    # 5B.4: 耐心区间语气描述 (boundary.patience_instruction_* 渲染结果,
    # ManagedPromptText 自带 prompt_key → trace 内可编辑)
    if patience_instruction:
        _append_section(
            sections, components, "情绪状态提醒", str(patience_instruction),
            prompt_key=getattr(patience_instruction, "prompt_key", None),
        )
    else:
        _record_skipped_section(diagnostics, "情绪状态提醒")

    # AI 自洽性约束 (§4 主回复路径). 告诉 LLM 当前状态 + 禁止主动展开,
    # 防止 ≥1min 延迟主回复路径下 LLM 编造跟实际状态矛盾的活动. 详见
    # CHAT_AI_STATE_CONSTRAINT_PROMPT 注释 (defaults.py).
    ai_state_appended = False
    if ai_status:
        activity = str(ai_status.get("activity", "")).strip()
        status_label = str(ai_status.get("status", "idle")).strip()
        if activity:
            tpl = await _get_optional_prompt("chat.ai_state_constraint")
            if tpl is not None:
                _append_section(
                    sections, components, "你的隐性状态约束",
                    _render_section(tpl, {"activity": activity, "status": status_label}),
                    prompt_key="chat.ai_state_constraint",
                )
                ai_state_appended = True
    if not ai_state_appended:
        _record_skipped_section(diagnostics, "你的隐性状态约束")

    # 回复要求 (n=random 1-3 每轮变, 不可 cache, 排末尾)
    if response_instruction is not None:
        _append_section(
            sections, components, "回复要求",
            _render_section(response_instruction, {
                "n": reply_count, "total": reply_total, "max_per": _MAX_PER_REPLY,
            }),
            prompt_key="chat.response_instruction",
        )
    else:
        _record_skipped_section(diagnostics, "回复要求")

    if diagnostics is not None:
        skipped = diagnostics.get("empty_prompt_sections_removed")
        diagnostics["empty_prompt_sections_removed_count"] = (
            len(skipped) if isinstance(skipped, list) else 0
        )
        diagnostics["system_prompt_section_count"] = len(sections)

    system_prompt = "\n\n".join(sections)
    record_prompt_render(
        system_prompt,
        prompt_key="chat.system_base",
        components=components,
        source="chat.system_prompt",
    )
    return system_prompt


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
